#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# lora_RX_with_server.py  —  Multi-agent LoRa RX with configurable LoRa params
#
# Changes from previous version:
#   - All LoRa parameters (SF, CR, os_factor, preamble_len, etc.) are CLI args
#   - Buffer size auto-scales with SF: max(2^sf * os_factor * 8, 65536)
#   - set_min_output_buffer called on correct variable names
#   - SlotCollector now tracks n_expected vs n_received for PER
#   - Report includes per_ratio = n_received / n_expected
#   - "" used for USRP device args
#
# Per-slot protocol (unchanged):
#   ← server: {"cmd":"start", "n_packets":N, "slot":t, "agent_id":i}
#   → server: {"status":"ready", "agent_id":i, "slot":t}
#   RX collects up to N packets (or timeout)
#   → server: {"type":"report", "agent_id":i,
#               "n_packets":N_received, "n_expected":N,
#               "per":PER,             ← new: packet error rate
#               "avg_rssi_dbm":…, "avg_snr_db":…, …}
#
# Deploy:
#   Agent 0:  python lora_RX_with_server.py --server 127.0.0.1 --agent-id 0
#   Agent 1:  python lora_RX_with_server.py --server 127.0.0.1 --agent-id 1 --sf 12

from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
from gnuradio import gr, blocks
import sys
import signal
import math
import threading
import time
import socket
import json
import numpy as np
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float
from gnuradio import uhd
from gnuradio import lora_sdr
from datetime import datetime
import pmt

RX_PORT_BASE     = 5600
WINDOW_SIZE      = 512
DETECT_MARGIN_DB = 6.0
NOISE_EMA_ALPHA  = 0.01
MIN_SIGNAL_WINS  = 3
NOISE_INIT_DBFS  = -60.0


# ═════════════════════════════════════════════════════════════════════════════
# Level 1 — Threshold-gated raw IQ probe (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
class SignalProbe(gr.sync_block):

    def __init__(self, rx_gain_db=10.0, noise_figure_db=6.0,
                 detect_margin_db=DETECT_MARGIN_DB,
                 noise_ema_alpha=NOISE_EMA_ALPHA,
                 win_size=WINDOW_SIZE,
                 min_signal_wins=MIN_SIGNAL_WINS):

        gr.sync_block.__init__(self, name="SignalProbe",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

        self.rx_gain_db       = rx_gain_db
        self.noise_figure_db  = noise_figure_db
        self.detect_margin_db = detect_margin_db
        self.alpha            = noise_ema_alpha
        self.win_size         = win_size
        self.min_signal_wins  = min_signal_wins

        self._noise_ema_lin = 10 ** (NOISE_INIT_DBFS / 10.0)
        self._above_count   = 0
        self._signal_acc    = []

        self._lock              = threading.Lock()
        self._noise_floor_dbfs  = NOISE_INIT_DBFS
        self._signal_rssi_dbfs  = NOISE_INIT_DBFS
        self._signal_rssi_dbm   = NOISE_INIT_DBFS + rx_gain_db - noise_figure_db - 30.0
        self._raw_snr_db        = 0.0
        self._is_signal_present = False
        self._burst_count       = 0

    @property
    def noise_floor_dbfs(self):
        with self._lock: return self._noise_floor_dbfs

    @property
    def signal_rssi_dbfs(self):
        with self._lock: return self._signal_rssi_dbfs

    @property
    def signal_rssi_dbm(self):
        with self._lock: return self._signal_rssi_dbm

    @property
    def raw_snr_db(self):
        with self._lock: return self._raw_snr_db

    @property
    def is_signal_present(self):
        with self._lock: return self._is_signal_present

    @property
    def burst_count(self):
        with self._lock: return self._burst_count

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n   = len(inp)
        out[:n] = inp[:n]

        power = np.abs(inp[:n]) ** 2
        for start in range(0, n - self.win_size + 1, self.win_size):
            win      = power[start : start + self.win_size]
            win_mean = float(np.mean(win))
            if win_mean <= 0: continue
            threshold = self._noise_ema_lin * (10 ** (self.detect_margin_db / 10.0))
            if win_mean < threshold:
                self._noise_ema_lin = (self.alpha * win_mean +
                                       (1 - self.alpha) * self._noise_ema_lin)
                if self._signal_acc:
                    with self._lock: self._burst_count += 1
                    self._signal_acc = []
                self._above_count = 0
                with self._lock:
                    self._noise_floor_dbfs  = 10.0 * math.log10(self._noise_ema_lin)
                    self._is_signal_present = False
            else:
                self._above_count += 1
                self._signal_acc.append(win_mean)
                if self._above_count >= self.min_signal_wins:
                    sig_mean   = float(np.mean(self._signal_acc))
                    sig_dbfs   = 10.0 * math.log10(sig_mean)
                    noise_dbfs = 10.0 * math.log10(self._noise_ema_lin)
                    sig_dbm    = sig_dbfs + self.rx_gain_db - self.noise_figure_db - 30.0
                    with self._lock:
                        self._signal_rssi_dbfs  = sig_dbfs
                        self._signal_rssi_dbm   = sig_dbm
                        self._raw_snr_db        = sig_dbfs - noise_dbfs
                        self._noise_floor_dbfs  = noise_dbfs
                        self._is_signal_present = True
        return n


# ═════════════════════════════════════════════════════════════════════════════
# Slot collector — now tracks n_expected for PER
# ═════════════════════════════════════════════════════════════════════════════
class SlotCollector(gr.basic_block):
    """
    Receives decoded packets from crc_verif 'msg' port.
    Discards when IDLE, accumulates when ARMED.

    PER support:
        arm(n_target) records n_expected = n_target.
        get_report() includes:
            n_packets  = actual received (CRC-valid) packets
            n_expected = packets TX was asked to send
            per        = (n_expected - n_packets) / n_expected
            goodput    = n_packets / n_expected  (= 1 - PER)
    """

    def __init__(self, probe, rx_gain_db=10.0, agent_id=0):
        gr.basic_block.__init__(self, name="SlotCollector",
                                in_sig=None, out_sig=None)
        self.probe      = probe
        self.rx_gain_db = rx_gain_db
        self.agent_id   = agent_id

        self._lock      = threading.Lock()
        self._armed     = False
        self._n_target  = 0
        self._records   = []
        self._done_ev   = threading.Event()
        self._slot      = -1

        self.message_port_register_in(pmt.intern("msg_in"))
        self.set_msg_handler(pmt.intern("msg_in"), self._handle_msg)

    def arm(self, n_target: int, slot: int):
        with self._lock:
            self._armed    = True
            self._n_target = n_target
            self._records  = []
            self._slot     = slot
        self._done_ev.clear()
        print(f"[RX{self.agent_id}] Collector armed: slot={slot}  "
              f"n_target={n_target}")

    def wait_complete(self, timeout: float) -> bool:
        return self._done_ev.wait(timeout=timeout)

    def get_report(self) -> dict:
        with self._lock:
            records    = list(self._records)
            slot       = self._slot
            n_expected = self._n_target

        n_received = len(records)

        # PER and goodput
        if n_expected > 0:
            per     = (n_expected - n_received) / n_expected
            goodput = n_received / n_expected
        else:
            per     = 0.0
            goodput = 0.0

        if not records:
            # No packets decoded successfully — return noise floor measurements
            # so the server still has a valid RSSI/SNR observation for the slot.
            noise_floor_dbfs = self.probe.noise_floor_dbfs
            noise_floor_dbm  = noise_floor_dbfs   # uncalibrated: dBm = dBFS + 0
            return {
                "type"            : "report",
                "agent_id"        : self.agent_id,
                "slot"            : slot,
                "n_packets"       : 0,
                "n_expected"      : n_expected,
                "per"             : 1.0,
                "goodput"         : 0.0,
                "avg_rssi_dbfs"   : round(noise_floor_dbfs, 2),
                "avg_rssi_dbm"    : round(noise_floor_dbm,  2),
                "avg_snr_db"      : 0.0,
                "avg_arduino_rssi": round(noise_floor_dbm,  2),
                "avg_arduino_snr" : 0.0,
                "noise_floor_dbfs": round(noise_floor_dbfs, 2),
                "source"          : "noise_floor",   # flag so server knows
                "timestamp"       : datetime.now().isoformat(),
            }

        n = len(records)
        avg_rssi_dbfs    = sum(r['rssi_dbfs']    for r in records) / n
        avg_rssi_dbm     = sum(r['rssi_dbm']     for r in records) / n
        avg_snr_db       = sum(r['snr_db']       for r in records) / n
        avg_arduino_rssi = sum(r['arduino_rssi'] for r in records) / n
        avg_arduino_snr  = sum(r['arduino_snr']  for r in records) / n

        return {
            "type"            : "report",
            "agent_id"        : self.agent_id,
            "slot"            : slot,
            "n_packets"       : n_received,
            "n_expected"      : n_expected,
            "per"             : round(per,     4),
            "goodput"         : round(goodput, 4),
            "avg_rssi_dbfs"   : round(avg_rssi_dbfs,    2),
            "avg_rssi_dbm"    : round(avg_rssi_dbm,     2),
            "avg_snr_db"      : round(avg_snr_db,       2),
            "avg_arduino_rssi": round(avg_arduino_rssi, 2),
            "avg_arduino_snr" : round(avg_arduino_snr,  2),
            "noise_floor_dbfs": round(self.probe.noise_floor_dbfs, 2),
            "payloads"        : [r['payload'] for r in records],
            "timestamp"       : datetime.now().isoformat(),
        }

    def _handle_msg(self, msg):
        with self._lock:
            if not self._armed: return

        try:
            if pmt.is_pair(msg):
                data = pmt.cdr(msg)
            else:
                data = msg
            if pmt.is_u8vector(data):
                payload = bytes(pmt.u8vector_elements(data)).decode(
                    'ascii', errors='replace').strip()
            elif pmt.is_symbol(data):
                payload = pmt.symbol_to_string(data).strip()
            else:
                payload = str(data).strip()
        except Exception as e:
            payload = f"<err:{e}>"

        pkt_agent_id = None
        if payload.startswith("AGT:"):
            try:
                parts        = payload.split(" ", 2)
                pkt_agent_id = int(parts[0][4:])
            except (ValueError, IndexError):
                pkt_agent_id = None

        if pkt_agent_id is None:
            # No AGT prefix — legacy packet, accept (single-agent fallback)
            # Do NOT print RSSI/SNR yet; fall through to measurement below
            print(f"[RX{self.agent_id}] No AGT prefix — accepting: {payload!r}")
        elif pkt_agent_id != self.agent_id:
            # Wrong agent — discard WITHOUT measuring or printing RSSI/SNR
            print(f"[RX{self.agent_id}] Discarding packet from "
                  f"agent {pkt_agent_id}: {payload!r}")
            return

        # ── Measurements — only reached after AGT match (or no-prefix fallback) ─
        # RSSI and SNR are read and printed only for packets that belong to
        # this receiver's agent_id. Packets from other agents are returned
        # above without touching the probe.
        rssi_dbfs  = self.probe.signal_rssi_dbfs
        rssi_dbm   = self.probe.signal_rssi_dbm
        snr_db     = self.probe.raw_snr_db
        noise_dbfs = self.probe.noise_floor_dbfs

        # Arduino SX1276 formula (Semtech AN1200.22 §3.5)
        pkt_rssi_reg = rssi_dbm + 157.0
        if snr_db >= 0:
            arduino_rssi = -157.0 + (16.0 / 15.0) * pkt_rssi_reg
        else:
            arduino_rssi = -157.0 + pkt_rssi_reg + snr_db
        arduino_snr = snr_db

        record = {
            "payload"     : payload,
            "rssi_dbfs"   : rssi_dbfs,
            "rssi_dbm"    : rssi_dbm,
            "snr_db"      : snr_db,
            "arduino_rssi": arduino_rssi,
            "arduino_snr" : arduino_snr,
        }

        # Print only after confirmed AGT match
        print(
            f"\n╔═ RX{self.agent_id} [{time.strftime('%H:%M:%S')}]  "
            f"Payload: {payload!r}"
            f"\n║  RSSI : {rssi_dbfs:+.1f} dBFS  ~{rssi_dbm:+.1f} dBm"
            f"\n║  SNR  : {snr_db:+.1f} dB"
            f"\n╚  Arduino RSSI={arduino_rssi:+.1f} dBm  "
            f"SNR={arduino_snr:+.1f} dB",
            flush=True
        )

        with self._lock:
            self._records.append(record)
            n_collected = len(self._records)
            n_target    = self._n_target

        if n_collected >= n_target:
            with self._lock:
                self._armed = False
            self._done_ev.set()
            print(f"[RX{self.agent_id}] Slot complete  "
                  f"received={n_collected}/{n_target}  "
                  f"PER={1 - n_collected/n_target:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# Server command handler
# ═════════════════════════════════════════════════════════════════════════════
class ServerCommandHandler:

    def __init__(self, tb, collector, server_host, server_port, agent_id,
                 sf=7):
        self.tb          = tb
        self.collector   = collector
        self.server_host = server_host
        self.server_port = server_port
        self.agent_id    = agent_id
        self.sf          = sf
        self._conn       = None
        self._running    = False
        self._stop_flag  = threading.Event()
        self._thread     = threading.Thread(target=self._connect_loop,
                                            daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop_flag.set()
        if self._conn:
            try: self._conn.close()
            except: pass

    def _connect_loop(self):
        while self._running and not self._stop_flag.is_set():
            try:
                print(f"[RX{self.agent_id}] Connecting to server "
                      f"{self.server_host}:{self.server_port} ...")
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5.0)
                conn.connect((self.server_host, self.server_port))
                conn.settimeout(None)
                self._conn = conn
                print(f"[RX{self.agent_id}] Connected")
                self._recv_loop(conn)
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"[RX{self.agent_id}] Connection failed: {e} — retry 5s")
                time.sleep(5)

    def _recv_loop(self, conn):
        buf = ""
        while self._running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='replace')
                if not data: break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:    self._handle(json.loads(line), conn)
                        except json.JSONDecodeError:
                            print(f"[RX{self.agent_id}] bad JSON: {line[:60]}")
            except OSError: break
        print(f"[RX{self.agent_id}] Server disconnected")
        self._conn = None

    def _send(self, msg, conn):
        try: conn.sendall((json.dumps(msg) + '\n').encode('utf-8'))
        except OSError as e: print(f"[RX{self.agent_id}] send failed: {e}")

    def _handle(self, cmd, conn):
        c = cmd.get('cmd')
        print(f"[RX{self.agent_id}] ← {cmd}")

        if c == 'start':
            n_packets = int(cmd.get('n_packets', 10))
            slot      = cmd.get('slot', -1)
            # period_ms tells us how long TX waits between packets
            # Use it to compute a realistic collection timeout
            period_ms = int(cmd.get('period_ms', 1000))

            self.collector.arm(n_packets, slot)

            self._send({
                "status"   : "ready",
                "agent_id" : self.agent_id,
                "slot"     : slot,
                "timestamp": datetime.now().isoformat(),
            }, conn)
            print(f"[RX{self.agent_id}] ready  slot={slot}")

            # Timeout = single burst duration + 60s margin.
            # start_delay_ms=0 for all agents so all TX fire simultaneously;
            # the full slot window is n_packets * period_ms.
            n_agents = int(cmd.get('n_agents', 1))
            timeout  = n_packets * (period_ms / 1000.0) + 60.0
            done    = self.collector.wait_complete(timeout=timeout)

            if not done:
                print(f"[RX{self.agent_id}] Timeout — reporting partial data")

            report = self.collector.get_report()
            self._send(report, conn)
            print(f"[RX{self.agent_id}] report  "
                  f"n_received={report['n_packets']}/{report['n_expected']}  "
                  f"PER={report['per']:.3f}  "
                  f"goodput={report['goodput']:.3f}  "
                  f"rssi={report['avg_rssi_dbm']} dBm  "
                  f"snr={report['avg_snr_db']} dB")

        elif c == 'stop':
            print(f"[RX{self.agent_id}] Stop — shutting down")
            self._running = False
            self._stop_flag.set()
            self.tb.request_stop()

        else:
            print(f"[RX{self.agent_id}] Unknown cmd: {c!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Periodic raw IQ stats printer
# ═════════════════════════════════════════════════════════════════════════════
class StatsPrinter:
    def __init__(self, probe, agent_id=0, interval=5.0):
        self.probe    = probe
        self.agent_id = agent_id
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self): self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        hdr = (f"\n[RX{self.agent_id}] "
               f"{'Time':>8}  {'Noise(dBFS)':>11}  "
               f"{'Sig(dBFS)':>10}  {'Sig(dBm~)':>10}  "
               f"{'SNR(dB)':>8}  {'Present':>7}  {'Bursts':>6}")
        print(hdr)
        print("─" * (len(hdr) - 1))
        while not self._stop.is_set():
            time.sleep(self.interval)
            p    = self.probe
            pres = "  YES  " if p.is_signal_present else "  ---  "
            sdb  = f"{p.signal_rssi_dbfs:>+10.2f}" if p.is_signal_present else f"{'---':>10}"
            sdm  = f"{p.signal_rssi_dbm:>+10.2f}"  if p.is_signal_present else f"{'---':>10}"
            snr  = f"{p.raw_snr_db:>+8.2f}"         if p.is_signal_present else f"{'---':>8}"
            print(f"[RX{self.agent_id}] "
                  f"{time.strftime('%H:%M:%S'):>8}  "
                  f"{p.noise_floor_dbfs:>+11.2f}  "
                  f"{sdb}  {sdm}  {snr}  {pres}  {p.burst_count:>6d}",
                  flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main flowgraph
# ═════════════════════════════════════════════════════════════════════════════
class lora_RX(gr.top_block):

    def __init__(self, center_freq=915000000, gain=10, samp_rate=250000,
                 noise_figure_db=6.0, detect_margin_db=DETECT_MARGIN_DB,
                 print_interval=5.0, payload_len=21,
                 sf=7, cr=1, os_factor=8, preamble_len=8,
                 has_crc=True, impl_head=False,
                 ldro_mode=2, sync_word=0x12,
                 server_host='127.0.0.1', server_port=None,
                 agent_id=0):

        gr.top_block.__init__(self, f"LoRa RX{agent_id}")

        self.agent_id    = agent_id
        self.center_freq = center_freq
        self.gain        = gain
        self.samp_rate   = samp_rate
        self._stop_event = threading.Event()

        if server_port is None:
            server_port = RX_PORT_BASE + agent_id

        bw             = samp_rate
        sync_word_list = [sync_word]

        # Buffer size auto-scales with SF and os_factor
        buf_size = max(2 ** sf * os_factor * 8, 65536)

        print(f"\n[RX{agent_id}] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  OS={os_factor}  preamble={preamble_len}")
        print(f"  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  payload_len={payload_len}")
        print(f"  freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  buf_size={buf_size}\n")

        # ── USRP RX — empty string = use whatever device is found ─────────────
        self.uhd_usrp_source_0 = uhd.usrp_source(
            "",
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
        )
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        self.uhd_usrp_source_0.set_min_output_buffer(buf_size)

        # ── Level 1 probe ─────────────────────────────────────────────────────
        self.probe = SignalProbe(
            rx_gain_db=gain,
            noise_figure_db=noise_figure_db,
            detect_margin_db=detect_margin_db,
        )

        # ── FIR filter ────────────────────────────────────────────────────────
        self.interp_fir_filter_xxx_0 = gr_filter.interp_fir_filter_ccf(
            os_factor,
            (-0.128616616593872, -0.212206590789194, -0.180063263231421,
              3.89817183251938e-17, 0.300105438719035, 0.636619772367581,
              0.900316316157106, 1.0, 0.900316316157106, 0.636619772367581,
              0.300105438719035, 3.89817183251938e-17, -0.180063263231421,
             -0.212206590789194, -0.128616616593872)
        )
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.interp_fir_filter_xxx_0.set_min_output_buffer(buf_size)

        # ── LoRa demodulator chain ────────────────────────────────────────────
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(center_freq), int(bw), sf, impl_head,
            sync_word_list, os_factor, preamble_len
        )
        self.lora_sdr_frame_sync_0.set_min_output_buffer(buf_size)

        self.lora_sdr_fft_demod_0     = lora_sdr.fft_demod(False, False)
        self.lora_sdr_gray_mapping_0  = lora_sdr.gray_mapping(False)
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver(False)
        self.lora_sdr_hamming_dec_0   = lora_sdr.hamming_dec(False)
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(
            impl_head, cr, payload_len, has_crc, ldro_mode, True
        )
        self.lora_sdr_dewhitening_0   = lora_sdr.dewhitening()
        self.lora_sdr_crc_verif_0     = lora_sdr.crc_verif(1, False)
        self.null_sink                = blocks.null_sink(gr.sizeof_char)

        # ── Slot collector (Level 1 + Level 2 + PER) ──────────────────────────
        self.collector = SlotCollector(
            probe=self.probe, rx_gain_db=gain, agent_id=agent_id)

        # ── Stream connections ────────────────────────────────────────────────
        self.connect((self.uhd_usrp_source_0,          0), (self.probe,                    0))
        self.connect((self.probe,                       0), (self.interp_fir_filter_xxx_0,  0))
        self.connect((self.interp_fir_filter_xxx_0,     0), (self.lora_sdr_frame_sync_0,    0))
        self.connect((self.lora_sdr_frame_sync_0,       0), (self.lora_sdr_fft_demod_0,     0))
        self.connect((self.lora_sdr_fft_demod_0,        0), (self.lora_sdr_gray_mapping_0,  0))
        self.connect((self.lora_sdr_gray_mapping_0,     0), (self.lora_sdr_deinterleaver_0, 0))
        self.connect((self.lora_sdr_deinterleaver_0,    0), (self.lora_sdr_hamming_dec_0,   0))
        self.connect((self.lora_sdr_hamming_dec_0,      0), (self.lora_sdr_header_decoder_0,0))
        self.connect((self.lora_sdr_header_decoder_0,   0), (self.lora_sdr_dewhitening_0,   0))
        self.connect((self.lora_sdr_dewhitening_0,      0), (self.lora_sdr_crc_verif_0,     0))
        self.connect((self.lora_sdr_crc_verif_0,        0), (self.null_sink,                0))

        # ── Message connections ───────────────────────────────────────────────
        self.msg_connect(
            (self.lora_sdr_header_decoder_0, 'frame_info'),
            (self.lora_sdr_frame_sync_0,     'frame_info')
        )
        self.msg_connect(
            (self.lora_sdr_crc_verif_0, 'msg'),
            (self.collector,            'msg_in')
        )

        # ── Server command handler ────────────────────────────────────────────
        self._cmd = ServerCommandHandler(
            tb=self, collector=self.collector,
            server_host=server_host,
            server_port=server_port,
            agent_id=agent_id,
            sf=sf,
        )

        # ── Stats printer ─────────────────────────────────────────────────────
        self._printer = StatsPrinter(
            self.probe, agent_id=agent_id, interval=print_interval)

    def start(self):
        super().start()
        self._cmd.start()
        self._printer.start()
        print(f"\n{'═'*54}")
        print(f"  LoRa RX  agent_id={self.agent_id}")
        print(f"  Port    : {RX_PORT_BASE + self.agent_id}")
        print(f"  Waiting for start command from server")
        print(f"{'═'*54}\n")

    def stop(self):
        self._printer.stop()
        self._cmd.stop()
        super().stop()

    def request_stop(self): self._stop_event.set()
    def wait_for_stop(self): self._stop_event.wait()


# ═════════════════════════════════════════════════════════════════════════════
def argument_parser():
    p = ArgumentParser(description="LoRa RX — MADDPG slot control")

    # RF
    p.add_argument("-f", "--center-freq", dest="center_freq",
                   type=eng_float, default="915.0M")
    p.add_argument("-g", "--gain", dest="gain",
                   type=eng_float, default="10.0")
    p.add_argument("-s", "--samp-rate", dest="samp_rate",
                   type=eng_float, default="250.0k")
    p.add_argument("-n", "--noise-figure", dest="noise_figure_db",
                   type=float, default=6.0)
    p.add_argument("-d", "--detect-margin", dest="detect_margin_db",
                   type=float, default=DETECT_MARGIN_DB)
    p.add_argument("-i", "--interval", dest="print_interval",
                   type=float, default=5.0)

    # LoRa modulation
    p.add_argument("--sf", dest="sf",
                   type=int, default=7, choices=range(7, 13),
                   help="Spreading factor 7-12 (default 7)")
    p.add_argument("--cr", dest="cr",
                   type=int, default=1, choices=[1, 2, 3, 4],
                   help="Coding rate 1-4 (default 1)")
    p.add_argument("--os-factor", dest="os_factor",
                   type=int, default=8, choices=[1, 2, 4, 8],
                   help="Oversampling factor (default 8, use 1 for SF>=10)")
    p.add_argument("--preamble-len", dest="preamble_len",
                   type=int, default=8,
                   help="Preamble symbols (default 8, use 12 for SF>=10)")
    p.add_argument("-l", "--payload-len", dest="payload_len",
                   type=int, default=21,
                   help="Expected payload length bytes (default 21)")
    p.add_argument("--has-crc", dest="has_crc",
                   action="store_true", default=True,
                   help="Enable CRC (default True)")
    p.add_argument("--no-crc", dest="has_crc",
                   action="store_false",
                   help="Disable CRC")
    p.add_argument("--impl-head", dest="impl_head",
                   action="store_true", default=False,
                   help="Implicit header mode (default False)")
    p.add_argument("--ldro", dest="ldro_mode",
                   type=int, default=2, choices=[0, 1, 2],
                   help="LDRO 0=off 1=on 2=auto (default 2)")
    p.add_argument("--sync-word", dest="sync_word",
                   type=lambda x: int(x, 0), default=0x12,
                   help="Sync word hex (default 0x12)")

    # Server
    p.add_argument("--server",   dest="server_host", default="127.0.0.1")
    p.add_argument("--port",     dest="server_port", type=int, default=None,
                   help="Override server port (default 5600+agent_id)")
    p.add_argument("--agent-id", dest="agent_id",    type=int, default=0,
                   help="Agent index (default 0)")
    return p


def main(options=None):
    if options is None:
        options = argument_parser().parse_args()

    tb = lora_RX(
        center_freq=options.center_freq,
        gain=options.gain,
        samp_rate=options.samp_rate,
        noise_figure_db=options.noise_figure_db,
        detect_margin_db=options.detect_margin_db,
        print_interval=options.print_interval,
        payload_len=options.payload_len,
        sf=options.sf,
        cr=options.cr,
        os_factor=options.os_factor,
        preamble_len=options.preamble_len,
        has_crc=options.has_crc,
        impl_head=options.impl_head,
        ldro_mode=options.ldro_mode,
        sync_word=options.sync_word,
        server_host=options.server_host,
        server_port=options.server_port,
        agent_id=options.agent_id,
    )

    def sig_handler(sig=None, frame=None):
        tb.stop(); tb.wait(); sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print("Press Enter to quit early.")
    t = threading.Thread(target=tb.wait_for_stop, daemon=True)
    t.start()
    try: input()
    except EOFError: t.join(timeout=2)
    tb.stop(); tb.wait()


if __name__ == '__main__':
    main()
