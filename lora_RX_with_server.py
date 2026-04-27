#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# lora_RX_with_server.py  —  Multi-agent LoRa RX
#
# Each RX instance identifies itself by --agent-id and connects to the
# server on port RX_PORT_BASE + agent_id  (default: 5600 + agent_id).
#
# Per-slot protocol:
#   ← server: {"cmd":"start", "n_packets":N, "slot":t, "agent_id":i}
#   → server: {"status":"ready", "agent_id":i, "slot":t}
#   RX collects exactly N decoded packets (or timeout)
#   → server: {"type":"report", "agent_id":i, "avg_rssi_dbm":…, …}
#   RX blocks until next start command
#   ← server: {"cmd":"stop"}  → clean shutdown
#
# Deploy:
#   Agent 0:  python lora_RX_with_server.py --server 127.0.0.1 --agent-id 0
#   Agent 1:  python lora_RX_with_server.py --server 127.0.0.1 --agent-id 1

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

RX_PORT_BASE = 5600

# Map agent_id → B210 serial number and antenna port
RX_SERIALS = {
    0: ("30CD3F7", "RX2"),   # agent 0 RX B210, RF A RX2 port
    1: ("XXXXXXX", "RX2"),   # agent 1 RX B210 — replace with real serial
}

WINDOW_SIZE = 512
DETECT_MARGIN_DB = 6.0
NOISE_EMA_ALPHA = 0.01
MIN_SIGNAL_WINS = 3
NOISE_INIT_DBFS = -60.0


# ═════════════════════════════════════════════════════════════════════════════
# Level 1 — Threshold-gated raw IQ probe
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

        self.rx_gain_db = rx_gain_db
        self.noise_figure_db = noise_figure_db
        self.detect_margin_db = detect_margin_db
        self.alpha = noise_ema_alpha
        self.win_size = win_size
        self.min_signal_wins = min_signal_wins

        self._noise_ema_lin = 10 ** (NOISE_INIT_DBFS / 10.0)
        self._above_count = 0
        self._signal_acc = []

        self._lock = threading.Lock()
        self._noise_floor_dbfs = NOISE_INIT_DBFS
        self._signal_rssi_dbfs = NOISE_INIT_DBFS
        self._signal_rssi_dbm = NOISE_INIT_DBFS + rx_gain_db - noise_figure_db - 30.0
        self._raw_snr_db = 0.0
        self._is_signal_present = False
        self._burst_count = 0

    @property
    def noise_floor_dbfs(self):
        with self._lock:
            return self._noise_floor_dbfs

    @property
    def signal_rssi_dbfs(self):
        with self._lock:
            return self._signal_rssi_dbfs

    @property
    def signal_rssi_dbm(self):
        with self._lock:
            return self._signal_rssi_dbm

    @property
    def raw_snr_db(self):
        with self._lock:
            return self._raw_snr_db

    @property
    def is_signal_present(self):
        with self._lock:
            return self._is_signal_present

    @property
    def burst_count(self):
        with self._lock:
            return self._burst_count

    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n = len(inp)
        out[:n] = inp[:n]

        power = np.abs(inp[:n]) ** 2
        for start in range(0, n - self.win_size + 1, self.win_size):
            win = power[start: start + self.win_size]
            win_mean = float(np.mean(win))
            if win_mean <= 0:
                continue
            threshold = self._noise_ema_lin * (10 ** (self.detect_margin_db / 10.0))
            if win_mean < threshold:
                self._noise_ema_lin = (self.alpha * win_mean +
                                       (1 - self.alpha) * self._noise_ema_lin)
                if self._signal_acc:
                    with self._lock:
                        self._burst_count += 1
                    self._signal_acc = []
                self._above_count = 0
                with self._lock:
                    self._noise_floor_dbfs = 10.0 * math.log10(self._noise_ema_lin)
                    self._is_signal_present = False
            else:
                self._above_count += 1
                self._signal_acc.append(win_mean)
                if self._above_count >= self.min_signal_wins:
                    sig_mean = float(np.mean(self._signal_acc))
                    sig_dbfs = 10.0 * math.log10(sig_mean)
                    noise_dbfs = 10.0 * math.log10(self._noise_ema_lin)
                    sig_dbm = sig_dbfs + self.rx_gain_db - self.noise_figure_db - 30.0
                    with self._lock:
                        self._signal_rssi_dbfs = sig_dbfs
                        self._signal_rssi_dbm = sig_dbm
                        self._raw_snr_db = sig_dbfs - noise_dbfs
                        self._noise_floor_dbfs = noise_dbfs
                        self._is_signal_present = True
        return n


# ═════════════════════════════════════════════════════════════════════════════
# Slot collector — accumulates N packets per slot then triggers report
# ═════════════════════════════════════════════════════════════════════════════
class SlotCollector(gr.basic_block):
    """
    Connected to crc_verif 'msg' port.
    Discards packets when IDLE; accumulates when ARMED.
    Reports averaged RSSI/SNR when n_target packets received.
    """

    def __init__(self, probe, rx_gain_db=10.0, agent_id=0):
        gr.basic_block.__init__(self, name="SlotCollector",
                                in_sig=None, out_sig=None)
        self.probe = probe
        self.rx_gain_db = rx_gain_db
        self.agent_id = agent_id

        self._lock = threading.Lock()
        self._armed = False
        self._n_target = 0
        self._records = []
        self._done_ev = threading.Event()
        self._slot = -1

        self.message_port_register_in(pmt.intern("msg_in"))
        self.set_msg_handler(pmt.intern("msg_in"), self._handle_msg)

    def arm(self, n_target: int, slot: int):
        with self._lock:
            self._armed = True
            self._n_target = n_target
            self._records = []
            self._slot = slot
        self._done_ev.clear()
        print(f"[RX{self.agent_id}] Collector armed: slot={slot}  "
              f"n_target={n_target}")

    def wait_complete(self, timeout: float) -> bool:
        return self._done_ev.wait(timeout=timeout)

    def get_report(self) -> dict:
        with self._lock:
            records = list(self._records)
            slot = self._slot

        if not records:
            return {
                "type": "report", "agent_id": self.agent_id,
                "slot": slot, "n_packets": 0,
                "avg_rssi_dbm": -100.0, "avg_snr_db": 0.0,
                "avg_rssi_dbfs": -100.0,
                "avg_arduino_rssi": -100.0, "avg_arduino_snr": 0.0,
                "noise_floor_dbfs": -60.0,
                "timestamp": datetime.now().isoformat(),
            }

        n = len(records)
        avg_rssi_dbfs = sum(r['rssi_dbfs'] for r in records) / n
        avg_rssi_dbm = sum(r['rssi_dbm'] for r in records) / n
        avg_snr_db = sum(r['snr_db'] for r in records) / n
        avg_arduino_rssi = sum(r['arduino_rssi'] for r in records) / n
        avg_arduino_snr = sum(r['arduino_snr'] for r in records) / n
        noise_floor = self.probe.noise_floor_dbfs

        return {
            "type": "report",
            "agent_id": self.agent_id,
            "slot": slot,
            "n_packets": n,
            "avg_rssi_dbfs": round(avg_rssi_dbfs,    2),
            "avg_rssi_dbm": round(avg_rssi_dbm,     2),
            "avg_snr_db": round(avg_snr_db,       2),
            "avg_arduino_rssi": round(avg_arduino_rssi, 2),
            "avg_arduino_snr": round(avg_arduino_snr,  2),
            "noise_floor_dbfs": round(noise_floor,      2),
            "payloads": [r['payload'] for r in records],
            "timestamp": datetime.now().isoformat(),
        }

    def _handle_msg(self, msg):
        with self._lock:
            if not self._armed:
                return

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

        rssi_dbfs = self.probe.signal_rssi_dbfs
        rssi_dbm = self.probe.signal_rssi_dbm
        snr_db = self.probe.raw_snr_db
        noise_dbfs = self.probe.noise_floor_dbfs

        # Arduino SX1276 AN1200.22 formula
        pkt_rssi_reg = rssi_dbm + 157.0
        pkt_snr_reg = int(snr_db * 4)
        if snr_db >= 0:
            arduino_rssi = -157.0 + (16.0 / 15.0) * pkt_rssi_reg
        else:
            arduino_rssi = -157.0 + pkt_rssi_reg + pkt_snr_reg * 0.25
        arduino_snr = pkt_snr_reg * 0.25

        record = {
            "payload": payload,
            "rssi_dbfs": rssi_dbfs,
            "rssi_dbm": rssi_dbm,
            "noise_dbfs": noise_dbfs,
            "snr_db": snr_db,
            "arduino_rssi": arduino_rssi,
            "arduino_snr": arduino_snr,
        }

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
            n_target = self._n_target

        if n_collected >= n_target:
            with self._lock:
                self._armed = False
            self._done_ev.set()
            print(f"[RX{self.agent_id}] Slot complete — "
                  f"{n_collected}/{n_target} packets")


# ═════════════════════════════════════════════════════════════════════════════
# Server command handler
# ═════════════════════════════════════════════════════════════════════════════
class ServerCommandHandler:

    def __init__(self, tb, collector, server_host, server_port, agent_id):
        self.tb = tb
        self.collector = collector
        self.server_host = server_host
        self.server_port = server_port
        self.agent_id = agent_id
        self._conn = None
        self._running = False
        self._stop_flag = threading.Event()
        self._thread = threading.Thread(target=self._connect_loop,
                                            daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._stop_flag.set()
        if self._conn:
            try:
                self._conn.close()
            except:
                pass

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
                print(f"[RX{self.agent_id}] Connected to server")
                self._recv_loop(conn)
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"[RX{self.agent_id}] Connection failed: {e} — retry 5s")
                time.sleep(5)

    def _recv_loop(self, conn):
        buf = ""
        while self._running:
            try:
                data = conn.recv(4096).decode('utf-8', errors='replace')
                if not data:
                    break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            self._handle(json.loads(line), conn)
                        except json.JSONDecodeError:
                            print(f"[RX{self.agent_id}] bad JSON: {line[:60]}")
            except OSError:
                break
        print(f"[RX{self.agent_id}] Server disconnected")
        self._conn = None

    def _send(self, msg, conn):
        try:
            conn.sendall((json.dumps(msg) + '\n').encode('utf-8'))
        except OSError as e:
            print(f"[RX{self.agent_id}] send failed: {e}")

    def _handle(self, cmd, conn):
        c = cmd.get('cmd')
        print(f"[RX{self.agent_id}] ← server: {cmd}")

        if c == 'start':
            n_packets = int(cmd.get('n_packets', 10))
            slot = cmd.get('slot', -1)

            self.collector.arm(n_packets, slot)

            self._send({
                "status": "ready",
                "agent_id": self.agent_id,
                "slot": slot,
                "timestamp": datetime.now().isoformat(),
            }, conn)
            print(f"[RX{self.agent_id}] → server: ready  slot={slot}")

            timeout = n_packets * 2.0 + 30.0
            done = self.collector.wait_complete(timeout=timeout)
            if not done:
                print(f"[RX{self.agent_id}] Slot {slot} collection timeout "
                      f"— reporting partial data")

            report = self.collector.get_report()
            self._send(report, conn)
            print(f"[RX{self.agent_id}] → server: report  "
                  f"n={report['n_packets']}  "
                  f"rssi={report['avg_rssi_dbm']} dBm  "
                  f"snr={report['avg_snr_db']} dB")

        elif c == 'stop':
            print(f"[RX{self.agent_id}] Stop received — shutting down")
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
        self.probe = probe
        self.agent_id = agent_id
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

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
            p = self.probe
            pres = "  YES  " if p.is_signal_present else "  ---  "
            sdb = f"{p.signal_rssi_dbfs:>+10.2f}" if p.is_signal_present else f"{'---':>10}"
            sdm = f"{p.signal_rssi_dbm:>+10.2f}" if p.is_signal_present else f"{'---':>10}"
            snr = f"{p.raw_snr_db:>+8.2f}" if p.is_signal_present else f"{'---':>8}"
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
                 server_host='127.0.0.1', server_port=None,
                 agent_id=0):

        gr.top_block.__init__(self, f"LoRa RX{agent_id}")

        self.agent_id = agent_id
        self.center_freq = center_freq
        self.gain = gain
        self.samp_rate = samp_rate
        self._stop_event = threading.Event()

        if server_port is None:
            server_port = RX_PORT_BASE + agent_id

        sf = 7
        impl_head = False
        has_crc = True
        cr = 1
        bw = samp_rate
        sync_word = [0x12]
        ldro_mode = 2
        os_factor = 8
        preamble_len = 8

        serial, antenna = RX_SERIALS.get(agent_id, ("30CD3F7", "RX2"))

        # ── USRP RX ───────────────────────────────────────────────────────────
        # self.usrp = uhd.usrp_source(
        #     f"serial={serial}",
        #     uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
        # )
        self.usrp = uhd.usrp_source(
            "",
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
        )
        self.usrp.set_center_freq(center_freq, 0)
        self.usrp.set_gain(gain, 0)
        self.usrp.set_antenna(antenna, 0)
        self.usrp.set_samp_rate(samp_rate)
        self.usrp.set_time_unknown_pps(uhd.time_spec())

        # ── Level 1 probe ─────────────────────────────────────────────────────
        self.probe = SignalProbe(
            rx_gain_db=gain,
            noise_figure_db=noise_figure_db,
            detect_margin_db=detect_margin_db,
        )

        # ── FIR filter ────────────────────────────────────────────────────────
        self.fir = gr_filter.interp_fir_filter_ccf(
            os_factor,
            (-0.128616616593872, -0.212206590789194, -0.180063263231421,
              3.89817183251938e-17, 0.300105438719035, 0.636619772367581,
              0.900316316157106, 1.0, 0.900316316157106, 0.636619772367581,
              0.300105438719035, 3.89817183251938e-17, -0.180063263231421,
             -0.212206590789194, -0.128616616593872)
        )
        self.fir.declare_sample_delay(0)

        # ── LoRa demodulator chain ────────────────────────────────────────────
        self.frame_sync = lora_sdr.frame_sync(
            int(center_freq), int(bw), sf, impl_head,
            sync_word, os_factor, preamble_len
        )
        self.fft_demod = lora_sdr.fft_demod(False, False)
        self.gray_mapping = lora_sdr.gray_mapping(False)
        self.deinterleaver = lora_sdr.deinterleaver(False)
        self.hamming_dec = lora_sdr.hamming_dec(False)
        self.header_decoder = lora_sdr.header_decoder(
            impl_head, cr, payload_len, has_crc, ldro_mode, True
        )
        self.dewhitening = lora_sdr.dewhitening()
        self.crc_verif = lora_sdr.crc_verif(1, False)
        self.null_sink = blocks.null_sink(gr.sizeof_char)

        # ── Slot collector ────────────────────────────────────────────────────
        self.collector = SlotCollector(
            probe=self.probe, rx_gain_db=gain, agent_id=agent_id)

        # ── Stream connections ────────────────────────────────────────────────
        self.connect((self.usrp,            0), (self.probe,          0))
        self.connect((self.probe,            0), (self.fir,            0))
        self.connect((self.fir,              0), (self.frame_sync,     0))
        self.connect((self.frame_sync,       0), (self.fft_demod,      0))
        self.connect((self.fft_demod,        0), (self.gray_mapping,   0))
        self.connect((self.gray_mapping,     0), (self.deinterleaver,  0))
        self.connect((self.deinterleaver,    0), (self.hamming_dec,    0))
        self.connect((self.hamming_dec,      0), (self.header_decoder, 0))
        self.connect((self.header_decoder,   0), (self.dewhitening,    0))
        self.connect((self.dewhitening,      0), (self.crc_verif,      0))
        self.connect((self.crc_verif,        0), (self.null_sink,      0))

        # ── Message connections ───────────────────────────────────────────────
        self.msg_connect(
            (self.header_decoder, 'frame_info'),
            (self.frame_sync,     'frame_info')
        )
        self.msg_connect(
            (self.crc_verif,  'msg'),
            (self.collector,  'msg_in')
        )

        # ── Server command handler ────────────────────────────────────────────
        self._cmd = ServerCommandHandler(
            tb=self, collector=self.collector,
            server_host=server_host,
            server_port=server_port,
            agent_id=agent_id,
        )

        # ── Stats printer ─────────────────────────────────────────────────────
        self._printer = StatsPrinter(
            self.probe, agent_id=agent_id, interval=print_interval)

    def start(self):
        super().start()
        self._cmd.start()
        self._printer.start()
        serial, ant = RX_SERIALS.get(self.agent_id, ("?", "?"))
        print(f"\n{'═'*54}")
        print(f"  LoRa RX  agent_id={self.agent_id}")
        print(f"  Freq    : {self.center_freq/1e6:.3f} MHz")
        print(f"  Serial  : {serial}  ant: {ant}")
        print(f"  Port    : {RX_PORT_BASE + self.agent_id}")
        print(f"  Sync    : 0x12  —  idle, waiting for start command")
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
    p.add_argument("-l", "--payload-len", dest="payload_len",
                   type=int, default=21,
                   help="Expected payload length in bytes (default 21)")
    p.add_argument("--server",   dest="server_host", default="127.0.0.1")
    p.add_argument("--port",     dest="server_port", type=int, default=None,
                   help="Override server port (default: 5600 + agent-id)")
    p.add_argument("--agent-id", dest="agent_id",    type=int, default=0,
                   help="Agent index 0,1,2,... (default 0)")
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
        server_host=options.server_host,
        server_port=options.server_port,
        agent_id=options.agent_id,
    )

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print("Press Enter to quit early.")
    t = threading.Thread(target=tb.wait_for_stop, daemon=True)
    t.start()
    try:
        input()
    except EOFError:
        t.join(timeout=2)
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
