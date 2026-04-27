#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# lora_TX_with_server.py  —  Multi-agent LoRa TX with configurable LoRa params
#
# Changes from previous version:
#   - All LoRa parameters (SF, CR, os_factor, preamble_len, etc.) are CLI args
#   - frame_zero_padd scales automatically with SF: 2^(sf-1)
#   - SlotStrobe tracks n_sent accurately for PER computation
#   - burst_done report includes n_sent so server can compute PER
#   - period_ms from server command overrides CLI default per slot
#   - "" used for USRP device args (find whatever is connected)
#
# Per-slot protocol (unchanged):
#   ← server: {"cmd":"set_gain", "gain":G, "n_packets":N, "slot":t,
#               "agent_id":i, "period_ms":M}
#   → server: {"status":"ready",      "agent_id":i, "slot":t}
#   TX sends exactly N packets with M ms interval
#   → server: {"status":"burst_done", "agent_id":i, "slot":t,
#               "n_sent":N_actual}   ← accurate count for PER
#   ← server: {"cmd":"stop"}  → clean shutdown
#
# Deploy:
#   Agent 0:  python lora_TX_with_server.py --server 127.0.0.1 --agent-id 0
#   Agent 1:  python lora_TX_with_server.py --server 127.0.0.1 --agent-id 1 --sf 12

import pmt
from gnuradio import gr
import sys
import signal
import time
import threading
import socket
import json
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float
from gnuradio import uhd
from gnuradio import lora_sdr
from datetime import datetime

TX_PORT_BASE = 5555


# ═════════════════════════════════════════════════════════════════════════════
# Slot-controlled strobe  —  now tracks n_sent accurately for PER
# ═════════════════════════════════════════════════════════════════════════════
class SlotStrobe(gr.basic_block):
    """
    Sends exactly n_packets per slot when arm() is called, then blocks.

    PER support:
        self.n_sent  is set to the actual number of packets dispatched
        after each burst, so the server can compare n_sent vs n_received
        to compute Packet Error Rate.

    The period between packets (frame_period_ms) is set per slot by
    arm() so the server can pass SF-appropriate intervals each time.
    """
    def __init__(self, frame_period_ms=1000):
        gr.basic_block.__init__(self, name="SlotStrobe",
                                in_sig=None, out_sig=None)
        self._period_ms = frame_period_ms
        self._seq       = 0
        self._n_sent    = 0          # accurate count for PER
        self._lock      = threading.Lock()
        self._armed     = threading.Event()
        self._done_ev   = threading.Event()
        self._stop      = threading.Event()
        self._n         = 0
        self._message   = "hello world"
        self._thread    = None
        self.message_port_register_out(pmt.intern("strobe"))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._armed.set()
        return True

    def arm(self, n_packets: int, message: str = None,
            frame_period_ms: int = None):
        """Arm for one burst. period_ms overrides CLI default if provided."""
        with self._lock:
            self._n      = n_packets
            self._n_sent = 0
            if message:         self._message = message
            if frame_period_ms: self._period_ms = frame_period_ms
        self._done_ev.clear()
        self._armed.set()

    def wait_done(self, timeout=None) -> bool:
        return self._done_ev.wait(timeout=timeout)

    @property
    def sequence_number(self):
        with self._lock: return self._seq

    @property
    def n_sent(self):
        """Actual packets sent in the last burst — used for PER."""
        with self._lock: return self._n_sent

    def _run(self):
        time.sleep(1.0)
        while not self._stop.is_set():
            self._armed.wait()
            self._armed.clear()
            if self._stop.is_set(): break

            with self._lock:
                n          = self._n
                message    = self._message
                period_ms  = self._period_ms

            sent = 0
            while sent < n and not self._stop.is_set():
                with self._lock:
                    seq = self._seq
                    self._seq = (self._seq + 1) % 100000

                payload = f"SEQ:{seq:05d} {message}"
                print(f"[TX] ({sent+1}/{n}) {payload!r}")
                self.message_port_pub(pmt.intern("strobe"),
                                      pmt.intern(payload))
                sent += 1
                if sent < n:
                    self._stop.wait(period_ms / 1000.0)

            with self._lock:
                self._n_sent = sent

            self._done_ev.set()
            print(f"[TX] Burst complete — {sent}/{n} packets sent")


# ═════════════════════════════════════════════════════════════════════════════
# Server command handler
# ═════════════════════════════════════════════════════════════════════════════
class ServerCommandHandler:

    def __init__(self, tb, strobe, server_host, server_port, agent_id,
                 default_period_ms=1000):
        self.tb                 = tb
        self.strobe             = strobe
        self.server_host        = server_host
        self.server_port        = server_port
        self.agent_id           = agent_id
        self.default_period_ms  = default_period_ms
        self._conn              = None
        self._running           = False
        self._stop_flag         = threading.Event()
        self._thread            = threading.Thread(target=self._connect_loop,
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
                print(f"[TX{self.agent_id}] Connecting to server "
                      f"{self.server_host}:{self.server_port} ...")
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(5.0)
                conn.connect((self.server_host, self.server_port))
                conn.settimeout(None)
                self._conn = conn
                print(f"[TX{self.agent_id}] Connected")
                self._recv_loop(conn)
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                print(f"[TX{self.agent_id}] Connection failed: {e} — retry 5s")
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
                            print(f"[TX{self.agent_id}] bad JSON: {line[:60]}")
            except OSError: break
        print(f"[TX{self.agent_id}] Server disconnected")
        self._conn = None

    def _send(self, msg, conn):
        try: conn.sendall((json.dumps(msg) + '\n').encode('utf-8'))
        except OSError as e: print(f"[TX{self.agent_id}] send failed: {e}")

    def _handle(self, cmd, conn):
        c = cmd.get('cmd')
        print(f"[TX{self.agent_id}] ← {cmd}")

        if c == 'set_gain':
            gain      = float(cmd.get('gain',      30))
            n_packets = int(cmd.get('n_packets',   10))
            slot      = cmd.get('slot', '?')
            message   = cmd.get('message',          None)
            # period_ms from server overrides local default
            # server sets this based on SF so TX waits long enough per packet
            period_ms = int(cmd.get('period_ms', self.default_period_ms))

            self.tb.set_tx_gain(gain)
            print(f"[TX{self.agent_id}] gain={gain} dB  "
                  f"n_packets={n_packets}  period={period_ms} ms")

            # Send ready ACK
            self._send({
                "status"   : "ready",
                "agent_id" : self.agent_id,
                "slot"     : slot,
                "gain"     : gain,
                "timestamp": datetime.now().isoformat(),
            }, conn)

            # Arm strobe and wait
            # Timeout = n_packets * period + generous margin
            timeout = n_packets * (period_ms / 1000.0) + 30.0
            self.strobe.arm(n_packets, message, period_ms)
            done = self.strobe.wait_done(timeout=timeout)

            # Report actual n_sent for PER computation
            self._send({
                "status"   : "burst_done",
                "agent_id" : self.agent_id,
                "slot"     : slot,
                "n_sent"   : self.strobe.n_sent,   # actual count
                "timestamp": datetime.now().isoformat(),
            }, conn)
            print(f"[TX{self.agent_id}] burst_done  "
                  f"n_sent={self.strobe.n_sent}/{n_packets}")

        elif c == 'stop':
            print(f"[TX{self.agent_id}] Stop — shutting down")
            self.strobe.stop()
            self._running = False
            self._stop_flag.set()
            self.tb.request_stop()

        else:
            print(f"[TX{self.agent_id}] Unknown cmd: {c!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Main flowgraph
# ═════════════════════════════════════════════════════════════════════════════
class lora_TX(gr.top_block):

    def __init__(self, center_freq=915000000, gain=30, samp_rate=250000,
                 user_message="hello world", frame_period=1000,
                 sf=7, cr=1, preamble_len=8,
                 has_crc=True, impl_head=False,
                 ldro_mode=2, sync_word=0x12,
                 server_host='127.0.0.1', server_port=None,
                 agent_id=0):

        gr.top_block.__init__(self, f"LoRa TX{agent_id}")

        self.agent_id    = agent_id
        self.center_freq = center_freq
        self.gain        = gain
        self.samp_rate   = samp_rate
        self._stop_event = threading.Event()

        if server_port is None:
            server_port = TX_PORT_BASE + agent_id

        bw             = 250000
        sync_word_list = [sync_word]

        # frame_zero_padd scales with SF — prevents TX overrun at high SF
        frame_zero_padd = 2 ** (sf - 1)

        print(f"\n[TX{agent_id}] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  preamble={preamble_len}")
        print(f"  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  frame_zero_padd={frame_zero_padd}")
        print(f"  freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  default period={frame_period} ms  "
              f"(server may override per slot)\n")

        # ── USRP TX — empty string = use whatever device is found ─────────────
        self.usrp = uhd.usrp_sink(
            "",
            uhd.stream_args(cpu_format="fc32", args='', channels=[0]),
            'frame_len',
        )
        self.usrp.set_center_freq(center_freq, 0)
        self.usrp.set_gain(gain, 0)
        self.usrp.set_antenna('TX/RX', 0)
        self.usrp.set_bandwidth(bw, 0)
        self.usrp.set_samp_rate(samp_rate)
        self.usrp.set_time_unknown_pps(uhd.time_spec())

        # ── Slot strobe ───────────────────────────────────────────────────────
        self.strobe = SlotStrobe(frame_period_ms=frame_period)

        # ── LoRa TX chain ─────────────────────────────────────────────────────
        self.whitening   = lora_sdr.whitening(False, False, ',', 'packet_len')
        self.modulate    = lora_sdr.modulate(
            sf, int(samp_rate), int(bw),
            sync_word_list, frame_zero_padd, preamble_len)
        self.modulate.set_min_output_buffer(10000000)
        self.interleaver = lora_sdr.interleaver(cr, sf, ldro_mode, int(bw))
        self.header      = lora_sdr.header(impl_head, has_crc, cr)
        self.hamming_enc = lora_sdr.hamming_enc(cr, sf)
        self.gray_demap  = lora_sdr.gray_demap(sf)
        self.add_crc     = lora_sdr.add_crc(has_crc)

        self.msg_connect((self.strobe,     'strobe'), (self.whitening, 'msg'))
        self.connect((self.whitening,   0), (self.header,      0))
        self.connect((self.header,      0), (self.add_crc,     0))
        self.connect((self.add_crc,     0), (self.hamming_enc, 0))
        self.connect((self.hamming_enc, 0), (self.interleaver, 0))
        self.connect((self.interleaver, 0), (self.gray_demap,  0))
        self.connect((self.gray_demap,  0), (self.modulate,    0))
        self.connect((self.modulate,    0), (self.usrp,        0))

        # ── Server command handler ────────────────────────────────────────────
        self._cmd = ServerCommandHandler(
            tb=self, strobe=self.strobe,
            server_host=server_host,
            server_port=server_port,
            agent_id=agent_id,
            default_period_ms=frame_period,
        )

    def start(self):
        super().start()
        self._cmd.start()
        print(f"\n{'═'*54}")
        print(f"  LoRa TX  agent_id={self.agent_id}")
        print(f"  Port    : {TX_PORT_BASE + self.agent_id}")
        print(f"  Waiting for slot command from server")
        print(f"{'═'*54}\n")

    def stop(self):
        self._cmd.stop()
        super().stop()

    def set_tx_gain(self, gain: float):
        self.gain = gain
        self.usrp.set_gain(gain, 0)

    def request_stop(self): self._stop_event.set()
    def wait_for_stop(self): self._stop_event.wait()


# ═════════════════════════════════════════════════════════════════════════════
def argument_parser():
    p = ArgumentParser(description="LoRa TX — MADDPG slot control")

    # RF
    p.add_argument("-f", "--center-freq", dest="center_freq",
                   type=eng_float, default="915.0M")
    p.add_argument("-g", "--gain", dest="gain",
                   type=eng_float, default="30.0")
    p.add_argument("-s", "--samp-rate", dest="samp_rate",
                   type=eng_float, default="250.0k")

    # LoRa modulation
    p.add_argument("--sf", dest="sf",
                   type=int, default=7, choices=range(7, 13),
                   help="Spreading factor 7-12 (default 7)")
    p.add_argument("--cr", dest="cr",
                   type=int, default=1, choices=[1, 2, 3, 4],
                   help="Coding rate 1-4 (default 1)")
    p.add_argument("--preamble-len", dest="preamble_len",
                   type=int, default=8,
                   help="Preamble symbols (default 8, use 12 for SF>=10)")
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

    # TX timing
    p.add_argument("-m", "--message", dest="user_message",
                   type=str, default="hello world")
    p.add_argument("-p", "--period", dest="frame_period",
                   type=int, default=1000,
                   help="Default ms between packets (default 1000, "
                        "use 4000+ for SF12). Server may override per slot.")

    # Server
    p.add_argument("--server",   dest="server_host", default="127.0.0.1")
    p.add_argument("--port",     dest="server_port", type=int, default=None,
                   help="Override server port (default 5555+agent_id)")
    p.add_argument("--agent-id", dest="agent_id",    type=int, default=0,
                   help="Agent index (default 0)")
    return p


def main(options=None):
    if options is None:
        options = argument_parser().parse_args()

    tb = lora_TX(
        center_freq=options.center_freq,
        gain=options.gain,
        samp_rate=options.samp_rate,
        user_message=options.user_message,
        frame_period=options.frame_period,
        sf=options.sf,
        cr=options.cr,
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
    print(f"Total packets sent: {tb.strobe.sequence_number}")


if __name__ == '__main__':
    main()
