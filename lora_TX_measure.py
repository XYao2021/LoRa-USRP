#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# lora_TX.py  —  Standalone LoRa transmitter, no server required
#
# Sends "SEQ:XXXXX <message>" continuously at --period ms interval.
# Use this to verify the RF link before deploying the MADDPG training system.
#
# All LoRa parameters are CLI arguments so you can test different SF/CR/OS
# settings without editing the file. Examples:
#
#   python lora_TX.py                                           # SF7 default
#   python lora_TX.py --sf 12 --preamble-len 12 --period 4000  # SF12
#   python lora_TX.py --sf 10 --preamble-len 12 --period 2000  # SF10
#
# RX must use matching: --sf, --cr, --preamble-len, --has-crc,
#                       --impl-head, --ldro, --sync-word

from gnuradio import gr
import pmt
import sys
import signal
import time
import threading
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float
from gnuradio import uhd
from gnuradio import lora_sdr


# ═════════════════════════════════════════════════════════════════════════════
# Sequenced message strobe
# ═════════════════════════════════════════════════════════════════════════════
class SeqStrobe(gr.basic_block):
    """
    Continuously sends "SEQ:XXXXX <message>" every frame_period_ms.
    Output port 'strobe' connects to lora_sdr.whitening 'msg' port.
    """
    def __init__(self, user_message="hello world", frame_period_ms=1000):
        gr.basic_block.__init__(self, name="SeqStrobe",
                                in_sig=None, out_sig=None)
        self.user_message    = user_message
        self.frame_period_ms = frame_period_ms
        self._seq            = 0
        self._lock           = threading.Lock()
        self._stop           = threading.Event()
        self._thread         = None
        self.message_port_register_out(pmt.intern("strobe"))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        return True

    def _run(self):
        time.sleep(2.0)   # wait for flowgraph to fully start
        while not self._stop.is_set():
            with self._lock:
                seq = self._seq
                self._seq = (self._seq + 1) % 100000
            msg = f"SEQ:{seq:05d} {self.user_message}"
            print(f"[TX] Sending: {msg!r}")
            self.message_port_pub(pmt.intern("strobe"), pmt.intern(msg))
            self._stop.wait(self.frame_period_ms / 1000.0)

    def set_period(self, ms):
        with self._lock: self.frame_period_ms = ms

    def set_message(self, m):
        with self._lock: self.user_message = m

    @property
    def sequence_number(self):
        with self._lock: return self._seq


# ═════════════════════════════════════════════════════════════════════════════
# Main flowgraph
# ═════════════════════════════════════════════════════════════════════════════
class lora_TX(gr.top_block):

    def __init__(self, center_freq=915000000, gain=30, samp_rate=250000,
                 user_message="hello world", frame_period=1000,
                 sf=7, cr=1, preamble_len=8,
                 has_crc=True, impl_head=False,
                 ldro_mode=2, sync_word=0x12):

        gr.top_block.__init__(self, "LoRa TX standalone")

        self.center_freq  = center_freq
        self.gain         = gain
        self.samp_rate    = samp_rate

        bw             = 250000
        sync_word_list = [sync_word]

        # frame_zero_padd scales with SF — prevents modulator overrun at high SF
        # 2^(sf-1) is safe for all SF values
        frame_zero_padd = 2 ** (sf - 1)

        print(f"\n[TX] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  preamble={preamble_len}")
        print(f"  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  frame_zero_padd={frame_zero_padd}")
        print(f"  freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  period={frame_period} ms  bw={bw} Hz\n")

        # ── USRP — "" finds whatever device is connected ──────────────────────
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            "",
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 1))),
            'frame_len',
        )
        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0.set_gain(gain, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_bandwidth(bw, 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec())

        # ── Sequenced strobe ──────────────────────────────────────────────────
        self.seq_strobe = SeqStrobe(
            user_message=user_message,
            frame_period_ms=frame_period,
        )

        # ── LoRa TX chain ─────────────────────────────────────────────────────
        self.lora_sdr_whitening_0   = lora_sdr.whitening(
            False, False, ',', 'packet_len')

        self.lora_sdr_modulate_0    = lora_sdr.modulate(
            sf, int(samp_rate), int(bw),
            sync_word_list, frame_zero_padd, preamble_len)
        self.lora_sdr_modulate_0.set_min_output_buffer(10000000)

        self.lora_sdr_interleaver_0 = lora_sdr.interleaver(
            cr, sf, ldro_mode, int(bw))

        self.lora_sdr_header_0      = lora_sdr.header(impl_head, has_crc, cr)
        self.lora_sdr_hamming_enc_0 = lora_sdr.hamming_enc(cr, sf)
        self.lora_sdr_gray_demap_0  = lora_sdr.gray_demap(sf)
        self.lora_sdr_add_crc_0     = lora_sdr.add_crc(has_crc)

        # ── Connections ───────────────────────────────────────────────────────
        self.msg_connect(
            (self.seq_strobe,           'strobe'),
            (self.lora_sdr_whitening_0, 'msg')
        )
        self.connect((self.lora_sdr_whitening_0,   0), (self.lora_sdr_header_0,      0))
        self.connect((self.lora_sdr_header_0,      0), (self.lora_sdr_add_crc_0,     0))
        self.connect((self.lora_sdr_add_crc_0,     0), (self.lora_sdr_hamming_enc_0, 0))
        self.connect((self.lora_sdr_hamming_enc_0, 0), (self.lora_sdr_interleaver_0, 0))
        self.connect((self.lora_sdr_interleaver_0, 0), (self.lora_sdr_gray_demap_0,  0))
        self.connect((self.lora_sdr_gray_demap_0,  0), (self.lora_sdr_modulate_0,    0))
        self.connect((self.lora_sdr_modulate_0,    0), (self.uhd_usrp_sink_0,        0))

    def get_gain(self): return self.gain
    def set_gain(self, v):
        self.gain = v
        self.uhd_usrp_sink_0.set_gain(v, 0)

    def get_center_freq(self): return self.center_freq
    def set_center_freq(self, v):
        self.center_freq = v
        self.uhd_usrp_sink_0.set_center_freq(v, 0)

    def get_samp_rate(self): return self.samp_rate
    def set_samp_rate(self, v):
        self.samp_rate = v
        self.uhd_usrp_sink_0.set_samp_rate(v)


# ═════════════════════════════════════════════════════════════════════════════
def argument_parser():
    p = ArgumentParser(description="LoRa TX — standalone, no server")

    # RF
    p.add_argument("-f", "--center-freq", dest="center_freq",
                   type=eng_float, default="915.0M",
                   help="Center frequency (default: 915 MHz)")
    p.add_argument("-g", "--gain", dest="gain",
                   type=eng_float, default="30.0",
                   help="TX gain dB (default: 30)")
    p.add_argument("-s", "--samp-rate", dest="samp_rate",
                   type=eng_float, default="250.0k",
                   help="Sample rate (default: 250 kHz)")

    # LoRa modulation — must match RX
    p.add_argument("--sf", dest="sf",
                   type=int, default=7, choices=range(7, 13),
                   help="Spreading factor 7-12 (default: 7)")
    p.add_argument("--cr", dest="cr",
                   type=int, default=1, choices=[1, 2, 3, 4],
                   help="Coding rate 1-4 (default: 1)")
    p.add_argument("--preamble-len", dest="preamble_len",
                   type=int, default=8,
                   help="Preamble symbols (default: 8, use 12 for SF>=10)")
    p.add_argument("--has-crc", dest="has_crc",
                   action="store_true", default=True,
                   help="Enable CRC (default: True)")
    p.add_argument("--no-crc", dest="has_crc",
                   action="store_false",
                   help="Disable CRC")
    p.add_argument("--impl-head", dest="impl_head",
                   action="store_true", default=False,
                   help="Implicit header mode (default: False)")
    p.add_argument("--ldro", dest="ldro_mode",
                   type=int, default=2, choices=[0, 1, 2],
                   help="LDRO: 0=off 1=on 2=auto (default: 2)")
    p.add_argument("--sync-word", dest="sync_word",
                   type=lambda x: int(x, 0), default=0x12,
                   help="Sync word hex (default: 0x12)")

    # TX timing
    p.add_argument("-m", "--message", dest="user_message",
                   type=str, default="hello world",
                   help="Payload message text")
    p.add_argument("-p", "--period", dest="frame_period",
                   type=int, default=1000,
                   help="ms between packets (default: 1000, use 4000+ for SF12)")

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
    )

    def sig_handler(sig=None, frame=None):
        tb.stop(); tb.wait(); sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()
    print(f"\nTotal packets sent: {tb.seq_strobe.sequence_number}")


if __name__ == '__main__':
    main()
