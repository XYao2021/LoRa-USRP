#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# Updated for tapparelj/gr-lora_sdr v1.0 (GNU Radio 3.10)
#
# Added:
#   1. Raw IQ RSSI/SNR measurement
#   2. Packet-level RSSI/SNR reporting through crc_verif message port
#   3. SX1276/Arduino-style packet RSSI/SNR equivalent estimate
#
# All original configurable LoRa parameters are kept.

from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr, blocks
import sys
import signal
import math
import threading
import time
import numpy as np
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
from gnuradio import lora_sdr
import pmt


WINDOW_SIZE       = 512
DETECT_MARGIN_DB = 6.0
NOISE_EMA_ALPHA  = 0.01
MIN_SIGNAL_WINS  = 3
NOISE_INIT_DBFS  = -60.0


class SignalProbe(gr.sync_block):
    """
    Passthrough block for raw IQ measurement.

    It does not modify samples.
    It estimates:
        raw noise floor in dBFS
        raw signal RSSI in dBFS
        approximate signal RSSI in dBm
        raw SNR in dB
    """

    def __init__(self, rx_gain_db=50.0, noise_figure_db=6.0,
                 detect_margin_db=DETECT_MARGIN_DB,
                 noise_ema_alpha=NOISE_EMA_ALPHA,
                 win_size=WINDOW_SIZE,
                 min_signal_wins=MIN_SIGNAL_WINS):

        gr.sync_block.__init__(
            self,
            name="SignalProbe",
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )

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
            win = power[start:start + self.win_size]
            win_mean = float(np.mean(win))

            if win_mean <= 0:
                continue

            threshold = self._noise_ema_lin * (
                10 ** (self.detect_margin_db / 10.0)
            )

            if win_mean < threshold:
                self._noise_ema_lin = (
                    self.alpha * win_mean
                    + (1.0 - self.alpha) * self._noise_ema_lin
                )

                if self._signal_acc:
                    with self._lock:
                        self._burst_count += 1
                    self._signal_acc = []

                self._above_count = 0
                noise_dbfs = 10.0 * math.log10(self._noise_ema_lin)

                with self._lock:
                    self._noise_floor_dbfs  = noise_dbfs
                    self._is_signal_present = False

            else:
                self._above_count += 1
                self._signal_acc.append(win_mean)

                if self._above_count >= self.min_signal_wins:
                    sig_mean   = float(np.mean(self._signal_acc))
                    sig_dbfs   = 10.0 * math.log10(sig_mean)
                    noise_dbfs = 10.0 * math.log10(self._noise_ema_lin)
                    snr_db     = sig_dbfs - noise_dbfs

                    # Approximate conversion from dBFS to dBm.
                    # This is calibration-dependent.
                    sig_dbm = (
                        sig_dbfs
                        + self.rx_gain_db
                        - self.noise_figure_db
                        - 30.0
                    )

                    with self._lock:
                        self._signal_rssi_dbfs  = sig_dbfs
                        self._signal_rssi_dbm   = sig_dbm
                        self._raw_snr_db        = snr_db
                        self._noise_floor_dbfs  = noise_dbfs
                        self._is_signal_present = True

        return n


class PacketReporter(gr.basic_block):
    """
    Packet-level reporter.

    It receives decoded packets from:
        crc_verif 'msg' port -> PacketReporter 'msg_in'

    Packet RSSI/SNR are estimated from the most recent raw IQ measurement.

    Important:
        These are SX1276/Arduino-style equivalent values.
        They are not actual SX1276 hardware registers.
    """

    def __init__(self, probe, rx_gain_db=50.0):
        gr.basic_block.__init__(
            self,
            name="PacketReporter",
            in_sig=None,
            out_sig=None,
        )

        self.probe      = probe
        self.rx_gain_db = rx_gain_db

        self._lock      = threading.Lock()
        self._pkt_count = 0
        self._records   = []

        self.message_port_register_in(pmt.intern("msg_in"))
        self.set_msg_handler(pmt.intern("msg_in"), self._handle_msg)

    def _handle_msg(self, msg):
        try:
            if pmt.is_pair(msg):
                data = pmt.cdr(msg)
            else:
                data = msg

            if pmt.is_u8vector(data):
                payload = bytes(pmt.u8vector_elements(data)).decode(
                    "ascii", errors="replace"
                ).strip()
            elif pmt.is_symbol(data):
                payload = pmt.symbol_to_string(data).strip()
            else:
                payload = str(data).strip()

        except Exception as e:
            payload = f"<parse error: {e}>"

        self._report(payload)

    def _report(self, payload):
        rssi_dbfs  = self.probe.signal_rssi_dbfs
        rssi_dbm   = self.probe.signal_rssi_dbm
        snr_db     = self.probe.raw_snr_db
        noise_dbfs = self.probe.noise_floor_dbfs

        # SX1276/Arduino-style equivalent packet SNR.
        # SX127x packet SNR register resolution is 0.25 dB.
        pkt_snr_reg = int(snr_db * 4.0)
        arduino_snr = pkt_snr_reg * 0.25

        # SX1276 HF-band equivalent packet RSSI estimate.
        # This is an equivalent mapping, not an actual register read.
        pkt_rssi_reg = rssi_dbm + 157.0

        if snr_db >= 0:
            arduino_rssi = -157.0 + (16.0 / 15.0) * pkt_rssi_reg
        else:
            arduino_rssi = -157.0 + pkt_rssi_reg + arduino_snr

        with self._lock:
            self._pkt_count += 1
            count = self._pkt_count

            record = dict(
                time=time.strftime("%H:%M:%S"),
                count=count,
                payload=payload,
                rssi_dbfs=rssi_dbfs,
                rssi_dbm=rssi_dbm,
                noise_dbfs=noise_dbfs,
                snr_db=snr_db,
                arduino_rssi=arduino_rssi,
                arduino_snr=arduino_snr,
            )

            self._records.append(record)

        self._print(record)

    def _print(self, r):
        print(
            f"\n╔═ [{r['time']}] Packet #{r['count']}"
            f"\n║ Payload              : {r['payload']!r}"
            f"\n║ ── Raw IQ measurement ─────────────────────"
            f"\n║ Noise floor          : {r['noise_dbfs']:+.1f} dBFS"
            f"\n║ Raw RSSI             : {r['rssi_dbfs']:+.1f} dBFS"
            f"\n║ Approx. RSSI         : {r['rssi_dbm']:+.1f} dBm"
            f"\n║ Raw SNR              : {r['snr_db']:+.1f} dB"
            f"\n║ ── SX1276-equivalent packet values ────────"
            f"\n║ LoRa.packetRssi      : {r['arduino_rssi']:+.1f} dBm"
            f"\n╚ LoRa.packetSnr       : {r['arduino_snr']:+.1f} dB",
            flush=True,
        )

    @property
    def packet_count(self):
        with self._lock:
            return self._pkt_count

    def summary(self):
        with self._lock:
            n = self._pkt_count
            if n == 0:
                return "No packets decoded."

            rssi = [r["arduino_rssi"] for r in self._records]
            snr  = [r["arduino_snr"] for r in self._records]

            return (
                f"Packets decoded: {n} | "
                f"Avg RSSI: {sum(rssi) / n:+.1f} dBm | "
                f"Avg SNR: {sum(snr) / n:+.1f} dB | "
                f"Min RSSI: {min(rssi):+.1f} dBm | "
                f"Max RSSI: {max(rssi):+.1f} dBm"
            )


class StatsPrinter:
    def __init__(self, probe, reporter, interval=2.0):
        self.probe    = probe
        self.reporter = reporter
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        hdr = (
            f"\n{'Time':>8}  {'Noise(dBFS)':>12}  "
            f"{'Sig(dBFS)':>10}  {'Sig(dBm~)':>10}  "
            f"{'SNR(dB)':>8}  {'Present':>8}  {'Bursts':>6}  {'Pkts':>5}"
        )
        print(hdr)
        print("─" * len(hdr))

        while not self._stop.is_set():
            time.sleep(self.interval)

            p = self.probe
            present = "YES" if p.is_signal_present else "---"

            sig_dbfs = f"{p.signal_rssi_dbfs:+10.2f}" if p.is_signal_present else f"{'---':>10}"
            sig_dbm  = f"{p.signal_rssi_dbm:+10.2f}"  if p.is_signal_present else f"{'---':>10}"
            snr      = f"{p.raw_snr_db:+8.2f}"         if p.is_signal_present else f"{'---':>8}"

            print(
                f"{time.strftime('%H:%M:%S'):>8}  "
                f"{p.noise_floor_dbfs:+12.2f}  "
                f"{sig_dbfs}  "
                f"{sig_dbm}  "
                f"{snr}  "
                f"{present:>8}  "
                f"{p.burst_count:>6d}  "
                f"{self.reporter.packet_count:>5d}",
                flush=True,
            )


class lora_RX(gr.top_block):

    def __init__(self, center_freq=861100000, gain=50, samp_rate=250000,
                 sf=7, cr=1, os_factor=8, preamble_len=8,
                 pay_len=21, has_crc=False, impl_head=False,
                 ldro_mode=2, sync_word=0x12,
                 noise_figure_db=6.0, detect_margin_db=DETECT_MARGIN_DB,
                 print_interval=2.0):

        gr.top_block.__init__(self, "Lora Rx")

        self.center_freq  = center_freq
        self.gain         = gain
        self.samp_rate    = samp_rate
        self.sf           = sf
        self.cr           = cr
        self.os_factor    = os_factor
        self.preamble_len = preamble_len
        self.pay_len      = pay_len
        self.has_crc      = has_crc
        self.impl_head    = impl_head
        self.ldro_mode    = ldro_mode
        self.sync_word    = [sync_word]
        self.bw           = samp_rate

        buf_size = max(2 ** sf * os_factor * 8, 65536)

        print(f"\n[RX] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  OS={os_factor}  preamble={preamble_len}")
        print(f"  pay_len={pay_len}  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  center_freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  buf_size={buf_size}\n")

        self.uhd_usrp_source_0 = uhd.usrp_source(
            "",
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        self.uhd_usrp_source_0.set_min_output_buffer(buf_size)

        self.signal_probe_0 = SignalProbe(
            rx_gain_db=gain,
            noise_figure_db=noise_figure_db,
            detect_margin_db=detect_margin_db,
        )
        self.signal_probe_0.set_min_output_buffer(buf_size)

        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccf(
            os_factor,
            (
                -0.128616616593872, -0.212206590789194, -0.180063263231421,
                 3.89817183251938e-17, 0.300105438719035, 0.636619772367581,
                 0.900316316157106, 1,
                 0.900316316157106, 0.636619772367581, 0.300105438719035,
                 3.89817183251938e-17, -0.180063263231421,
                -0.212206590789194, -0.128616616593872
            )
        )
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.interp_fir_filter_xxx_0.set_min_output_buffer(buf_size)

        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(center_freq), int(self.bw), sf, impl_head,
            self.sync_word, os_factor, preamble_len
        )
        self.lora_sdr_frame_sync_0.set_min_output_buffer(buf_size)

        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod(False, False)
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping(False)
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver(False)
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(False)

        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(
            impl_head, cr, pay_len, has_crc, ldro_mode, True
        )

        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif(1, False)

        self.packet_reporter_0 = PacketReporter(
            probe=self.signal_probe_0,
            rx_gain_db=gain,
        )

        self.null_sink_0 = blocks.null_sink(gr.sizeof_char)

        self.msg_connect(
            (self.lora_sdr_header_decoder_0, "frame_info"),
            (self.lora_sdr_frame_sync_0,     "frame_info")
        )

        self.msg_connect(
            (self.lora_sdr_crc_verif_0, "msg"),
            (self.packet_reporter_0,    "msg_in")
        )

        self.connect((self.uhd_usrp_source_0,        0), (self.signal_probe_0,              0))
        self.connect((self.signal_probe_0,           0), (self.interp_fir_filter_xxx_0,     0))
        self.connect((self.interp_fir_filter_xxx_0,  0), (self.lora_sdr_frame_sync_0,       0))
        self.connect((self.lora_sdr_frame_sync_0,    0), (self.lora_sdr_fft_demod_0,        0))
        self.connect((self.lora_sdr_fft_demod_0,     0), (self.lora_sdr_gray_mapping_0,     0))
        self.connect((self.lora_sdr_gray_mapping_0,  0), (self.lora_sdr_deinterleaver_0,    0))
        self.connect((self.lora_sdr_deinterleaver_0, 0), (self.lora_sdr_hamming_dec_0,      0))
        self.connect((self.lora_sdr_hamming_dec_0,   0), (self.lora_sdr_header_decoder_0,   0))
        self.connect((self.lora_sdr_header_decoder_0,0), (self.lora_sdr_dewhitening_0,      0))
        self.connect((self.lora_sdr_dewhitening_0,   0), (self.lora_sdr_crc_verif_0,        0))
        self.connect((self.lora_sdr_crc_verif_0,     0), (self.null_sink_0,                 0))

        self._printer = StatsPrinter(
            self.signal_probe_0,
            self.packet_reporter_0,
            interval=print_interval,
        )

    def start(self):
        super().start()
        self._printer.start()

    def stop(self):
        self._printer.stop()
        super().stop()
        print("\n[RX] Session summary:")
        print(" ", self.packet_reporter_0.summary())

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, v):
        self.center_freq = v
        self.uhd_usrp_source_0.set_center_freq(v, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, v):
        self.gain = v
        self.uhd_usrp_source_0.set_gain(v, 0)
        self.signal_probe_0.rx_gain_db = v
        self.packet_reporter_0.rx_gain_db = v

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, v):
        self.samp_rate = v
        self.bw = v
        self.uhd_usrp_source_0.set_samp_rate(v)


def argument_parser():
    parser = ArgumentParser(description="LoRa RX — configurable parameters with RSSI/SNR")

    parser.add_argument("-f", "--center-freq", dest="center_freq",
                        type=eng_float, default="915.0M",
                        help="Center frequency (default: 915 MHz)")
    parser.add_argument("-g", "--gain", dest="gain",
                        type=eng_float, default="50.0",
                        help="RX gain dB (default: 50)")
    parser.add_argument("-s", "--samp-rate", dest="samp_rate",
                        type=eng_float, default="250.0k",
                        help="Sample rate (default: 250 kHz)")

    parser.add_argument("--sf", dest="sf",
                        type=int, default=7, choices=range(7, 13),
                        help="Spreading factor 7-12 (default: 7)")
    parser.add_argument("--cr", dest="cr",
                        type=int, default=1, choices=[1, 2, 3, 4],
                        help="Coding rate 1-4 (default: 1)")
    parser.add_argument("--os-factor", dest="os_factor",
                        type=int, default=8, choices=[1, 2, 4, 8],
                        help="Oversampling factor (default: 8)")
    parser.add_argument("--preamble-len", dest="preamble_len",
                        type=int, default=8,
                        help="Preamble length symbols (default: 8)")
    parser.add_argument("--pay-len", dest="pay_len",
                        type=int, default=21,
                        help="Expected payload length bytes (default: 21)")
    parser.add_argument("--has-crc", dest="has_crc",
                        action="store_true", default=False,
                        help="Enable CRC verification")
    parser.add_argument("--impl-head", dest="impl_head",
                        action="store_true", default=False,
                        help="Implicit header mode")
    parser.add_argument("--ldro", dest="ldro_mode",
                        type=int, default=2, choices=[0, 1, 2],
                        help="LDRO mode: 0=off 1=on 2=auto")
    parser.add_argument("--sync-word", dest="sync_word",
                        type=lambda x: int(x, 0), default=0x12,
                        help="Sync word hex e.g. 0x12")

    parser.add_argument("-n", "--noise-figure", dest="noise_figure_db",
                        type=float, default=6.0,
                        help="Noise figure dB for approximate dBm estimate")
    parser.add_argument("-d", "--detect-margin", dest="detect_margin_db",
                        type=float, default=DETECT_MARGIN_DB,
                        help="dB above noise floor to declare signal")
    parser.add_argument("-i", "--interval", dest="print_interval",
                        type=float, default=2.0,
                        help="Stats print interval in seconds")

    return parser


def main(top_block_cls=lora_RX, options=None):
    if options is None:
        options = argument_parser().parse_args()

    tb = top_block_cls(
        center_freq=options.center_freq,
        gain=options.gain,
        samp_rate=options.samp_rate,
        sf=options.sf,
        cr=options.cr,
        os_factor=options.os_factor,
        preamble_len=options.preamble_len,
        pay_len=options.pay_len,
        has_crc=options.has_crc,
        impl_head=options.impl_head,
        ldro_mode=options.ldro_mode,
        sync_word=options.sync_word,
        noise_figure_db=options.noise_figure_db,
        detect_margin_db=options.detect_margin_db,
        print_interval=options.print_interval,
    )

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input("Press Enter to quit: ")
    except EOFError:
        pass

    tb.stop()
    tb.wait()


if __name__ == "__main__":
    main()
