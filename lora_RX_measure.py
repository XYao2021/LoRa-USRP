#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# lora_RX.py  —  Standalone LoRa receiver, no server required
#
# Prints two levels of RSSI/SNR for every received packet:
#
#   Raw IQ level (SignalProbe)  — from windowed power measurement on raw IQ
#       RSSI dBFS, RSSI dBm~, SNR dB, noise floor dBFS
#   Packet level (PacketReporter) — stamped at the moment crc_verif outputs
#       payload, same raw measurements taken at packet decode time
#       plus Arduino SX1276 formula RSSI and SNR for comparison
#
# Also prints a periodic raw IQ stats table every --interval seconds
# regardless of whether packets are being received.
#
# Use this to verify the RF link before deploying the MADDPG training system.
#
# All LoRa parameters are CLI arguments. Examples:
#
#   python lora_RX.py                                          # SF7 default
#   python lora_RX.py --sf 12 --os-factor 1 --preamble-len 12 # SF12
#   python lora_RX.py --sf 10 --os-factor 4 --preamble-len 12 # SF10
#
# Must match TX settings: --sf, --cr, --preamble-len, --has-crc,
#                         --impl-head, --ldro, --sync-word
#
# Buffer size scales automatically: max(2^sf * os_factor * 8, 65536)
# Applied to uhd_usrp_source_0, interp_fir_filter_xxx_0, lora_sdr_frame_sync_0

from gnuradio import filter as gr_filter
from gnuradio.filter import firdes
from gnuradio import gr, blocks
import sys
import signal
import math
import threading
import time
import numpy as np
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float
from gnuradio import uhd
from gnuradio import lora_sdr
import pmt

WINDOW_SIZE      = 512
DETECT_MARGIN_DB = 6.0
NOISE_EMA_ALPHA  = 0.01
MIN_SIGNAL_WINS  = 3
NOISE_INIT_DBFS  = -60.0


# ═════════════════════════════════════════════════════════════════════════════
# Raw IQ probe — threshold-gated windowed power measurement
# ═════════════════════════════════════════════════════════════════════════════
class SignalProbe(gr.sync_block):
    """
    Sits in-stream between USRP and FIR filter.
    Tracks noise floor via EMA and detects signal bursts.
    Properties are thread-safe and read by PacketReporter at decode time.
    """

    def __init__(self, rx_gain_db=10.0, adc_offset_db=0.0,
                 detect_margin_db=DETECT_MARGIN_DB,
                 noise_ema_alpha=NOISE_EMA_ALPHA,
                 win_size=WINDOW_SIZE,
                 min_signal_wins=MIN_SIGNAL_WINS):
        """
        rx_gain_db     : USRP RX gain setting in dB (for reference only)
        adc_offset_db  : Calibration offset so that
                             RSSI_dBm = RSSI_dBFS + adc_offset_db
                         Derived empirically: inject a known power P_ref_dBm,
                         read RSSI_dBFS, then:
                             adc_offset_db = P_ref_dBm - RSSI_dBFS
                         Default 0 → dBm reading equals dBFS (uncalibrated).
        """
        gr.sync_block.__init__(self, name="SignalProbe",
                               in_sig=[np.complex64],
                               out_sig=[np.complex64])

        self.rx_gain_db       = rx_gain_db
        self.adc_offset_db    = adc_offset_db   # calibration constant
        self.detect_margin_db = detect_margin_db
        self.alpha            = noise_ema_alpha
        self.win_size         = win_size
        self.min_signal_wins  = min_signal_wins

        self._noise_ema_lin     = 10 ** (NOISE_INIT_DBFS / 10.0)
        self._above_count       = 0
        self._signal_acc        = []

        self._lock              = threading.Lock()
        self._noise_floor_dbfs  = NOISE_INIT_DBFS
        self._signal_rssi_dbfs  = NOISE_INIT_DBFS
        self._signal_rssi_dbm   = NOISE_INIT_DBFS + adc_offset_db
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
            if win_mean <= 0:
                continue
            threshold = self._noise_ema_lin * (10 ** (self.detect_margin_db / 10.0))
            if win_mean < threshold:
                self._noise_ema_lin = (self.alpha * win_mean +
                                       (1.0 - self.alpha) * self._noise_ema_lin)
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

                    # ── RSSI dBFS ─────────────────────────────────────────────
                    # P_fullscale = 1.0 (UHD normalises IQ to [-1, 1])
                    # RSSI_dBFS = 10 log10(P_measured / P_fullscale)
                    #           = 10 log10(P_measured)          [ref: P_fs=1]
                    sig_dbfs   = 10.0 * math.log10(sig_mean)
                    noise_dbfs = 10.0 * math.log10(self._noise_ema_lin)

                    # ── RSSI dBm ──────────────────────────────────────────────
                    # Correct: P_dBm = RSSI_dBFS − RX_gain_dB + C_ADC
                    # C_ADC is the ADC full-scale calibration constant (hardware-
                    # specific, must be measured empirically with a known source).
                    # We store dBFS and set dBm = dBFS + adc_offset_db.
                    # adc_offset_db = C_ADC − RX_gain_dB, default 0 means the
                    # dBm reading is uncalibrated (relative only).
                    sig_dbm    = sig_dbfs + self.adc_offset_db

                    # ── SNR dB ────────────────────────────────────────────────
                    # P_measured = P_signal + P_noise  (signal windows)
                    # P_signal   = P_measured − P_noise
                    # SNR        = P_signal / P_noise
                    #            = (P_measured − P_noise) / P_noise
                    # Ref: Proakis & Manolakis, "Digital Signal Processing" §12
                    # Note: approximating SNR ≈ P_measured/P_noise (i.e. using
                    # sig_dbfs − noise_dbfs) overestimates by ε = 10 log10(1+1/SNR)
                    # which is < 0.5 dB for SNR > 10 dB.
                    p_signal_lin = max(sig_mean - self._noise_ema_lin, 1e-30)
                    snr_lin      = p_signal_lin / self._noise_ema_lin
                    snr_db       = 10.0 * math.log10(snr_lin)

                    with self._lock:
                        self._signal_rssi_dbfs  = sig_dbfs
                        self._signal_rssi_dbm   = sig_dbm
                        self._raw_snr_db        = snr_db
                        self._noise_floor_dbfs  = noise_dbfs
                        self._is_signal_present = True
        return n


# ═════════════════════════════════════════════════════════════════════════════
# Packet reporter — reads probe at the moment crc_verif outputs a packet
# ═════════════════════════════════════════════════════════════════════════════
class PacketReporter(gr.basic_block):
    """
    Connected to crc_verif 'msg' port.
    On every decoded packet:
      1. Reads current probe measurements (raw IQ level at decode time)
      2. Computes Arduino SX1276 formula RSSI and SNR
      3. Prints a formatted packet report

    This gives packet-level measurements rather than averaged-over-time.
    """

    def __init__(self, probe, rx_gain_db=10.0):
        gr.basic_block.__init__(self, name="PacketReporter",
                                in_sig=None, out_sig=None)
        self.probe      = probe
        self.rx_gain_db = rx_gain_db
        self._pkt_count = 0

        self.message_port_register_in(pmt.intern("msg_in"))
        self.set_msg_handler(pmt.intern("msg_in"), self._handle_msg)

    def _handle_msg(self, msg):
        # Decode payload
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

        self._pkt_count += 1

        # Raw IQ measurements at this instant
        rssi_dbfs  = self.probe.signal_rssi_dbfs
        rssi_dbm   = self.probe.signal_rssi_dbm   # dBFS + adc_offset_db
        snr_db     = self.probe.raw_snr_db
        noise_dbfs = self.probe.noise_floor_dbfs

        # ── Arduino SX1276 formula (Semtech AN1200.22 §3.5 / datasheet §5.5.5) ─
        #
        # The SX1276 stores two 8-bit registers after each received packet:
        #   PacketRssi  (uint8):  PacketRssi = RSSI_dBm + 157
        #   PacketSnr   (int8):   PacketSnr  = SNR_dB × 4   (range −32 .. +31.75 dB)
        #
        # Reading back (what software sees):
        #   SNR_dB   = PacketSnr / 4
        #
        #   If SNR ≥ 0 (strong signal, corrected formula):
        #     RSSI_dBm = −157 + (16/15) × PacketRssi
        #
        #   If SNR < 0 (weak signal, near sensitivity):
        #     RSSI_dBm = −157 + PacketRssi + PacketSnr/4
        #              = −157 + PacketRssi + SNR_dB
        #
        # We simulate what an SX1276 would report given our SDR measurements.
        # We use rssi_dbm (calibrated if adc_offset_db was set, else = rssi_dbfs).
        #
        # Note: the formula is only as accurate as rssi_dbm. If adc_offset_db
        # is not calibrated, treat arduino_rssi as a relative indicator only.
        pkt_rssi_reg = rssi_dbm + 157.0            # simulate PacketRssi register
        pkt_snr_reg  = snr_db * 4.0                # simulate PacketSnr register
        if snr_db >= 0:
            arduino_rssi = -157.0 + (16.0 / 15.0) * pkt_rssi_reg
        else:
            arduino_rssi = -157.0 + pkt_rssi_reg + snr_db   # = rssi_dbm + SNR_dB
        arduino_snr = snr_db    # SNR_dB = PacketSnr/4, no conversion needed

        print(
            f"\n╔═ PKT #{self._pkt_count:04d}  [{time.strftime('%H:%M:%S')}]"
            f"\n║  Payload : {payload!r}"
            f"\n║  ── Raw IQ level ──────────────────────────────"
            f"\n║  RSSI    : {rssi_dbfs:>+8.2f} dBFS"
            f"   ~{rssi_dbm:>+8.2f} dBm"
            f"\n║  SNR     : {snr_db:>+8.2f} dB"
            f"\n║  Noise   : {noise_dbfs:>+8.2f} dBFS"
            f"\n║  ── Arduino SX1276 formula ─────────────────────"
            f"\n╚  RSSI    : {arduino_rssi:>+8.2f} dBm"
            f"   SNR: {arduino_snr:>+6.2f} dB",
            flush=True
        )


# ═════════════════════════════════════════════════════════════════════════════
# Periodic raw IQ stats printer
# ═════════════════════════════════════════════════════════════════════════════
class StatsPrinter:
    def __init__(self, probe, interval=5.0):
        self.probe    = probe
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self): self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        hdr = (f"\n{'Time':>8}  {'Noise(dBFS)':>11}  "
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
            print(f"{time.strftime('%H:%M:%S'):>8}  "
                  f"{p.noise_floor_dbfs:>+11.2f}  "
                  f"{sdb}  {sdm}  {snr}  {pres}  {p.burst_count:>6d}",
                  flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main flowgraph
# ═════════════════════════════════════════════════════════════════════════════
class lora_RX(gr.top_block):

    def __init__(self, center_freq=915000000, gain=10, samp_rate=250000,
                 adc_offset_db=0.0, detect_margin_db=DETECT_MARGIN_DB,
                 print_interval=5.0, payload_len=21,
                 sf=7, cr=1, os_factor=8, preamble_len=8,
                 has_crc=True, impl_head=False,
                 ldro_mode=2, sync_word=0x12):

        gr.top_block.__init__(self, "LoRa RX standalone")

        self.center_freq = center_freq
        self.gain        = gain
        self.samp_rate   = samp_rate

        bw             = samp_rate
        sync_word_list = [sync_word]

        # Buffer size scales with SF and os_factor
        # frame_sync requires at least 2^sf * os_factor samples input
        # Multiply by 8 for safety margin
        buf_size = max(2 ** sf * os_factor * 8, 65536)

        print(f"\n[RX] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  os_factor={os_factor}  preamble={preamble_len}")
        print(f"  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  payload_len={payload_len}")
        print(f"  freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  buf_size={buf_size}\n")

        # ── USRP — "" finds whatever device is connected ──────────────────────
        self.uhd_usrp_source_0 = uhd.usrp_source(
            "",
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 1))),
        )
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        # Buffer fix — correct variable name uhd_usrp_source_0 (not self.usrp)
        self.uhd_usrp_source_0.set_min_output_buffer(buf_size)

        # ── Raw IQ probe — in-stream, between USRP and FIR ───────────────────
        self.probe = SignalProbe(
            rx_gain_db=gain,
            adc_offset_db=adc_offset_db,
            detect_margin_db=detect_margin_db,
        )

        # ── Interpolating FIR filter ──────────────────────────────────────────
        self.interp_fir_filter_xxx_0 = gr_filter.interp_fir_filter_ccf(
            os_factor,
            (
                -0.128616616593872, -0.212206590789194, -0.180063263231421,
                 3.89817183251938e-17, 0.300105438719035, 0.636619772367581,
                 0.900316316157106, 1.0,
                 0.900316316157106, 0.636619772367581, 0.300105438719035,
                 3.89817183251938e-17, -0.180063263231421,
                -0.212206590789194, -0.128616616593872
            )
        )
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        # Buffer fix — correct variable name interp_fir_filter_xxx_0 (not self.fir)
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
        self.blocks_null_sink_0       = blocks.null_sink(gr.sizeof_char)

        # ── Packet reporter — triggered by crc_verif msg port ─────────────────
        self.reporter = PacketReporter(probe=self.probe, rx_gain_db=gain)

        # ── Stream connections ────────────────────────────────────────────────
        self.connect((self.uhd_usrp_source_0,         0), (self.probe,                    0))
        self.connect((self.probe,                      0), (self.interp_fir_filter_xxx_0,  0))
        self.connect((self.interp_fir_filter_xxx_0,    0), (self.lora_sdr_frame_sync_0,    0))
        self.connect((self.lora_sdr_frame_sync_0,      0), (self.lora_sdr_fft_demod_0,     0))
        self.connect((self.lora_sdr_fft_demod_0,       0), (self.lora_sdr_gray_mapping_0,  0))
        self.connect((self.lora_sdr_gray_mapping_0,    0), (self.lora_sdr_deinterleaver_0, 0))
        self.connect((self.lora_sdr_deinterleaver_0,   0), (self.lora_sdr_hamming_dec_0,   0))
        self.connect((self.lora_sdr_hamming_dec_0,     0), (self.lora_sdr_header_decoder_0,0))
        self.connect((self.lora_sdr_header_decoder_0,  0), (self.lora_sdr_dewhitening_0,   0))
        self.connect((self.lora_sdr_dewhitening_0,     0), (self.lora_sdr_crc_verif_0,     0))
        self.connect((self.lora_sdr_crc_verif_0,       0), (self.blocks_null_sink_0,       0))

        # ── Message connections ───────────────────────────────────────────────
        self.msg_connect(
            (self.lora_sdr_header_decoder_0, 'frame_info'),
            (self.lora_sdr_frame_sync_0,     'frame_info')
        )
        self.msg_connect(
            (self.lora_sdr_crc_verif_0, 'msg'),
            (self.reporter,             'msg_in')
        )

        # ── Periodic stats printer ────────────────────────────────────────────
        self._printer = StatsPrinter(self.probe, interval=print_interval)

    def start(self):
        super().start()
        self._printer.start()
        print(f"\n{'═'*56}")
        print(f"  LoRa RX  standalone  —  listening")
        print(f"  Freq    : {self.center_freq/1e6:.3f} MHz  gain={self.gain} dB")
        print(f"  Sync    : 0x12")
        print(f"  Packets : printed as received")
        print(f"{'═'*56}\n")

    def stop(self):
        self._printer.stop()
        super().stop()

    def get_gain(self): return self.gain
    def set_gain(self, v):
        self.gain = v
        self.uhd_usrp_source_0.set_gain(v, 0)

    def get_center_freq(self): return self.center_freq
    def set_center_freq(self, v):
        self.center_freq = v
        self.uhd_usrp_source_0.set_center_freq(v, 0)

    def get_samp_rate(self): return self.samp_rate
    def set_samp_rate(self, v):
        self.samp_rate = v
        self.uhd_usrp_source_0.set_samp_rate(v)


# ═════════════════════════════════════════════════════════════════════════════
def argument_parser():
    p = ArgumentParser(description="LoRa RX — standalone, no server")

    # RF
    p.add_argument("-f", "--center-freq", dest="center_freq",
                   type=eng_float, default="915.0M",
                   help="Center frequency (default: 915 MHz)")
    p.add_argument("-g", "--gain", dest="gain",
                   type=eng_float, default="10.0",
                   help="RX gain dB (default: 10)")
    p.add_argument("-s", "--samp-rate", dest="samp_rate",
                   type=eng_float, default="250.0k",
                   help="Sample rate (default: 250 kHz)")
    p.add_argument("-n", "--adc-offset", dest="adc_offset_db",
                   type=float, default=0.0,
                   help="ADC calibration offset dB: adc_offset = P_ref_dBm - RSSI_dBFS "
                        "measured with a known input. Default 0 = uncalibrated "
                        "(RSSI_dBm = RSSI_dBFS, relative only).")
    p.add_argument("-d", "--detect-margin", dest="detect_margin_db",
                   type=float, default=DETECT_MARGIN_DB,
                   help=f"Signal detection margin dB (default: {DETECT_MARGIN_DB})")
    p.add_argument("-i", "--interval", dest="print_interval",
                   type=float, default=5.0,
                   help="Raw IQ stats print interval seconds (default: 5)")

    # LoRa modulation — must match TX
    p.add_argument("--sf", dest="sf",
                   type=int, default=7, choices=range(7, 13),
                   help="Spreading factor 7-12 (default: 7)")
    p.add_argument("--cr", dest="cr",
                   type=int, default=1, choices=[1, 2, 3, 4],
                   help="Coding rate 1-4 (default: 1)")
    p.add_argument("--os-factor", dest="os_factor",
                   type=int, default=8, choices=[1, 2, 4, 8],
                   help="Oversampling factor (default: 8, use 1 for SF>=10)")
    p.add_argument("--preamble-len", dest="preamble_len",
                   type=int, default=8,
                   help="Preamble symbols (default: 8, use 12 for SF>=10)")
    p.add_argument("-l", "--payload-len", dest="payload_len",
                   type=int, default=21,
                   help="Expected payload length bytes (default: 21)")
    p.add_argument("--has-crc", dest="has_crc",
                   action="store_true", default=True,
                   help="Enable CRC verification (default: True)")
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

    return p


def main(options=None):
    if options is None:
        options = argument_parser().parse_args()

    tb = lora_RX(
        center_freq=options.center_freq,
        gain=options.gain,
        samp_rate=options.samp_rate,
        adc_offset_db=options.adc_offset_db,
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
    print(f"\nTotal packets received: {tb.reporter._pkt_count}")


if __name__ == '__main__':
    main()
