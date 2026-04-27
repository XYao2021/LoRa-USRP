#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# Updated for tapparelj/gr-lora_sdr v1.0 (GNU Radio 3.10)
#
# All key LoRa parameters are now CLI arguments:
#   --sf           Spreading factor 7-12        (default 7)
#   --cr           Coding rate 1-4              (default 1)
#   --os-factor    Oversampling factor          (default 8, use 1 for SF>=10)
#   --preamble-len Preamble symbols             (default 8, use 12 for SF>=10)
#   --pay-len      Payload length bytes         (default 21)
#   --has-crc      Enable CRC check             (default False)
#   --impl-head    Implicit header mode         (default False)
#   --ldro         LDRO mode 0=off,1=on,2=auto  (default 2)
#   --sync-word    Sync word hex e.g. 0x12      (default 0x12)
#
# Buffer fix:
#   set_min_output_buffer is called on the correct variable names
#   (uhd_usrp_source_0 and interp_fir_filter_xxx_0, not self.usrp/self.fir)
#   Buffer size scales with SF: 2^sf * os_factor * 8

from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio import lora_sdr


class lora_RX(gr.top_block):

    def __init__(self, center_freq=861100000, gain=50, samp_rate=250000,
                 sf=7, cr=1, os_factor=8, preamble_len=8,
                 pay_len=21, has_crc=False, impl_head=False,
                 ldro_mode=2, sync_word=0x12):

        gr.top_block.__init__(self, "Lora Rx")

        ##################################################
        # Parameters
        ##################################################
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

        # Buffer size — scales with SF and os_factor
        # frame_sync needs at least 2^sf * os_factor samples
        # Add margin × 8 to be safe
        buf_size = max(2 ** sf * os_factor * 8, 65536)

        print(f"\n[RX] LoRa parameters:")
        print(f"  SF={sf}  CR={cr}  OS={os_factor}  preamble={preamble_len}")
        print(f"  pay_len={pay_len}  has_crc={has_crc}  impl_head={impl_head}")
        print(f"  ldro={ldro_mode}  sync_word=0x{sync_word:02X}")
        print(f"  center_freq={center_freq/1e6:.3f} MHz  gain={gain} dB")
        print(f"  buf_size={buf_size}\n")

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            "",
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())
        # Buffer fix — correct variable name
        self.uhd_usrp_source_0.set_min_output_buffer(buf_size)

        # Interpolating FIR filter
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
        # Buffer fix — correct variable name
        self.interp_fir_filter_xxx_0.set_min_output_buffer(buf_size)

        # frame_sync
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(center_freq), int(self.bw), sf, impl_head,
            self.sync_word, os_factor, preamble_len
        )
        self.lora_sdr_frame_sync_0.set_min_output_buffer(buf_size)

        # fft_demod
        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod(False, False)

        # gray_mapping
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping(False)

        # deinterleaver
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver(False)

        # hamming_dec
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(False)

        # header_decoder
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(
            impl_head, cr, pay_len, has_crc, ldro_mode, True
        )

        # dewhitening
        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()

        # crc_verif
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif(1, False)

        ##################################################
        # Connections
        ##################################################
        self.msg_connect(
            (self.lora_sdr_header_decoder_0, 'frame_info'),
            (self.lora_sdr_frame_sync_0,     'frame_info')
        )

        self.connect((self.uhd_usrp_source_0,         0), (self.interp_fir_filter_xxx_0,   0))
        self.connect((self.interp_fir_filter_xxx_0,    0), (self.lora_sdr_frame_sync_0,     0))
        self.connect((self.lora_sdr_frame_sync_0,      0), (self.lora_sdr_fft_demod_0,      0))
        self.connect((self.lora_sdr_fft_demod_0,       0), (self.lora_sdr_gray_mapping_0,   0))
        self.connect((self.lora_sdr_gray_mapping_0,    0), (self.lora_sdr_deinterleaver_0,  0))
        self.connect((self.lora_sdr_deinterleaver_0,   0), (self.lora_sdr_hamming_dec_0,    0))
        self.connect((self.lora_sdr_hamming_dec_0,     0), (self.lora_sdr_header_decoder_0, 0))
        self.connect((self.lora_sdr_header_decoder_0,  0), (self.lora_sdr_dewhitening_0,    0))
        self.connect((self.lora_sdr_dewhitening_0,     0), (self.lora_sdr_crc_verif_0,      0))

    def get_center_freq(self): return self.center_freq
    def set_center_freq(self, v):
        self.center_freq = v
        self.uhd_usrp_source_0.set_center_freq(v, 0)

    def get_gain(self): return self.gain
    def set_gain(self, v):
        self.gain = v
        self.uhd_usrp_source_0.set_gain(v, 0)

    def get_samp_rate(self): return self.samp_rate
    def set_samp_rate(self, v):
        self.samp_rate = v
        self.bw = v
        self.uhd_usrp_source_0.set_samp_rate(v)


def argument_parser():
    parser = ArgumentParser(description="LoRa RX — configurable parameters")

    # RF parameters
    parser.add_argument("-f", "--center-freq", dest="center_freq",
                        type=eng_float, default="915.0M",
                        help="Center frequency (default: 915 MHz)")
    parser.add_argument("-g", "--gain", dest="gain",
                        type=eng_float, default="50.0",
                        help="RX gain dB (default: 50)")
    parser.add_argument("-s", "--samp-rate", dest="samp_rate",
                        type=eng_float, default="250.0k",
                        help="Sample rate (default: 250 kHz)")

    # LoRa modulation parameters
    parser.add_argument("--sf", dest="sf",
                        type=int, default=7, choices=range(7, 13),
                        help="Spreading factor 7-12 (default: 7)")
    parser.add_argument("--cr", dest="cr",
                        type=int, default=1, choices=[1, 2, 3, 4],
                        help="Coding rate 1-4 (default: 1)")
    parser.add_argument("--os-factor", dest="os_factor",
                        type=int, default=8, choices=[1, 2, 4, 8],
                        help="Oversampling factor (default: 8, use 1 for SF>=10)")
    parser.add_argument("--preamble-len", dest="preamble_len",
                        type=int, default=8,
                        help="Preamble length symbols (default: 8, use 12 for SF>=10)")
    parser.add_argument("--pay-len", dest="pay_len",
                        type=int, default=21,
                        help="Expected payload length bytes (default: 21)")
    parser.add_argument("--has-crc", dest="has_crc",
                        action="store_true", default=False,
                        help="Enable CRC verification (default: False)")
    parser.add_argument("--impl-head", dest="impl_head",
                        action="store_true", default=False,
                        help="Implicit header mode (default: False)")
    parser.add_argument("--ldro", dest="ldro_mode",
                        type=int, default=2, choices=[0, 1, 2],
                        help="LDRO mode: 0=off 1=on 2=auto (default: 2)")
    parser.add_argument("--sync-word", dest="sync_word",
                        type=lambda x: int(x, 0), default=0x12,
                        help="Sync word hex e.g. 0x12 (default: 0x12)")

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
    )

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
