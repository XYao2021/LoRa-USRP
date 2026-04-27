#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# Updated for tapparelj/gr-lora_sdr v1.0 (GNU Radio 3.10)
# Block signature changes from original GR 3.8 version:
#   frame_sync(bw,bw,sf,impl_head,[18]) -> frame_sync(center_freq, bw, sf, impl_head, sync_word, os_factor, preamble_len)
#   fft_demod(sf, impl_head)            -> fft_demod(soft_decoding, max_log_approx)
#   gray_enc()                          -> gray_mapping()   [renamed]
#   deinterleaver(sf)                   -> deinterleaver(soft_decoding)
#   header_decoder(impl_head,cr,pl,crc) -> header_decoder(impl_head, cr, pay_len, has_crc, ldro, print_header)

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

    def __init__(self, center_freq=861100000, gain=50, samp_rate=250000):
        gr.top_block.__init__(self, "Lora Rx")

        ##################################################
        # Parameters
        ##################################################
        self.center_freq = center_freq
        self.gain        = gain
        self.samp_rate   = samp_rate

        ##################################################
        # Variables
        ##################################################
        self.sf        = sf        = 7
        self.pay_len   = pay_len   = 11
        self.impl_head = impl_head = False
        self.has_crc   = has_crc   = False
        self.cr        = cr        = 1
        self.bw        = bw        = samp_rate
        self.sync_word = sync_word = [0x12]   # must match TX
        self.ldro_mode = ldro_mode = 2        # auto LDRO
        self.os_factor = os_factor = 8        # oversampling factor
        self.preamble_len = preamble_len = 8  # must match TX

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            "serial=30CD3F7",
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

        # ── Updated block constructors ────────────────────────────────────────

        # frame_sync(center_freq, bandwidth, sf, impl_head, sync_word, os_factor, preamble_len)
        # self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
        #     center_freq, bw, sf, impl_head, sync_word, os_factor, preamble_len
        # )

        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(
            int(center_freq), int(bw), sf, impl_head, [0x12], 8, 8
        )

        # fft_demod(soft_decoding, max_log_approx)
        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod(False, False)

        # gray_mapping (was gray_enc in GR 3.8 version)
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping(False)

        # deinterleaver(soft_decoding)
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver(False)

        # hamming_dec — unchanged
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(False)

        # header_decoder(impl_head, cr, pay_len, has_crc, ldro, print_header)
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(
            impl_head, cr, pay_len, has_crc, ldro_mode, True
        )

        # dewhitening — unchanged
        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()

        # crc_verif — unchanged
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif(1, False)

        # Interpolating FIR filter — unchanged
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

        ##################################################
        # Connections
        ##################################################
        # Message feedback (mandatory for lora_sdr)
        self.msg_connect(
            (self.lora_sdr_header_decoder_0, 'frame_info'),
            (self.lora_sdr_frame_sync_0,     'frame_info')
        )

        # Stream connections
        self.connect((self.uhd_usrp_source_0,        0), (self.interp_fir_filter_xxx_0,    0))
        self.connect((self.interp_fir_filter_xxx_0,   0), (self.lora_sdr_frame_sync_0,      0))
        self.connect((self.lora_sdr_frame_sync_0,     0), (self.lora_sdr_fft_demod_0,       0))
        self.connect((self.lora_sdr_fft_demod_0,      0), (self.lora_sdr_gray_mapping_0,    0))
        self.connect((self.lora_sdr_gray_mapping_0,   0), (self.lora_sdr_deinterleaver_0,   0))
        self.connect((self.lora_sdr_deinterleaver_0,  0), (self.lora_sdr_hamming_dec_0,     0))
        self.connect((self.lora_sdr_hamming_dec_0,    0), (self.lora_sdr_header_decoder_0,  0))
        self.connect((self.lora_sdr_header_decoder_0, 0), (self.lora_sdr_dewhitening_0,     0))
        self.connect((self.lora_sdr_dewhitening_0,    0), (self.lora_sdr_crc_verif_0,       0))

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
        self.set_bw(v)
        self.uhd_usrp_source_0.set_samp_rate(v)

    def get_sf(self): return self.sf
    def set_sf(self, v): self.sf = v

    def get_pay_len(self): return self.pay_len
    def set_pay_len(self, v): self.pay_len = v

    def get_impl_head(self): return self.impl_head
    def set_impl_head(self, v): self.impl_head = v

    def get_has_crc(self): return self.has_crc
    def set_has_crc(self, v): self.has_crc = v

    def get_cr(self): return self.cr
    def set_cr(self, v): self.cr = v

    def get_bw(self): return self.bw
    def set_bw(self, v):
        self.bw = v
        self.uhd_usrp_source_0.set_bandwidth(v, 1)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--center-freq", dest="center_freq", type=eng_float, default="915.0M",
        help="Set center_freq [default=%(default)r]")
    parser.add_argument(
        "-g", "--gain", dest="gain", type=eng_float, default="50.0",
        help="Set gain [default=%(default)r]")
    parser.add_argument(
        "-s", "--samp-rate", dest="samp_rate", type=eng_float, default="250.0k",
        help="Set samp_rate [default=%(default)r]")
    return parser


def main(top_block_cls=lora_RX, options=None):
    if options is None:
        options = argument_parser().parse_args()
    tb = top_block_cls(
        center_freq=options.center_freq,
        gain=options.gain,
        samp_rate=options.samp_rate
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
