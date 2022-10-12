#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Problem2
# GNU Radio version: 3.9.5.0

from gnuradio import blocks
from gnuradio import filter
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import random
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy.io as sio
from gnuradio import analog
from sklearn import preprocessing

DVB_link =  '/home/leesiho/matlab/lab/p1_SISO_filtered.dat'
WIFI_link = '/home/leesiho/matlab/lab/heSU_bw20_filtered.dat'
Radar_vector = 12*([1]*100 + [0]*100)
class problem2(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Problem2", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.offset1 = offset1 = 0
        self.offset2 = offset2 = 0
        self.offset3 = offset3 = 0
        self.head = head = 1000000
        self.skip_head = skip_head = int(1e6)
        self.file_path = file_path = './student_test.bin'
        self.Tx_rate1 = Tx_rate1 = 1e6
        self.Tx_rate2 = Tx_rate2 = 1e6
        self.Tx_rate3 = Tx_rate3 = 1e6
        self.Tx_freq1 = Tx_freq1 = 1.2e9 
        self.Tx_freq2 = Tx_freq2 = 1.2e9 
        self.Tx_freq3 = Tx_freq3 = 1.2e9
        self.Rx_rate = Rx_rate = 10e6
        self.Rx_freq = Rx_freq = 1.2e9
        self.vector_source = vector_source = Radar_vector
        self.DVB_2T_source = DVB_2T_source = DVB_link
        self.WIFI_source = WIFI_source = WIFI_link
        self.delay1 = delay1 = 0
        self.delay2 = delay2 = 0
    
        ##################################################
        # Blocks
        ##################################################
        ############ source ##############################
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("addr=166.104.231.194", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(Rx_rate)
        self.uhd_usrp_source_0.set_center_freq(Rx_freq, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(10, 0)

        ########### Radar ################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("addr=166.104.231.198", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_samp_rate(Tx_rate1)
        self.uhd_usrp_sink_0.set_center_freq(Tx_freq1, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_gain(5, 0)

        ########### DVB-2T ################################
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join(("addr=166.104.231.199", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            '',
        )

        self.uhd_usrp_sink_0_0.set_samp_rate(Tx_rate2)
        self.uhd_usrp_sink_0_0.set_center_freq(Tx_freq2, 0
                                               )
        self.uhd_usrp_sink_0_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0_0.set_gain(15, 0)

        ########### WIFI ################################
        self.uhd_usrp_sink_0_1 = uhd.usrp_sink(
            ",".join(("addr=166.104.231.184", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            '',
        )
        self.uhd_usrp_sink_0_1.set_samp_rate(Tx_rate3)
        self.uhd_usrp_sink_0_1.set_center_freq(Tx_freq3, 0)
        self.uhd_usrp_sink_0_1.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0_1.set_gain(1, 0)

        ##################################################
        self.hilbert_fc_0 = filter.hilbert_fc(50, window.WIN_BLACKMAN_hARRIS, 6.76)
        self.blocks_vco_f_0 = blocks.vco_f(8e5, 8e5*1.04, 1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.analog_sig_source_x_0 = analog.sig_source_f(8e5, analog.GR_TRI_WAVE, 8e3, 1, 0, 0)
        self.blocks_vector_source_x_0_0 = blocks.vector_source_c(vector_source, False, 1, [])
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, skip_head)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, head)
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_gr_complex*1, DVB_2T_source, False, 0, 0)
        self.blocks_file_source_0_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, WIFI_source, False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, file_path, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, delay1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, delay2)
        ##################################################
        # Connections
        
        ##################################################
        
        (self.uhd_usrp_sink_0, 0)
        
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_vco_f_0, 0))
        self.connect((self.blocks_vco_f_0, 0), (self.hilbert_fc_0, 0))
        self.connect((self.hilbert_fc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_vector_source_x_0_0, 0),(self.blocks_multiply_xx_0, 0)) 
        self.connect((self.blocks_multiply_xx_0, 0), (self.uhd_usrp_sink_0, 0))
        
        # self.connect((self.blocks_file_source_0_0, 0),(self.blocks_delay_0, 0))
        # self.connect((self.blocks_delay_0, 0), (self.uhd_usrp_sink_0_0, 0))
        
        # self.connect((self.blocks_file_source_0, 0),(self.blocks_delay_0_0, 0))
        # self.connect((self.blocks_delay_0_0, 0), (self.uhd_usrp_sink_0_1, 0))
        
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_skiphead_0, 0))

    def get_delay1(self):
        return self.delay1

    def set_delay1(self,delay1):
        self.delay1 = delay1
        self.blocks_delay_0.set_dly(self.delay1)
        
    def get_delay2(self):
        return self.delay2

    def set_delay2(self,delay2):
        self.delay2 = delay2
        self.blocks_delay_0_0.set_dly(self.delay2)    
       
    def get_vector_source(self):
        return self.vector_source

    def set_vector_source(self,vector_source):
        self.vector_source = vector_source
        self.blocks_vector_source_x_0_0.set_data(self.vector_source)
    
    def get_DVB_2T_source(self):
        return self.DVB_2T_source

    def set_DVB_2T_source(self,DVB_2T_source):
        self.DVB_2T_source = DVB_2T_source
        self.blocks_file_source_0_0.open(DVB_2T_source, False, 0, 0)
        

    def get_WIFI_source(self):
        return self.WIFI_source

    def set_WIFI_source(self,WIFI_source):
        self.WIFI_source = WIFI_source
        self.blocks_file_source_0.open(WIFI_source, False, 0, 0)


    def get_offset1(self):
        return self.offset1

    def set_offset1(self, offset1):
        self.offset1 = offset1
        self.set_Tx_freq1(1.2e9 - 5e6 + self.offset1*1e6)

    def get_offset2(self):
        return self.offset2

    def set_offset2(self, offset2):
        self.offset2 = offset2    
        self.set_Tx_freq2(1.2e9 - 5e6 + self.offset2*1e6)

    def get_offset3(self):
        return self.offset3

    def set_offset3(self, offset3):
        self.offset3 = offset3    
        self.set_Tx_freq3(1.2e9 - 5e6 + self.offset3*1e6)

    def get_head(self):
        return self.head

    def set_head(self, head):
        self.head = head
        self.blocks_head_0.set_length(self.head)

    def set_skip_head(self, skip_head):
        self.skip_head = skip_head  

    def get_file_path(self):
        return self.file_path

    def set_file_path(self, file_path):
        self.file_path = file_path
        self.blocks_file_sink_0.open(self.file_path)

    def get_Tx_rate1(self):
        return self.Tx_rate1

    def set_Tx_rate1(self, Tx_rate1):
        self.Tx_rate1 = Tx_rate1
        self.uhd_usrp_sink_0.set_samp_rate(self.Tx_rate1)

    def get_Tx_rate2(self):
        return self.Tx_rate2

    def set_Tx_rate2(self, Tx_rate2):
        self.Tx_rate2 = Tx_rate2
        self.uhd_usrp_sink_0_0.set_samp_rate(self.Tx_rate2)

    def get_Tx_rate3(self):
        return self.Tx_rate3

    def set_Tx_rate3(self, Tx_rate3):
        self.Tx_rate3 = Tx_rate3
        self.uhd_usrp_sink_0_1.set_samp_rate(self.Tx_rate3)

    def get_Tx_freq1(self):
        return self.Tx_freq1

    def set_Tx_freq1(self, Tx_freq1):
        self.Tx_freq1 = Tx_freq1
        self.uhd_usrp_sink_0.set_center_freq(self.Tx_freq1, 0)

    def get_Tx_freq2(self):
        return self.Tx_freq2

    def set_Tx_freq2(self, Tx_freq2):
        self.Tx_freq2 = Tx_freq2
        self.uhd_usrp_sink_0_0.set_center_freq(self.Tx_freq2, 0)

    def get_Tx_freq3(self):
        return self.Tx_freq3

    def set_Tx_freq3(self, Tx_freq3):
        self.Tx_freq3 = Tx_freq3
        self.uhd_usrp_sink_0_1.set_center_freq(self.Tx_freq3, 0)

    def get_Rx_rate(self):
        return self.Rx_rate

    def set_Rx_rate(self, Rx_rate):
        self.Rx_rate = Rx_rate
        self.uhd_usrp_source_0.set_samp_rate(self.Rx_rate)

    def get_Rx_freq(self):
        return self.Rx_freq

    def set_Rx_freq(self, Rx_freq):
        self.Rx_freq = Rx_freq
        self.uhd_usrp_source_0.set_center_freq(self.Rx_freq, 0)
        
def main(top_block_cls=problem2, options=None):
    import pickle
    import os
    import struct
    import scipy.io as sio
    
    # arr = sio.loadmat('/home/leesiho/matlab/lab/p1_SISO_filtered.mat')
    tb = top_block_cls()
    
    idx_a = 1
    idx_b = 10
    idx_c = 1
    idx_d = 9
    
    anslist1 = []
    anslist2 = []
    anslist3 = []
    dlylist1= []
    dlylist2= []
    anslist = []
    n_ans = 3
    
    n_samples_for_slot = 30000 # 10k samples for 1 time slot / timeslot = 0.001 sec => 
    T_slot = 1/tb.get_Rx_rate() * n_samples_for_slot
    print(T_slot)
    
    vector_source = Radar_vector
    DVB_2T_source = DVB_link
    WIFI_source = WIFI_link

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    for i in range(n_ans):
        a = random.randint(idx_a,idx_b)
        b = random.randint(idx_a,idx_b)
        dlylist1.append(a)
        dlylist2.append(b)

    # random sub index and power setting
    for i in range(n_ans):
         #randint makes random integer (idx_a<= x <= idx_b)
         # n: Radar m: DVB-2T k: WIFI
        k = random.randint(idx_c,idx_d)
        if i%2 ==0:
            n = 5
            m = 1
        elif i%100 == 99:
            n = 4001
            m = 4002
            k = 4003
        else:
            n= 6
            m = 4000
        while(1):
            if k == n or k == m:
                k = random.randint(idx_c,idx_d)
            else:
                break

        anslist1.append(n)
        anslist2.append(m)
        anslist3.append(k)
        
 # Throw away garbege data 
    temp_time = time.time()
    tb.set_head(0)
    tb.start()
    while(1):
        curr_time = time.time()
        if curr_time - temp_time >0.2:
            temp_time = curr_time
            
            break
    tb.stop()
    tb.wait()
    print('garbege data check = ',tb.blocks_file_sink_0.nitems_read(0))

    tb.set_skip_head(int(1e7))  

    for i in range(0, n_ans):
        
        tb.set_offset1(anslist1[i])
        tb.set_head((i+1)*n_samples_for_slot)
        tb.set_vector_source(vector_source)
        tb.start()            
        while(1):
            curr_time = time.time()
            if curr_time - temp_time > 2:
                temp_time = curr_time
                break
        print(str(i)+'th data =',tb.blocks_file_sink_0.nitems_read(0))
        
        tb.stop()
        tb.wait()
        
    # label_list = np.array([1,2,3,4,5,6,7,8,9])
    # lb = preprocessing.LabelBinarizer()
    # lb.fit(label_list)
    # radar_label = lb.transform(anslist1)
    # DVB_label = 2*lb.transform(anslist2)
    # WIFI_label = 3*lb.transform(anslist3)
    # label = radar_label + DVB_label + WIFI_label
    # print(label)
    
    
    IQFileName = '/home/leesiho/matlab/lab/ex.dat'
    IQFileName_raw = '/home/leesiho/matlab/lab/ex_raw.dat'
    
    f =  open(tb.get_file_path(), 'rb')
    rawdata = f.read()
    iq_data_lenth = int(len(rawdata)/4)

    # f2 = open(IQFileName_raw,'wb')
    # f2.write(rawdata)

    # iq = []
    # for i in range(iq_data_lenth):
    #     bit_32 = rawdata[i*4:(i+1)*4]
    #     iq_unpacked = struct.unpack('f',bit_32)
    #     iq.append(list(iq_unpacked))
    # iq = np.array(iq)
    # iq_reshaped = iq.reshape(int(len(iq)/2),-1)
    # i_data = iq_reshaped[:,0]
    # q_data = iq_reshaped[:,1]

    # iq_array = np.array([i_data, q_data])

    # with open(IQFileName,'wb') as f:
    #     pickle.dump(iq_array,f)
    #     f.close()

if __name__ == '__main__':
    main()


