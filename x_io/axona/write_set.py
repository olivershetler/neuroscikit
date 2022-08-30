def write_set(filename, session_parameters):
    session_path, session_filename = os.path.split(filename)
    trial_date, trial_time = "na", "na"

    if session_parameters['n_channels'] >= 128:
        mode = 1
    else:
        mode = 0

    with open(filename, 'w') as f:
        date = 'trial_date %s' % (trial_date)
        time_head = '\ntrial_time %s' % (trial_time)
        experimenter = '\nexperimenter %s' % (session_parameters['experimenter'])
        comments = '\ncomments %s' % (session_parameters['comments'])
        duration = '\nduration %d' % (session_parameters['duration'])

        sw_vers = '\nsw_version %s' % (session_parameters['version'])
        write_order = [date, time_head, experimenter, comments, duration, sw_vers]

        lines = ['\nADC_fullscale_mv %s' % session_parameters['fullscale'], # +/- 5mv
                 '\ntracker_version 0',
                 '\nstim_version 1',
                 '\naudio_version 0']

        write_order += lines
        gains = session_parameters['gain']
        for chan in range(0, session_parameters['n_channels']):
            lines = ['\ngain_ch_%d %d' % (chan, gains[chan]),
                     '\nfilter_ch_%d 2' % (chan),
                     '\na_in_ch_%d %d' % (chan, chan),
                     '\nb_in_ch_%d 15' % (chan),
                     '\nmode_ch_%d 5' % (chan),
                     '\nfiltresp_ch_%d 2' % (chan),
                     '\nfiltkind_ch_%d 0' % (chan),
                     '\nfiltfreq1_ch_%d 300' % (chan),
                     '\nfiltfreq2_ch_%d 7000' % (chan),
                     '\nfiltripple_ch_%d 0.10' % (chan),
                     '\nfiltdcblock_ch_%d 1' % (chan),
                     '\ndispmode_ch_%d 1' % (chan),
                     '\nchanname_ch_%d' % (chan)]
            write_order += lines

        write_order += ['\nsecond_audio 3',
                 '\ndefault_filtresp_hp 2',
                 '\ndefault_filtkind_hp 0',
                 '\ndefault_filtfreq1_hp %d' % (300),
                 '\ndefault_filtfreq2_hp %d' % (7000),
                 '\ndefault_filtripple_hp %.1f' % (0.1),
                 '\ndefault_filtdcblock_hp %d' % (1),
                 '\ndefault_filtresp_lp 0',
                 '\ndefault_filtkind_lp 1',
                 '\ndefault_filtfreq1_lp %d' % (500),
                 '\ndefault_filtfreq2_lp 0',
                 '\ndefault_filtripple_lp %.1f' % (0.10),
                 '\ndefault_filtdcblock_lp %d' % (1),
                 '\nnotch_frequency %d' % (session_parameters['notch_frequency']),
                 '\nref_0 4',
                 '\nref_1 5',
                 '\nref_2 0',
                 '\nref_3 2',
                 '\nref_4 3',
                 '\nref_5 7',
                 '\nref_6 6',
                 '\nref_7 0',
                 '\ntrigger_chan 5',
                 '\nselected_slot 5',
                 '\nsweeprate 5',
                 '\ntrig_point 3',
                 '\ntrig_slope 1',
                 '\nthreshold 13312',
                 '\nleftthreshold 0',
                 '\nrightthreshold 0',
                 '\naud_threshold 1',
                 '\nchan_group 1',
                 '\ngroups_1_0 0',
                 '\ngroups_1_1 1',
                 '\ngroups_1_2 2',
                 '\ngroups_1_3 3',
                 '\ngroups_1_4 -1',
                 '\ngroups_1_5 4',
                 '\ngroups_1_6 5',
                 '\ngroups_1_7 6',
                 '\ngroups_1_8 7',
                 '\ngroups_1_9 -1',
                 '\ngroups_2_0 8',
                 '\ngroups_2_1 9',
                 '\ngroups_2_2 10',
                 '\ngroups_2_3 11',
                 '\ngroups_2_4 -1',
                 '\ngroups_2_5 12',
                 '\ngroups_2_6 13',
                 '\ngroups_2_7 14',
                 '\ngroups_2_8 15',
                 '\ngroups_2_9 -1',
                 '\ngroups_3_0 16',
                 '\ngroups_3_1 17',
                 '\ngroups_3_6 21',
                 '\ngroups_3_7 22',
                 '\ngroups_3_8 23',
                 '\ngroups_3_9 -1',
                 '\ngroups_4_0 24',
                 '\ngroups_4_1 25',
                 '\ngroups_4_2 26',
                 '\ngroups_4_3 27',
                 '\ngroups_4_4 -1',
                 '\ngroups_4_5 28',
                 '\ngroups_4_6 29',
                 '\ngroups_4_7 30',
                 '\ngroups_4_8 31',
                 '\ngroups_4_9 -1',
                 '\ngroups_5_0 0',
                 '\ngroups_5_1 1',
                 '\ngroups_5_2 2',
                 '\ngroups_5_3 3',
                 '\ngroups_5_4 -1',
                 '\ngroups_5_5 4',
                 '\ngroups_5_6 5',
                 '\ngroups_5_7 6',
                 '\ngroups_5_8 7',
                 '\ngroups_5_9 -1',
                 '\ngroups_6_0 40',
                 '\ngroups_6_1 41',
                 '\ngroups_6_2 42',
                 '\ngroups_6_3 43',
                 '\ngroups_6_4 -1',
                 '\ngroups_6_5 44',
                 '\ngroups_6_6 45',
                 '\ngroups_6_7 46',
                 '\ngroups_6_8 47',
                 '\ngroups_6_9 -1',
                 '\ngroups_7_0 48',
                 '\ngroups_7_1 49',
                 '\ngroups_7_2 50',
                 '\ngroups_7_3 51',
                 '\ngroups_7_4 -1',
                 '\ngroups_7_5 52',
                 '\ngroups_7_6 53',
                 '\ngroups_7_7 54',
                 '\ngroups_7_8 55',
                 '\ngroups_7_9 -1',
                 '\ngroups_8_0 56',
                 '\ngroups_8_1 57',
                 '\ngroups_8_2 58',
                 '\ngroups_8_3 59',
                 '\ngroups_8_4 -1',
                 '\ngroups_8_5 60',
                 '\ngroups_8_6 61',
                 '\ngroups_8_7 62',
                 '\ngroups_8_8 63',
                 '\ngroups_8_9 -1',
                 '\ngroups_9_0 -1',
                 '\ngroups_9_1 -1',
                 '\ngroups_9_2 -1',
                 '\ngroups_9_3 -1',
                 '\ngroups_9_4 -1',
                 '\ngroups_9_5 -1',
                 '\ngroups_9_6 -1',
                 '\ngroups_9_7 -1',
                 '\ngroups_9_8 -1',
                 '\ngroups_9_9 -1',
                 '\ngroups_10_0 72',
                 '\ngroups_10_1 73',
                 '\ngroups_10_2 74',
                 '\ngroups_10_3 75',
                 '\ngroups_10_4 -1',
                 '\ngroups_10_5 76',
                 '\ngroups_10_6 77',
                 '\ngroups_10_7 78',
                 '\ngroups_10_8 79',
                 '\ngroups_10_9 -1',
                 '\ngroups_11_0 80',
                 '\ngroups_11_1 81',
                 '\ngroups_11_2 82',
                 '\ngroups_11_3 83',
                 '\ngroups_11_4 -1',
                 '\ngroups_11_5 84',
                 '\ngroups_11_6 85',
                 '\ngroups_11_7 86',
                 '\ngroups_11_8 87',
                 '\ngroups_11_9 -1',
                 '\ngroups_12_0 88',
                 '\ngroups_12_1 89',
                 '\ngroups_12_2 90',
                 '\ngroups_12_3 91',
                 '\ngroups_12_4 -1',
                 '\ngroups_12_5 92',
                 '\ngroups_12_6 93',
                 '\ngroups_12_7 94',
                 '\ngroups_12_8 95',
                 '\ngroups_12_9 -1',
                 '\ngroups_13_0 96',
                 '\ngroups_13_1 97',
                 '\ngroups_13_2 98',
                 '\ngroups_13_3 99',
                 '\ngroups_13_4 -1',
                 '\ngroups_13_5 100',
                 '\ngroups_13_6 101',
                 '\ngroups_13_7 102',
                 '\ngroups_13_8 103',
                 '\ngroups_13_9 -1',
                 '\ngroups_14_0 104',
                 '\ngroups_14_1 105',
                 '\ngroups_14_2 106',
                 '\ngroups_14_3 107',
                 '\ngroups_14_4 -1',
                 '\ngroups_14_5 108',
                 '\ngroups_14_6 109',
                 '\ngroups_14_7 110',
                 '\ngroups_14_8 111',
                 '\ngroups_14_9 -1',
                 '\ngroups_15_0 112',
                 '\ngroups_15_1 113',
                 '\ngroups_15_2 114',
                 '\ngroups_15_3 115',
                 '\ngroups_15_4 -1',
                 '\ngroups_15_5 116',
                 '\ngroups_15_6 117',
                 '\ngroups_15_7 118',
                 '\ngroups_15_8 119',
                 '\ngroups_15_9 -1',
                 '\ngroups_16_0 120',
                 '\ngroups_16_1 121',
                 '\ngroups_16_2 122',
                 '\ngroups_16_3 123',
                 '\ngroups_16_4 -1',
                 '\ngroups_16_5 124',
                 '\ngroups_16_6 125',
                 '\ngroups_16_7 126',
                 '\ngroups_16_8 127',
                 '\ngroups_16_9 -1',
                 '\ngroups_17_0 -1',
                 '\ngroups_17_1 -1',
                 '\ngroups_17_2 -1',
                 '\ngroups_17_3 0',
                 '\ngroups_17_4 -1',
                 '\ngroups_17_5 -1',
                 '\ngroups_17_6 -1',
                 '\ngroups_17_7 -1',
                 '\ngroups_17_8 -1',
                 '\ngroups_17_9 -1',
                 '\nslot_chan_0 0',
                 '\nslot_chan_1 1',
                 '\nslot_chan_2 2',
                 '\nslot_chan_3 3',
                 '\nslot_chan_4 -1',
                 '\nslot_chan_5 4',
                 '\nslot_chan_6 5',
                 '\nslot_chan_7 6',
                 '\nslot_chan_8 7',
                 '\nslot_chan_9 -1',
                 '\nextin_port 4',
                 '\nextin_bit 5',
                 '\nextin_edge 1',
                 '\ntrigholdwait 1',
                 '\noverlap 0',
                 '\nxmin %s' % session_parameters['window_xmin'],
                 '\nxmax %s' % session_parameters['window_xmax'],
                 '\nymin %s' % session_parameters['window_ymin'],
                 '\nymax %s' % session_parameters['window_ymax'],
                 '\nbrightness 248',
                 '\ncontrast 112',
                 '\nsaturation 110',
                 '\nhue 238',
                 '\ngamma 0',
                 '\ncolmap_1_rmin 100',
                 '\ncolmap_1_rmax 100',
                 '\ncolmap_1_gmin 1',
                 '\ncolmap_1_gmax 10',
                 '\ncolmap_1_bmin 5',
                 '\ncolmap_1_bmax 0',
                 '\ncolmap_2_rmin 0',
                 '\ncolmap_2_rmax 0',
                 '\ncolmap_2_gmin 2',
                 '\ncolmap_2_gmax 31',
                 '\ncolmap_2_bmin 0',
                 '\ncolmap_2_bmax 0',
                 '\ncolmap_3_rmin 5',
                 '\ncolmap_3_rmax 0',
                 '\ncolmap_3_gmin 0',
                 '\ncolmap_3_gmax 5',
                 '\ncolmap_3_bmin 11',
                 '\ncolmap_3_bmax 0',
                 '\ncolmap_4_rmin 2',
                 '\ncolmap_4_rmax 20',
                 '\ncolmap_4_gmin 0',
                 '\ncolmap_4_gmax 0',
                 '\ncolmap_4_bmin 0',
                 '\ncolmap_4_bmax 0',
                 '\ncolactive_1 1',
                 '\ncolactive_2 0',
                 '\ncolactive_3 0',
                 '\ncolactive_4 0',
                 '\ntracked_spots 1',
                 '\ncolmap_algorithm 1',
                 '\ncluster_delta 10',
                 '\ntracker_pixels_per_metre %s' % (session_parameters['ppm']),
                 '\ntwo_cameras 0',
                 '\nxcoordsrc 0',
                 '\nycoordsrc 1',
                 '\nzcoordsrc 3',
                 '\ntwocammode 0',
                 '\nstim_pwidth 500000',
                 '\nstim_pamp 1',
                 '\nstim_pperiod 1000000',
                 '\nstim_prepeat 0',
                 '\nstim_tnumber 1',
                 '\nstim_tperiod 1000',
                 '\nstim_trepeat 0',
                 '\nstim_bnumber 1',
                 '\nstim_bperiod 1000000',
                 '\nstim_brepeat 0',
                 '\nstim_gnumber 1',
                 '\nsingle_pulse_width 100',
                 '\nsingle_pulse_amp 100000',
                 '\nstim_patternmask_1 1',
                 '\nstim_patterntimes_1 600',
                 '\nstim_patternnames_1 Baseline 100 æs pulse every 30 s',
                 '\nstim_patternmask_2 0',
                 '\nstim_patterntimes_2 0',
                 '\nstim_patternnames_2 pause (no stimulation)',
                 '\nstim_patternmask_3 0',
                 '\nstim_patterntimes_3 0',
                 '\nstim_patternnames_3 pause (no stimulation)',
                 '\nstim_patternmask_4 0',
                 '\nstim_patterntimes_4 0',
                 '\nstim_patternnames_4 pause (no stimulation)',
                 '\nstim_patternmask_5 0',
                 '\nstim_patterntimes_5 0',
                 '\nstim_patternnames_5 pause (no stimulation)',
                 '\nscopestimtrig 1',
                 '\nstim_start_delay 1',
                 '\nbiphasic 1',
                 '\nuse_dacstim 0',
                 '\nstimscript 0',
                 '\nstimfile ',
                 '\nnumPatterns 1',
                 '\nstim_patt_1 "One 100 us pulse every 30 s" 100 100 ',
                 '\n30000000 0 1 1000 0 1 1000000 0 1',
                 '\nnumProtocols 1',
                 '\nstim_prot_1 "Ten minutes of 30 s pulse baseline" 1 ',
                 '\n600 "One 100 us pulse every 30 s" 0 0 "Pause (no ',
                 '\nstimulation)" 0 0 "Pause (no stimulation)" 0 0 ',
                 '\n"Pause (no stimulation)" 0 0 "Pause (no ',
                 '\nstimulation)"',
                 '\nstim_during_rec 0',
                 '\ninfo_subject ',
                 '\ninfo_trial ',
                 '\nwaveform_period 32',
                 '\npretrig_period 1',
                 '\ndeadzone_period 500',
                 '\nfieldtrig 0',
                 '\nsa_manauto 1',
                 '\nsl_levlat 0',
                 '\nsp_manauto 0',
                 '\nvsa_time 1.00000',
                 '\nsl_levstart 0.00000',
                 '\nsl_levend 0.50000',
                 '\nsl_latstart 2.00000',
                 '\nsl_latend 2.50000',
                 '\nsp_startt 3.00000',
                 '\nsp_endt 10.00000',
                 '\nresp_endt 32.00000',
                 '\nrecordcol 4']

        f.seek(0, 2)
        f.writelines(write_order)

        for chan in range(0, int(session_parameters['n_channels'] / 4)):
            line = '\ncollectMask_%d %d' % (chan + 1, 1)
            f.seek(0, 2)
            f.write(line)

        for chan in range(0, int(session_parameters['n_channels'] / 4)):
            line = '\nstereoMask_%d %d' % (chan + 1, 0)
            f.seek(0, 2)
            f.write(line)

        for chan in range(0, int(session_parameters['n_channels'] / 4)):
            line = '\nmonoMask_%d %d' % (chan + 1, 0)
            f.seek(0, 2)
            f.write(line)

        for chan in range(0, int(session_parameters['n_channels'] / 4)):
            line = '\nEEGmap_%d %d' % (chan + 1, 1)
            f.seek(0, 2)
            f.write(line)

        for chan in range(0, int(session_parameters['n_channels'])):
            if int(chan) + 1 == 1:
                line = ['\nEEG_ch_%d %d' % (chan + 1, chan + 1),
                        '\nsaveEEG_ch_%d %d' % (chan + 1, 1),
                        '\nnullEEG %d' % (0)]
            else:
                line = ['\nEEG_ch_%d %d' % (chan + 1, chan + 1),
                        '\nsaveEEG_ch_%d %d' % (chan + 1, 1)]
            f.seek(0, 2)
            f.writelines(line)

        lines = ['\nEEGdisplay 0',
                 '\nlightBearing_1 0',
                 '\nlightBearing_2 0',
                 '\nlightBearing_3 0',
                 '\nlightBearing_4 0',
                 '\nartefactReject 1',
                 '\nartefactRejectSave 0',
                 '\nremoteStart 1',
                 '\nremoteChan 16',
                 '\nremoteStop 0',
                 '\nremoteStopChan 14',
                 '\nendBeep 1',
                 '\nrecordExtin 0',
                 '\nrecordTracker 1',
                 '\nshowTracker 1',
                 '\ntrackerSerial 0',
                 '\nserialColour 0',
                 '\nrecordVideo 0',
                 '\ndacqtrackPos 0',
                 '\nstimSerial 0',
                 '\nrecordSerial 0',
                 '\nuseScript 0',
                 '\nscript C:\\Users\\Rig-432\\Desktop\\test.ba',
                 '\npostProcess 0',
                 '\npostProcessor ',
                 '\npostProcessorParams ',
                 '\nsync_out 0',
                 '\nsyncRate 25.00000',
                 '\nmark_out 1',
                 '\nmarkChan 16',
                 '\nsyncDelay 0',
                 '\nautoTrial 0',
                 '\nnumTrials 10',
                 '\ntrialPrefix trial',
                 '\nautoPrompt 0',
                 '\ntrigMode 0',
                 '\ntrigChan 1',
                 '\nsaveEGF 0',
                 '\nrejstart %d' % session_parameters['rejstart'],
                 '\nrejthreshtail %d' % session_parameters['rejthreshtail'],
                 '\nrejthreshupper %d' % session_parameters['rejthreshupper'],
                 '\nrejthreshlower %d' % session_parameters['rejthreshlower'],
                 '\nrawGate 0',
                 '\nrawGateChan 0',
                 '\nrawGatePol 1',
                 '\ndefaultTime 600',
                 '\ndefaultMode 0',
                 '\ntrial_comment ',
                 '\nexperimenter %s' % (session_parameters['experimenter']),
                 '\ndigout_state 32768',
                 '\nstim_phase 90',
                 '\nstim_period 100',
                 '\nbp1lowcut 0',
                 '\nbp1highcut 10',
                 '\nthresh_lookback 2',
                 '\npalette C:\DACQ\default.gvp',
                 '\ncheckUpdates 0',
                 '\nSpike2msMode 0',
                 '\nDIOTimeBase 0',
                 '\npretrigSamps  %d' % (session_parameters['pretrigSamps']),
                 '\nspikeLockout %d' % (session_parameters['spikeLockout']),
                 '\nBPFspikelen 2',
                 '\nBPFspikeLockout 86']
        f.seek(0, 2)
        f.writelines(lines)

        for chan in range(0, int(session_parameters['n_channels'])):
            if 30 + int(chan) >= 31:
                line = '\nBPFEEG_ch_%d %d' % (chan + 1, 31 + int(chan))
            else:
                line = '\nBPFEEG_ch_%d %d' % (chan + 1, 30 + int(chan))
            f.seek(0, 2)
            f.write(line)

        lines = ['\nBPFrecord1 1',
                 '\nBPFrecord2 0',
                 '\nBPFrecord3 0',
                 '\nBPFbit1 0',
                 '\nBPFbit2 1',
                 '\nBPFbit3 2',
                 '\nBPFEEGin1 28',
                 '\nBPFEEGin2 27',
                 '\nBPFEEGin3 26',
                 '\nBPFsyncin1 31',
                 '\nBPFrecordSyncin1 1',
                 '\nBPFunitrecord 0',
                 '\nBPFinsightmode 0',
                 '\nBPFcaladjust 1.00000000',
                 '\nBPFcaladjustmode 0',
                 '\nrawRate %d' % (session_parameters['Fs']),
                 '\nRawRename 1',
                 '\nRawScope 1',
                 '\nRawScopeMode 0',
                 '\nmuxhs_fast_settle_en 0',
                 '\nmuxhs_fast_settle_chan 0',
                 '\nmuxhs_ext_out_en 0',
                 '\nmuxhs_ext_out_chan 0',
                 '\nmuxhs_cable_delay 4',
                 '\nmuxhs_ttl_mode 0',
                 '\nmuxhs_ttl_out 0',
                 '\nmuxhs_upper_bw 7000.00000',
                 '\nmuxhs_lower_bw 0.70000',
                 '\nmuxhs_dsp_offset_en 0',
                 '\nmuxhs_dsp_offset_freq 13',
                 '\ndemux_dac_manual 32768',
                 '\ndemux_en_dac_1 0',
                 '\ndemux_src_dac_1 0',
                 '\ndemux_gain_dac_1 0',
                 '\ndemux_noise_dac_1 0',
                 '\ndemux_enhpf_dac_1 0',
                 '\ndemux_hpfreq_dac_1 300.00000',
                 '\ndemux_thresh_dac_1 32768',
                 '\ndemux_polarity_dac_1 1',
                 '\ndemux_en_dac_2 0',
                 '\ndemux_src_dac_2 1',
                 '\ndemux_gain_dac_2 0',
                 '\ndemux_noise_dac_2 0',
                 '\ndemux_enhpf_dac_2 0',
                 '\ndemux_hpfreq_dac_2 300.00000',
                 '\ndemux_thresh_dac_2 32768',
                 '\ndemux_polarity_dac_2 1',
                 '\ndemux_en_dac_3 0',
                 '\ndemux_src_dac_3 2',
                 '\ndemux_gain_dac_3 0',
                 '\ndemux_noise_dac_3 0',
                 '\ndemux_enhpf_dac_3 0,'
                 '\ndemux_hpfreq_dac_3 300.00000',
                 '\ndemux_thresh_dac_3 32768',
                 '\ndemux_polarity_dac_3 1',
                 '\ndemux_en_dac_4 0',
                 '\ndemux_src_dac_4 3',
                 '\ndemux_gain_dac_4 0',
                 '\ndemux_noise_dac_4 0',
                 '\ndemux_enhpf_dac_4 0',
                 '\ndemux_hpfreq_dac_4 300.00000',
                 '\ndemux_thresh_dac_4 32768',
                 '\ndemux_polarity_dac_4 1',
                 '\ndemux_en_dac_5 0',
                 '\ndemux_src_dac_5 4',
                 '\ndemux_gain_dac_5 0',
                 '\ndemux_noise_dac_5 0',
                 '\ndemux_enhpf_dac_5 0',
                 '\ndemux_hpfreq_dac_5 300.00000',
                 '\ndemux_thresh_dac_5 32768',
                 '\ndemux_polarity_dac_5 1',
                 '\ndemux_en_dac_6 0',
                 '\ndemux_src_dac_6 5',
                 '\ndemux_gain_dac_6 0',
                 '\ndemux_noise_dac_6 0',
                 '\ndemux_enhpf_dac_6 0',
                 '\ndemux_hpfreq_dac_6 300.00000',
                 '\ndemux_thresh_dac_6 32768',
                 '\ndemux_polarity_dac_6 1',
                 '\ndemux_en_dac_7 0',
                 '\ndemux_src_dac_7 6',
                 '\ndemux_gain_dac_7 0',
                 '\ndemux_noise_dac_7 0',
                 '\ndemux_enhpf_dac_7 0',
                 '\ndemux_hpfreq_dac_7 300.00000',
                 '\ndemux_thresh_dac_7 32768',
                 '\ndemux_polarity_dac_7 1',
                 '\ndemux_en_dac_8 0',
                 '\ndemux_src_dac_8 7',
                 '\ndemux_gain_dac_8 0',
                 '\ndemux_noise_dac_8 0',
                 '\ndemux_enhpf_dac_8 0',
                 '\ndemux_hpfreq_dac_8 300.00000',
                 '\ndemux_thresh_dac_8 32768',
                 '\ndemux_polarity_dac_8 1',
                 '\ndemux_adc_in_1 -1',
                 '\ndemux_adc_in_2 -1',
                 '\ndemux_adc_in_3 -1',
                 '\ndemux_adc_in_4 -1',
                 '\ndemux_adc_in_5 -1',
                 '\ndemux_adc_in_6 -1',
                 '\ndemux_adc_in_7 -1',
                 '\ndemux_adc_in_8 -1',
                 '\ndemux_ttlin_ch -1',
                 '\ndemux_ttlout_ch -1',
                 '\ndemux_ttlinouthi_ch -1',
                 '\nlastfileext set',
                 '\nlasttrialdatetime 1470311661',
                 '\nlastupdatecheck 0',
                 '\nuseupdateproxy 0',
                 '\nupdateproxy ',
                 '\nupdateproxyid ',
                 '\nupdateproxypw ',
                 '\ncontaudio 0',
                 '\nmode128channels %d' % mode,
                 '\nmodeanalog32 0',
                 '\nmodemux 0',
                 '\nIMUboard 0']
        f.seek(0, 2)
        f.writelines(lines)

def get_session_parameters(pos_file, rhd_file):

    session_parameters = {}
    intan_headers = read_data(rhd_file)

    with open(pos_file, 'rb+') as f:  # opening the .pos file
        for line in f:  # reads line by line to read the header of the file
            if 'comments' in str(line):
                session_parameters['comments'] = line.decode(encoding='UTF-8')[len('comments '):-2]
            elif 'pixels_per_metre' in str(line):
                session_parameters['ppm'] = float(line.decode(encoding='UTF-8')[len('pixels_per_metre '):-2])
            elif 'sw_version' in str(line):
                session_parameters['version'] = line.decode(encoding='UTF-8')[len('sw_version '):-2]
            elif 'experimenter' in str(line):
                session_parameters['experimenter'] = line.decode(encoding='UTF-8')[len('experimenter '):-2]
            elif 'min_x' in str(line) and 'window' not in str(line):
                session_parameters['xmin'] = int(line.decode(encoding='UTF-8')[len('min_x '):-2])
            elif 'max_x' in str(line) and 'window' not in str(line):
                session_parameters['xmax'] = int(line.decode(encoding='UTF-8')[len('max_x '):-2])
            elif 'min_y' in str(line) and 'window' not in str(line):
                session_parameters['ymin'] = int(line.decode(encoding='UTF-8')[len('min_y '):-2])
            elif 'max_y' in str(line) and 'window' not in str(line):
                session_parameters['ymax'] = int(line.decode(encoding='UTF-8')[len('max_y '):-2])

            elif 'window_min_x' in str(line):
                session_parameters['window_xmin'] = int(line.decode(encoding='UTF-8')[len('window_min_x '):-2])

            elif 'window_max_x' in str(line):
                session_parameters['window_xmax'] = int(line.decode(encoding='UTF-8')[len('window_max_x '):-2])

            elif 'window_min_y' in str(line):
                session_parameters['window_ymin'] = int(line.decode(encoding='UTF-8')[len('window_min_y '):-2])

            elif 'window_max_y' in str(line):
                session_parameters['window_ymax'] = int(line.decode(encoding='UTF-8')[len('window_max_y '):-2])

            elif 'duration' in str(line):
                session_parameters['duration'] = int(line.decode(encoding='UTF-8')[len('duration '):-2])

            if 'data_start' in str(line):
                break

        # ----------------
        n_channels = len(intan_headers['amplifier_channels'])
        notch_filter_setting = intan_headers['frequency_parameters']['notch_filter_frequency']

        ADC_Fullscale = 1500

        session_parameters['gain'] = np.zeros(n_channels)
        session_parameters['fullscale'] = ADC_Fullscale
        session_parameters['n_channels'] = n_channels
        session_parameters['notch_frequency'] = notch_filter_setting
        session_parameters['pretrigSamps'] = 10
        session_parameters['spikeLockout'] = 40
        session_parameters['rejthreshtail'] = 43
        session_parameters['rejstart'] = 30
        session_parameters['rejthreshupper'] = 100
        session_parameters['rejthreshlower'] = -100
        session_parameters['Fs'] = 48e3

    return session_parameters