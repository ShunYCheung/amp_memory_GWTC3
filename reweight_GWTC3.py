from reweight_mem_parallel import reweight_mem_parallel
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result
from pathfinder import extract_files
from calculate_snr_simple import calculate_amp_vs_snr
from waveforms import mem_freq_XPHM

import bilby
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import copy


if __name__ == '__main__':
    cpus = int(sys.argv[1])
    event_name = str(sys.argv[2])
    amplitude = float(sys.argv[3])
    outdir = str(sys.argv[4])
    waveform = 'IMRPhenomXPHM'
    print('Amplitude = ', amplitude)

    file_path, data_file = extract_files(event_name)
    print(f"opening {file_path}")
    
    extension = os.path.splitext(file_path)[1].lstrip('.')
    if 'h5' in extension:
        samples_dict, meta_dict, config_dict, priors_dict, psds = create_post_dict(file_path, waveform)
        args = extract_relevant_info(meta_dict, config_dict)
    elif 'json' in extension:
        result = bilby.core.result.read_in_result(file_path)
        samples_dict = result.posterior
        args = process_bilby_result(result.meta_data['command_line_args'])
        priors_dict = result.priors
        psds=None
    else:
        print('Cannot recognise file type.')
        exit()
    
    # amplitudes = np.arange(0, 400, 20)
    # result = calculate_amp_vs_snr(event_name, amplitudes, samples_dict, priors_dict, psds, data_file, args, mem_freq_XPHM, outdir=None)
    # plt.figure()
    # plt.plot(result[:, 0], result[:, 1])
    # plt.xlabel('A')
    # plt.ylabel('optimal SNR')
    # plt.xlim(0, max(result[:, 0]))
    # plt.savefig('tests/test_results/GW170818_amp_full_waveform_snr.png')
    print("reweighting {}".format(event_name))

    weights, bf = reweight_mem_parallel(event_name, 
                                        samples_dict, 
                                        args,
                                        priors_dict,
                                        outdir + event_name,
                                        "weights_{}".format(event_name), 
                                        amplitude = amplitude,
                                        data_file=data_file,
                                        psds = psds,
                                        n_parallel=cpus,)
    
        
        

        
        
        
        
        
        