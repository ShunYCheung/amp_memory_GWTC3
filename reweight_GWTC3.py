from reweight_mem_parallel import reweight_mem_parallel
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result
from event_table import call_event_table

import json
import bilby
import sys
import os

cpus = int(sys.argv[1])
event_number = int(sys.argv[2])
amplitude = float(sys.argv[3])
print(amplitude)


if __name__ == '__main__':
    event_name, file_path, trigger_time, duration, waveform, data_file = \
        call_event_table()[event_number]
    print(f"opening {file_path}")
    
    extension = os.path.splitext(file_path)[1].lstrip('.')
    if 'h5' in extension:
        samples_dict, meta_dict, config_dict, priors_dict, psds, calibration = create_post_dict(file_path, waveform)
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
            
    print("reweighting {}".format(event_name))
    weights, bf = reweight_mem_parallel(event_name, 
                                        samples_dict, 
                                        args,
                                        priors_dict,
                                        "/home/shunyin.cheung/amp_memory_GWTC3/run2/{}".format(event_name),
                                        "weights_{}".format(event_name), 
                                        amplitude = amplitude,
                                        data_file=data_file,
                                        psds = psds,
                                        calibration = None,
                                        n_parallel=cpus)
        
        

        
        
        
        
        
        