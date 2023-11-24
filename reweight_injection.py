from reweight_mem_parallel import reweight_mem_parallel
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result

import bilby
import sys
import os
import glob

cpus = int(sys.argv[1])
event_number = int(sys.argv[2])
amplitude = float(sys.argv[3])
print(amplitude)


if __name__ == '__main__':
    
    path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/PE_result_files/*.json")
    for path in path_list:
        number = path.split("run") 
        number2 = number[1].split("_")[0]
        if int(number2) == event_number:
            file_path = path

    print(f"opening {file_path}")
    
    data_path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/data_dump/*.pickle")
    for data_path in data_path_list:
        number = data_path.split("run") 
        number2 = number[1].split("_")[0]
        if int(number2) == event_number:
            data_file = data_path
    print(f"opening {data_file}")
    
    result = bilby.core.result.read_in_result(file_path)
    samples_dict = result.posterior
    args = process_bilby_result(result.meta_data['command_line_args'])
    priors_dict = result.priors
    psds=None
    event_name = f'injection{event_number}'
    TD_path = "/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/GW170818_injection_LIGO_data/TD.npz"
            
    print("reweighting {}".format(event_name))
    weights, bf = reweight_mem_parallel(event_name, 
                                        samples_dict, 
                                        args,
                                        priors_dict,
                                        "/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/reweighting_results/{}".format(event_name),
                                        "weights_{}".format(event_name), 
                                        amplitude = amplitude,
                                        data_file=data_file,
                                        TD_path = TD_path,
                                        psds = psds,
                                        calibration = None,
                                        n_parallel=cpus)