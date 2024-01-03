import bilby
import copy
import numpy as np
import pickle
import glob
import sys
sys.path.append("/home/shunyin.cheung/amp_memory_GWTC3/")

from create_post_dict import process_bilby_result
from waveforms import mem_freq_XPHM_only


def calculate_amp_vs_snr(event_name, amplitudes, samples, priors, data_file, args, source_model, outdir):


    print("opening {}".format(data_file))
    with open(data_file, 'rb') as f:
        data_dump = pickle.load(f)
    try:
        ifo_list = data_dump.interferometers

    except AttributeError:
        ifo_list = data_dump['ifo_list']

    sampling_frequency = ifo_list.sampling_frequency
    maximum_frequency = args['maximum_frequency']
    minimum_frequency = args['minimum_frequency']
    reference_frequency = args['reference_frequency']
    roll_off = args['tukey_roll_off']
    duration = args['duration']
    
    waveform_name = args['waveform_approximant']
    event_snr = []

    max_like = np.argmax(samples['log_likelihood'])
    parameter = samples.iloc[max_like].to_dict()

    if np.iscomplexobj(parameter['mass_2']):
        for keys in parameter:
            parameter[keys] = float(np.real(parameter[keys]))

    for amplitude in amplitudes:
        print('amplitude = ',amplitude)
        priors2 = copy.copy(priors)
        # test if bilby oscillatory waveform = gwmemory oscillatory waveform.

        
        waveform_generator_mem = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model= source_model,
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(duration=duration,
                                    roll_off=roll_off,
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    sampling_frequency=sampling_frequency,
                                    reference_frequency=reference_frequency,
                                    waveform_approximant = waveform_name,
                                    amplitude=amplitude)

        )
        
        if args['time_marginalization']=="True":
            print('time marginalisation on')
            time_marginalization = True
            jitter_time = True
        else:
            time_marginalization = False
            jitter_time = False
        
        if args['distance_marginalization']=="True":
            print('distance marginalisation on')
            distance_marginalization = True
        else:
            distance_marginalization = False
        if args['time_marginalization']:
            print('time marginalisation on')
            time_marginalization = True
            jitter_time = True
        else:
            time_marginalization = False
            jitter_time = False
        
        if args['distance_marginalization']:
            print('distance marginalisation on')
            distance_marginalization = True
        else:
            distance_marginalization = False

        target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            ifo_list,
            waveform_generator_mem,
            time_marginalization = time_marginalization,
            distance_marginalization = distance_marginalization,
            #distance_marginalization_lookup_table = TD_path,
            jitter_time=jitter_time,
            priors = priors2,
            reference_frame = args['reference_frame'],
            time_reference = args['time_reference'],
        )

        frequency_domain_strain = waveform_generator_mem.frequency_domain_strain(parameter)

        target_likelihood.parameters.update(parameter)

        if len(ifo_list) == 2:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            snr_array_L1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[1])
            opt_snr_sq_H = snr_array_H1.optimal_snr_squared
            opt_snr_sq_L = snr_array_L1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_sq_H+opt_snr_sq_L)
        elif len(ifo_list) == 1:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            opt_snr_sq_H = snr_array_H1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_sq_H)
        elif len(ifo_list) == 3:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            snr_array_L1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[1])
            snr_array_V1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[2])
            opt_snr_sq_H = snr_array_H1.optimal_snr_squared
            opt_snr_sq_L = snr_array_L1.optimal_snr_squared
            opt_snr_sq_V = snr_array_V1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_sq_H+opt_snr_sq_L+opt_snr_sq_V)
        else:
            print('¯\_(ツ)_/¯')
            exit()
        event_snr.append(opt_snr)

    result = np.stack((np.array(amplitudes), np.array(event_snr)), axis=1)
    np.savetxt(f'{outdir}/{event_name}/{event_name}_memory_snr_vs_amp.csv', result)


###############################################
# for event_number in range(60, 101):
#     path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/PE_result_files/*.json")
#     for path in path_list:
#         number = path.split("run") 
#         number2 = number[1].split("_")[0]
#         if int(number2) == event_number:
#             file_path = path


#     print(f"opening {file_path}")

#     data_path_list = glob.glob("/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/data_dump/*.pickle")
#     for data_path in data_path_list:
#         number = data_path.split("run") 
#         number2 = number[1].split("_")[0]
#         if int(number2) == event_number:
#             data_file = data_path
#     print(f"opening {data_file}")

#     result = bilby.core.result.read_in_result(file_path)
#     samples_dict = result.posterior
#     args = process_bilby_result(result.meta_data['command_line_args'])
#     priors_dict = result.priors
#     psds=None
#     event_name = f'injection{event_number}'
#     TD_path = "/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/GW170818_injection_LIGO_data/TD.npz"

#     amplitudes = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#     source_model = mem_freq_XPHM_only
#     outdir = 'injection_studies/posterior_results'

#     calculate_amp_vs_snr(event_name, amplitudes, samples_dict, priors_dict, data_file, args, source_model, outdir)