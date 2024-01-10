import bilby
import copy
import numpy as np
import pickle
import sys
import os
sys.path.append("/home/shunyin.cheung/amp_memory_GWTC3/")

from pathfinder import extract_files
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result
from waveforms import mem_freq_XPHM_only
from reweight_mem_parallel import call_data_GWOSC


def calculate_network_snr(ifo_list, waveform_generator, target_likelihood, parameter):
    frequency_domain_strain = waveform_generator.frequency_domain_strain(parameter)

    target_likelihood.parameters.update(parameter)

    optimal_snr_sq_list = []
    for ifo in ifo_list:
        snr_array = target_likelihood.calculate_snrs(frequency_domain_strain, ifo)
        optimal_snr_sq = snr_array.optimal_snr_squared
        optimal_snr_sq_list.append(optimal_snr_sq)
    optimal_snr = np.sqrt(sum(optimal_snr_sq_list))
    return optimal_snr


def calculate_amp_vs_snr(event_name, amplitudes, samples, priors, psds, data_file, args, source_model, outdir=None):


    logger = bilby.core.utils.logger

    # adds in detectors and the specs for the detectors. 
    if data_file is not None:
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
    else:
        sampling_frequency = args['sampling_frequency']
        maximum_frequency = args['maximum_frequency']
        minimum_frequency = args['minimum_frequency']
        reference_frequency = args['reference_frequency']
        roll_off = args['tukey_roll_off']
        duration = args['duration']
        post_trigger_duration = args['post_trigger_duration']
        trigger_time = args['trigger_time']
        
        if args['trigger_time'] is not None:
            end_time = trigger_time + post_trigger_duration
            start_time = end_time - duration
        elif args['start_time'] is not None:
            start_time = args['start_time']
            end_time = args['end_time']
        else:
            print("Error: Trigger time or start time not extracted properly.")
            exit()

        # define psd parameters in case they don't have the psd in the result file.
        psd_duration = 32*duration
        psd_start_time = start_time - psd_duration
        psd_end_time = start_time

        ifo_list = call_data_GWOSC(logger, args, start_time, end_time, psd_start_time, psd_end_time, psds_array=psds)
    
    waveform_name = args['waveform_approximant']
    event_snr = []

    max_like = np.argmax(samples['log_likelihood'])
    parameter = samples.iloc[max_like].to_dict()

    if np.iscomplexobj(parameter['mass_2']):
        for keys in parameter:
            parameter[keys] = float(np.real(parameter[keys]))

    if isinstance(args['time_marginalization'], str):
        args['time_marginalization'] = eval(args['time_marginalization'])
    
    if isinstance(args['distance_marginalization'], str):
        args['distance_marginalization'] = eval(args['distance_marginalization'])
    
    if isinstance(args['jitter_time'], str):
        args['distance_marginalization'] = eval(args['distance_marginalization'])

    for amplitude in amplitudes:
        print('amplitude = ',amplitude)
        priors2 = copy.copy(priors)
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

        target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            ifo_list,
            waveform_generator_mem,
            time_marginalization = args['time_marginalization'],
            distance_marginalization = args['distance_marginalization'],
            distance_marginalization_lookup_table = 'TD_path',
            jitter_time=args['jitter_time'],
            priors = priors2,
            reference_frame = args['reference_frame'],
            time_reference = args['time_reference'],
        )

        frequency_domain_strain = waveform_generator_mem.frequency_domain_strain(parameter)

        target_likelihood.parameters.update(parameter)

        optimal_snr_sq_list = []
        for ifo in ifo_list:
            snr_array = target_likelihood.calculate_snrs(frequency_domain_strain, ifo)
            optimal_snr_sq = snr_array.optimal_snr_squared
            optimal_snr_sq_list.append(optimal_snr_sq)
        optimal_snr = np.sqrt(sum(optimal_snr_sq_list))
        event_snr.append(optimal_snr)

    result = np.stack((np.array(amplitudes), np.array(event_snr)), axis=1)
    if outdir is not None:
        np.savetxt(f'{outdir}/{event_name}/{event_name}_memory_snr_vs_amp.csv', result)
    else:
        return result


###############################################
# event_name = 'GW170818'
# waveform = 'IMRPhenomXPHM'
# amplitudes = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# source_model = mem_freq_XPHM_only
# outdir = 'results/'

# file_path, data_file = extract_files(event_name)
# print(f"opening {file_path}")

# extension = os.path.splitext(file_path)[1].lstrip('.')
# if 'h5' in extension:
#     samples_dict, meta_dict, config_dict, priors_dict, psds = create_post_dict(file_path, waveform)
#     args = extract_relevant_info(meta_dict, config_dict)
# elif 'json' in extension:
#     result = bilby.core.result.read_in_result(file_path)
#     samples_dict = result.posterior
#     args = process_bilby_result(result.meta_data['command_line_args'])
#     priors_dict = result.priors
#     psds=None
# else:
#     print('Cannot recognise file type.')
#     exit()

# calculate_amp_vs_snr(event_name, amplitudes, samples_dict, priors_dict, data_file, args, source_model, outdir)