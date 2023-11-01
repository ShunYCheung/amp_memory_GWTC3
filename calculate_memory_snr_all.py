import pandas as pd
import numpy as np
import bilby
import lal
import gwpy
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import copy
import pickle
import os

from waveforms import mem_freq_XPHM_only
from create_post_dict import create_post_dict, extract_relevant_info, process_bilby_result
from event_table import call_event_table

def call_data_GWOSC(logger, args, calibration, samples, detectors, start_time, end_time, psd_start_time, psd_end_time, duration, sampling_frequency, roll_off, minimum_frequency, maximum_frequency, psds_array=None, plot=False):
    
    ifo_list = bilby.gw.detector.InterferometerList([])
    
    # define interferometer objects
    for det in detectors:   
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        
        channel_type = args['channel_dict'][det]
        channel = f"{det}:{channel_type}"
        
        kwargs = dict(
            start=start_time,
            end=end_time,
            verbose=False,
            allow_tape=True,
        )

        type_kwargs = dict(
            dtype="float64",
            subok=True,
            copy=False,
        )
        data = gwpy.timeseries.TimeSeries.get(channel, **kwargs).astype(
                **type_kwargs)
        
        # Resampling timeseries to sampling_frequency using lal.
        lal_timeseries = data.to_lal()
        lal.ResampleREAL8TimeSeries(
            lal_timeseries, float(1/sampling_frequency)
        )
        data = TimeSeries(
            lal_timeseries.data.data,
            epoch=lal_timeseries.epoch,
            dt=lal_timeseries.deltaT
        )
    
        # define some attributes in ifo
        ifo.strain_data.roll_off = roll_off
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        
        # set data as the strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)
        
        # compute the psd
        if det in psds_array.keys():
            print("Using pre-computed psd from results file")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psds_array[det][: ,0], psd_array=psds_array[det][:, 1]
            )
        else:
            print('Error: PSD is missing!')
            exit()

        ifo_list.append(ifo)

    return ifo_list



event_list = call_event_table()

events_done = np.array(['GW150914', 'GW151012', 'GW151226', 'GW170104','GW170818','GW190412','GW190421','GW190521_074359', 'GW190630', 'GW190725', 'GW190814',  'GW190828_065509', 'GW190917', 'GW200129', 'GW200224', 'GW200302' , 'GW200316'])


events_remaining = []
for event in event_list:
    event_name, file_path, trigger_time, durations, waveform, data =event
    if event_name not in events_done:
        events_remaining.append(event)

# print('events remaining', events_remaining)
# print('no. of remaining events', len(events_remaining))

for ele in events_remaining[54:55]:
    event_name, file_path, trigger_time, durations, waveform, data_file =ele
    print('event: ', event_name)
    print('file open', file_path)
    waveform = "C01:IMRPhenomXPHM"
    
    extension = os.path.splitext(file_path)[1].lstrip('.')
    if 'h5' in extension:
        samples, meta_dict, config_dict, priors, psds, calibration = create_post_dict(file_path, waveform)
        args = extract_relevant_info(meta_dict, config_dict)
    elif 'json' in extension:
        result = bilby.core.result.read_in_result(file_path)
        samples = result.posterior
        args = process_bilby_result(result.meta_data['command_line_args'])
        priors = result.priors
        psds=None
        calibration=None
    else:
        print('Cannot recognise file type.')
        exit()


    logger = bilby.core.utils.logger
    print(data_file)

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
        post_trigger_duration = float(args['post_trigger_duration'])
        trigger_time = float(args['trigger_time'])
        
        print('trigger_time', type(trigger_time))
        print('post_trigger_duration', type(post_trigger_duration))
        
        detectors = args['detectors']
        if 'V1' in detectors:
            detectors.remove('V1')
        
        if args['trigger_time'] is not None:
            end_time = trigger_time + post_trigger_duration
            start_time = end_time - duration
        elif args['start_time'] is not None:
            start_time = args['start_time']
            end_time = args['end_time']
        else:
            print("Error: Trigger time or start time not extracted properly.")
            exit()

        psd_duration = 32*duration # deprecated
        psd_start_time = start_time - psd_duration # deprecated
        psd_end_time = start_time # deprecated

        ifo_list = call_data_GWOSC(logger, args, 
                                calibration, samples, detectors,
                                start_time, end_time, 
                                psd_start_time, psd_end_time, 
                                duration, sampling_frequency, 
                                roll_off, minimum_frequency, maximum_frequency,
                                psds_array=psds)
    
    waveform_name = args['waveform_approximant'] 
    mean_parameters = dict()
    for key in samples:
        if isinstance(samples[key][0], str):
            continue

        value = np.mean(samples[key])
        mean_parameters[key] = value
    posterior = mean_parameters
    

    event_snr = []

    amplitudes = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 100, 128]

    for amplitude in amplitudes:
        print('amplitude = ',amplitude)
        priors2 = copy.copy(priors)
        # test if bilby oscillatory waveform = gwmemory oscillatory waveform.
        waveform_generator_osc = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(duration=duration,
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    sampling_frequency=sampling_frequency,
                                    reference_frequency=reference_frequency,
                                    waveform_approximant=waveform_name,
                                    )

        )

        waveform_generator_mem = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model= mem_freq_XPHM_only,
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(duration=duration,
                                    roll_off=roll_off,
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    sampling_frequency=sampling_frequency,
                                    reference_frequency=reference_frequency,
                                    bilby_generator = waveform_generator_osc,
                                    amplitude=amplitude)

        )

        target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            ifo_list,
            waveform_generator_mem,
            time_marginalization = True,
            distance_marginalization = True,
            distance_marginalization_lookup_table = "TD.npz",
            jitter_time=True,
            priors = priors2,
            reference_frame = args['reference_frame'],
            time_reference = args['time_reference'],
        )

        frequency_domain_strain = waveform_generator_mem.frequency_domain_strain(posterior)
        target_likelihood.parameters.update(posterior)
        print(len(ifo_list))
        if len(ifo_list) == 2:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            snr_array_L1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[1])
            opt_snr_H = snr_array_H1.optimal_snr_squared
            opt_snr_L = snr_array_L1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_H+opt_snr_L)
        elif len(ifo_list) == 1:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            opt_snr_H = snr_array_H1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_H)
        elif len(ifo_list) == 3:
            snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
            snr_array_L1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[1])
            snr_array_V1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[2])
            opt_snr_H = snr_array_H1.optimal_snr_squared
            opt_snr_L = snr_array_L1.optimal_snr_squared
            opt_snr_V = snr_array_V1.optimal_snr_squared
            opt_snr = np.sqrt(opt_snr_H+opt_snr_L+opt_snr_V)
        else:
            print('¯\_(ツ)_/¯')
            exit()
        event_snr.append(opt_snr)

    result = np.stack((np.array(amplitudes), np.array(event_snr)), axis=1)
    np.savetxt('results/{0}/{1}_memory_snr_vs_amp.csv'.format(event_name, event_name), result)
