
import pandas as pd
import numpy as np
import bilby
import lal
import copy
import gwpy
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

from waveforms import osc_freq_XPHM, mem_freq_XPHM, mem_freq_XPHM_only
from create_post_dict import create_post_dict, extract_relevant_info


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



events = [('GW150914', '/home/shunyin.cheung/GWOSC_posteriors/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5', 1126259462.4, 4.0, "C01:IMRPhenomXPHM", None)]
event_name, file_path, trigger_time, duration, waveform, data_file = events[0]
amplitude = 100
print('amplitude = ',amplitude)

samples, meta_dict, config_dict, priors, psds, calibration = create_post_dict(file_path, waveform)
args = extract_relevant_info(meta_dict, config_dict)

logger = bilby.core.utils.logger

sampling_frequency = args['sampling_frequency']
maximum_frequency = args['maximum_frequency']
minimum_frequency = args['minimum_frequency']
reference_frequency = args['reference_frequency']
roll_off = args['tukey_roll_off']
duration = args['duration']
post_trigger_duration = args['post_trigger_duration']
trigger_time = args['trigger_time']
detectors = args['detectors']
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

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

# define oscillatory + memory model using gwmemory.
waveform_generator_full = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model= mem_freq_XPHM,
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
    waveform_generator_full,
    time_marginalization = True,
    distance_marginalization = True,
    distance_marginalization_lookup_table = "'TD.npz'.npz",
    jitter_time=True,
    priors = priors,
    reference_frame = args['reference_frame'],
    time_reference = args['time_reference'],
)

injection_parameters = dict(
    mass_1_source=35.3,
    mass_2_source=29.6,
    a_1=0.34,
    a_2=0.32,
    luminosity_distance=440,
    theta_jn=np.pi/2,
    psi=2.659,                   # guess
    phase=5,                   # guess
    geocent_time=1126259642.413,
    ra=3,                
    dec=-1.2108,
    tilt_1 = 0.79,
    tilt_2 = 2.36,
    time_jitter=0.001
)

true_parameters = dict()
# print(samples.keys())
# for key in samples.keys():
#     print(key)
#     true_parameters[key]= np.mean(samples[key])

#posterior = true_parameters
#posterior = injection_parameters
posterior = samples.iloc[10].to_dict()
#reference_dict = {'geocent_time': priors['geocent_time'],
#                'luminosity_distance': priors['luminosity_distance']}
#posterior.update(reference_dict)
frequency_domain_strain = waveform_generator_full.frequency_domain_strain(posterior)
time_domain_strain_osc = waveform_generator_osc.time_domain_strain(posterior)
time_domain_strain_full = waveform_generator_full.time_domain_strain(posterior)
time_domain_strain_mem = waveform_generator_mem.time_domain_strain(posterior)
time_array = waveform_generator_full.time_array
target_likelihood.parameters.update(posterior)
snr_array_H1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[0])
snr_array_L1 = target_likelihood.calculate_snrs(frequency_domain_strain, ifo_list[1])
opt_snr_H = snr_array_H1.optimal_snr_squared
opt_snr_L = snr_array_L1.optimal_snr_squared
cmf_snr_H = snr_array_H1.complex_matched_filter_snr
cmf_snr_L = snr_array_L1.complex_matched_filter_snr

print('network opt SNR: ', np.sqrt(opt_snr_H+opt_snr_L))
print('network complex matched_filter SNR', cmf_snr_H+cmf_snr_L)
reference_dict = {'geocent_time': priors['geocent_time'],
                'luminosity_distance': priors['luminosity_distance']}
posterior.update(reference_dict)
target_likelihood.parameters.update(posterior)
print('likelihood', target_likelihood.log_likelihood_ratio())

plt.figure()
plt.plot(time_array, time_domain_strain_full['plus']-1j*time_domain_strain_full['cross'], linestyle='dashed', label='full')
plt.plot(time_array, time_domain_strain_osc['plus']-1j*time_domain_strain_osc['cross'], label='osc only')
plt.plot(time_array, time_domain_strain_mem['plus']-1j*time_domain_strain_mem['cross'], label='mem only')
plt.xlim(start_time,start_time+0.1)
plt.legend()
plt.savefig('tests/td_test_zoomed.png')



