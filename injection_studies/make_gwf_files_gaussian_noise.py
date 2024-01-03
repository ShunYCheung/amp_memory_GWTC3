import numpy as np
import bilby
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/shunyin.cheung/amp_memory_GWTC3/")

import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

from event_table import call_event_table
from waveforms import mem_freq_XPHM
from create_post_dict import create_post_dict

event_number = 0
event_name, file_path, waveform, data_file = call_event_table()[event_number]
samples, meta_dict, config_dict, priors, psds, calibration = create_post_dict(file_path, waveform)

sampling_frequency = 2048
reference_frequency = 20
minimum_frequency = 20
maximum_frequency = 1024
roll_off=0.4

true_amplitude=1.0

max_like = np.argmax(samples['log_likelihood'])
max_like_parameters = samples.iloc[max_like].to_dict()
del max_like_parameters['iota']
max_like_parameters.update({'theta_jn': np.pi/2})       # maximise the memory signal


# event properties
trigger_time = 1126259462.4
end_time = trigger_time + 2
duration = 4
start_time = end_time - duration

detectors = ['H1', 'L1']

waveform_name = 'IMRPhenomXPHM'

waveform_generator_inject = bilby.gw.waveform_generator.WaveformGenerator(
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
                                waveform_approximant = waveform_name,
                                amplitude=true_amplitude))


true_parameters = dict(mass_1=max_like_parameters['mass_1'],
                            mass_2=max_like_parameters['mass_2'],
                            luminosity_distance=max_like_parameters['luminosity_distance'],
                            a_1=max_like_parameters['a_1'],
                            a_2=max_like_parameters['a_2'], 
                            tilt_1=max_like_parameters['tilt_1'], 
                            phi_12=max_like_parameters['phi_12'], 
                            tilt_2=max_like_parameters['tilt_2'], 
                            phi_jl=max_like_parameters['phi_jl'], 
                            theta_jn=max_like_parameters['theta_jn'], 
                            phase=max_like_parameters['phase'],
                            geocent_time=trigger_time,
                            psi=max_like_parameters['psi'],
                            ra=max_like_parameters['ra'],
                            dec=max_like_parameters['dec'])

for i in range(5, 10):
    interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for interferometer in interferometers:
        det = interferometer.name
        interferometer.minimum_frequency = 20
        interferometers.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
        )

        interferometers.inject_signal(
        parameters=true_parameters, waveform_generator=waveform_generator_inject,
        )

        timeseries = TimeSeries(
            data=interferometer.strain_data.time_domain_strain, times=interferometer.strain_data.time_array)

        psd_array = interferometer.power_spectral_density_array
        freq_array = interferometer.frequency_array

        print(psd_array)
        print(freq_array)

        psd_array = np.column_stack((freq_array, psd_array))
        np.savetxt(f'/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/GW150914_A1_injection_gaussian_noise/data/{det}_psd_run{i}.dat', psd_array)
        
        timeseries.write(f'/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/GW150914_A1_injection_gaussian_noise/data/GW150914_mem_A1_{det}_run{i}.txt')

        timeseries.plot()
        plt.savefig(f'GW150914_A1_injection_gaussian_noise/data/check_timeseries_run{i}.png')

        
    