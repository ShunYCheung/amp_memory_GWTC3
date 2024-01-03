import numpy as np
import bilby
import lal
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/shunyin.cheung/amp_memory_GWTC3/")

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from event_table import call_event_table
from waveforms import mem_freq_XPHM, mem_freq_XPHM_only
from create_post_dict import create_post_dict

event_number = 8
event_name, file_path, trigger_time, duration, waveform, data_file = call_event_table()[event_number]
samples, meta_dict, config_dict, priors, psds, calibration = create_post_dict(file_path, waveform)

sampling_frequency = 2048
reference_frequency = 20
minimum_frequency = 20
maximum_frequency = 1024
roll_off=0.4

true_amplitude=1.0

max_like = np.argmax(samples['log_likelihood'])
max_like_parameters = samples.iloc[max_like].to_dict()

# event properties
duration = 4
psd_duration = 32*duration
detectors = ['H1', 'L1']

waveform_name = 'IMRPhenomXPHM'

waveform_generator_cbc = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= bilby.gw.source.lal_binary_black_hole,
        parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(duration=duration,
                                roll_off=roll_off,
                                minimum_frequency=minimum_frequency,
                                maximum_frequency=maximum_frequency,
                                sampling_frequency=sampling_frequency,
                                reference_frequency=reference_frequency,
                                waveform_approximant = waveform_name,))

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
                                waveform_approximant = waveform_name,
                                amplitude=true_amplitude))


trigger_time = 1.186897191000000000e9
end_time = trigger_time + 2
start_time = end_time - duration
psd_end_time = start_time
psd_start_time = start_time - psd_duration

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
                    zenith=max_like_parameters['azimuth'],
                    azimuth=max_like_parameters['azimuth'],
                    psi=max_like_parameters['psi'],
                    ra=max_like_parameters['ra'],
                    dec=max_like_parameters['dec'])

# Set up interferometers.
ifo_list = bilby.gw.detector.InterferometerList([])

logger = bilby.core.utils.logger

for det in detectors:   # for loop to add info about detector into ifo_list
    ifo = bilby.gw.detector.get_empty_interferometer(det)

    logger.info("Downloading analysis data for ifo {}".format(det))

    data = TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=4096)

    # Resampling using lal as that was what was done in bilby_pipe.
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
    ifo.strain_data.roll_off=roll_off
    ifo.sampling_frequency = sampling_frequency
    ifo.duration = duration
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))

    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=4096)

    # again, we resample the psd_data using lal.
    psd_lal_timeseries = psd_data.to_lal()
    lal.ResampleREAL8TimeSeries(
        psd_lal_timeseries, float(1/sampling_frequency)
    )
    psd_data = TimeSeries(
        psd_lal_timeseries.data.data,
        epoch=psd_lal_timeseries.epoch,
        dt=psd_lal_timeseries.deltaT
    )

    psd_alpha = 2 * roll_off / duration                                        
    psd = psd_data.psd(                                                       
        fftlength=duration, overlap=0.5*duration, window=("tukey", psd_alpha), method="median"
    )

    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )

    ifo_list.append(ifo)


strain_cbc = waveform_generator_cbc.frequency_domain_strain(true_parameters)
strain_mem = waveform_generator_mem.frequency_domain_strain(true_parameters)

H1_strain_cbc = ifo_list[0].get_detector_response(strain_cbc, true_parameters)
L1_strain_cbc = ifo_list[1].get_detector_response(strain_cbc, true_parameters)
H1_strain_mem = ifo_list[0].get_detector_response(strain_mem, true_parameters)
L1_strain_mem = ifo_list[1].get_detector_response(strain_mem, true_parameters)

plt.figure()
plt.loglog(ifo_list[0].frequency_array, ifo_list[0].amplitude_spectral_density_array, label='ASD data before injection')
plt.loglog(ifo_list[0].frequency_array, np.abs(H1_strain_cbc), label='CBC waveform')
plt.loglog(ifo_list[0].frequency_array, np.abs(H1_strain_mem), label='memory waveform')
plt.xlim(10, 1024)
plt.xlabel('frequency (Hz)')
plt.ylabel('strain')
plt.legend()
plt.savefig(f'H1_ASD_{trigger_time}.png')

plt.figure()
plt.loglog(ifo_list[1].frequency_array, ifo_list[1].amplitude_spectral_density_array, label='ASD data before injection')
plt.loglog(ifo_list[1].frequency_array, np.abs(L1_strain_cbc), label='CBC waveform')
plt.loglog(ifo_list[1].frequency_array, np.abs(L1_strain_mem), label='memory waveform')
plt.xlim(10, 1024)
plt.xlabel('frequency (Hz)')
plt.ylabel('strain')
plt.legend()
plt.savefig(f'L1_ASD_{trigger_time}.png')
    