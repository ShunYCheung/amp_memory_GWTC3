import numpy as np
import bilby
import lal
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from event_table import call_event_table
from waveforms import mem_freq_XPHM_v2
from create_post_dict import create_post_dict
from utils import check_data_quality

event_number = 8
event_name, file_path, trigger_time, duration, waveform, data_file = call_event_table()[event_number]
samples, meta_dict, config_dict, priors, psds, calibration = create_post_dict(file_path, waveform)

sampling_frequency = 2048
reference_frequency = 20
minimum_frequency = 20
maximum_frequency = 1024
roll_off=0.4

true_amplitude=1

max_like = np.argmax(samples['log_likelihood'])
true_parameters = samples.iloc[max_like].to_dict()

# Using O2 data segment.
start_segment = 1164556817
end_segment = 1187733618

# event properties
duration = 4
psd_duration = 32
detectors = ['H1', 'L1']

for i in range(7, 100):

    bad_data = True
    while bad_data:
        trigger_time = np.random.randint(start_segment, end_segment)
        end_time = trigger_time + 2
        start_time = end_time - duration
        psd_end_time = start_time
        psd_start_time = start_time - psd_duration
        print(f'check data quality segement [{psd_start_time}, {end_time}]')
        count = 0
        for det in detectors:
            det_pass = check_data_quality(psd_start_time, end_time, det)
            if det_pass:
                count+=1
        if count == len(detectors):
            bad_data=False


    # Set up interferometers.
    ifo_list = bilby.gw.detector.InterferometerList([])

    for det in detectors:   # for loop to add info about detector into ifo_list
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        data = TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=16384)


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
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo.strain_data.set_from_gwpy_timeseries(data)

        psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=16384)

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

        psd_alpha = 2 * roll_off / duration                                         # psd_alpha might affect BF
        psd = psd_data.psd(                                                         # this function might affect BF
            fftlength=duration, overlap=0.5*duration, window=("tukey", psd_alpha), method="median"
        )

        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value
        )

        ifo_list.append(ifo)

    waveform_name = 'IMRPhenomXPHM'

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
    waveform_generator_full = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model= mem_freq_XPHM_v2,
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(duration=duration,
                                    roll_off=roll_off,
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    sampling_frequency=sampling_frequency,
                                    reference_frequency=reference_frequency,
                                    bilby_generator = waveform_generator_osc,
                                    amplitude=true_amplitude))


    ifo_list.inject_signal(
    parameters=true_parameters, waveform_generator=waveform_generator_full
    )

    ifo_list.save_data(outdir='/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/GW170818_injection_LIGO_data/data', label=f'GW170818_a{true_amplitude}_run{i+1}')

    # plt.figure()
    # plt.plot(ifo.time_array, ifo.time_domain_strain)
    # plt.savefig('plot_time_domain_data.png')
