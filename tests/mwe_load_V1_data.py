import pandas as pd
import numpy as np
import bilby
import lal
import gwpy
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import copy


logger = bilby.core.utils.logger

sampling_frequency = 2048
maximum_frequency = 896
minimum_frequency = 20
reference_frequency = 20
roll_off = 0.4
duration = 4
post_trigger_duration = 2.0
trigger_time = 1187058327.081509
detectors = ['H1', 'L1', 'V1']
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

ifo_list = bilby.gw.detector.InterferometerList([])
    

for det in detectors:   # for loop to add info about detector into ifo_list
    logger.info("Downloading analysis data for ifo {}".format(det))
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

    logger.info("Downloading psd data for ifo {}".format(det))                  # psd = power spectral density
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
                            waveform_approximant='IMRPhenomXPHM',
                            )

)

target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list,
    waveform_generator_osc,
    reference_frame ='H1L1V1',
    time_reference = 'geocent',
)

posteriors = {'chirp_mass': 32.998254736586915, 'mass_ratio': 0.8947786175325931, 'a_1': 0.3370565831153669, 'a_2': 0.40915043659777794, 
 'tilt_1': 1.762066847270146, 'tilt_2': 1.4514314256527152, 'phi_12': 0.6609705523734681, 'phi_jl': 4.383066285667695, 
 'theta_jn': 2.714602636914769, 'psi': 2.666386312764287, 'phase': 2.8722163664327867, 'azimuth': 0.003513333509039874, 'zenith': 2.270162185132061, 
 'time_jitter': 5.8894168313453326e-05, 'luminosity_distance': 1315.0218415920726, 'geocent_time': 1187058327.085401,
 'total_mass': 75.95070029964477, 'mass_1': 40.08420804249354, 'mass_2': 35.86649225715122, 'ra': 5.947967618186716, 'dec': 0.3500638112086276}


target_likelihood.parameters.update(posteriors)
likelihood = target_likelihood.log_likelihood_ratio()
print(likelihood)