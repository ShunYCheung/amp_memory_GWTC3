
import numpy as np
import bilby
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

# number of times to calculate the log likelihood.
n_trials = 10

# define parameters
sampling_frequency = 4096
minimum_frequency=20
reference_frequency = 20
maximum_frequency = 2048
detectors = ['H1', 'L1']
roll_off = 0.4
duration = 4
psd_duration = 32*duration
post_trigger_duration = 2.0
waveform_name = 'IMRPhenomXPHM'

trigger_time = 1126259462.413       # Use GW150914 trigger time.
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration
psd_start_time = start_time-psd_duration
psd_end_time = start_time

calib_file_paths={"H1":"H1.dat", "L1":"L1.dat", }

logger = bilby.core.utils.logger

ifo_list = bilby.gw.detector.InterferometerList([])

# set up ifos
for det in detectors: 
    ifo = bilby.gw.detector.get_empty_interferometer(det)

    logger.info("Downloading analysis data for ifo {}".format(det))
    # call analysis data
    data = TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=4096)

    # define some attributes in ifo
    ifo.strain_data.roll_off=roll_off
    ifo.sampling_frequency = sampling_frequency
    ifo.duration = duration
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency

    ifo.strain_data.set_from_gwpy_timeseries(data)
    
    logger.info("Downloading psd data for ifo {}".format(det))
    
    # call psd data
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=4096)

    psd_alpha = 2 * roll_off / duration                                         
    psd = psd_data.psd(                                                       
        fftlength=duration, overlap=0.5*duration, window=("tukey", psd_alpha), method="median"
    )

    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )

    ifo_list.append(ifo)


# define waveform generator
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
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

results = []

# setting the liklihood object and calculating the log likelihood.
for i in range(n_trials):
    priors = bilby.core.prior.dict.PriorDict.from_json('GW150914_prior.json')

    # Loading in the calibration envelope model.
    for ifo in ifo_list:
        model = bilby.gw.calibration.Precomputed
        det = ifo.name
        
        ifo.calibration_model = model.from_envelope_file(
        calib_file_paths[det],
        frequency_array=ifo.frequency_array[ifo.frequency_mask],
        n_nodes=10,
        label=det,
        n_curves=1000,
        )

    # define likelihood object
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors = priors,
        reference_frame = 'sky',
        time_reference = 'geocent',
        calibration_marginalization=True,
    )

    # Use random parameters.
    parameters = dict(
        mass_1_source=35,
        mass_2_source=30,
        a_1=0.34,
        a_2=0.32,
        luminosity_distance=470,
        theta_jn=np.pi/2,
        psi=2.659,                   
        phase=5,                   
        geocent_time=1126259642.413,
        ra=3,                
        dec=-1.2108,
        tilt_1 = 0.79,
        tilt_2 = 2.36,
        time_jitter=0.001
    )

    # calculate log likelihood.
    likelihood.parameters.update(parameters)
    ln_likelihood_value = likelihood.log_likelihood_ratio()
    results.append(ln_likelihood_value)

    print("log likelihood", ln_likelihood_value)    

# plot distribution of log likelihoods.
bins = int(np.round(np.sqrt(n_trials)))
plt.figure()
plt.hist(results, bins=bins)
plt.savefig('test_results/ln_likelihood_distribution.png')