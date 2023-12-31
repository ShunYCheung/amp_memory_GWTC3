"""
Code for a function "reweight" to calculate the weights to turn a proposal posterior into a target posterior.
Also calculates the Bayes factor between no memory and memory hypothesis. 

To use this code, change the filepath to whichever data file with the posteriors 
you wish to reweight.

Author: Shun Yin Cheung

"""

import numpy as np
import bilby
import lal
import copy
import pickle
import gwpy
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.special import logsumexp
from scipy.signal.windows import tukey

from waveforms import mem_freq_XPHM

# profiling modules
import psutil
import os
from pympler import asizeof
from pympler import muppy
from pympler import summary


def reweight_mem_parallel(event_name, samples, args, priors, out_folder, outfile_name_w, amplitude = 1.0, data_file=None, TD_path="TD.npz", psds = None, n_parallel=2):


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
    
    print('waveform used: ', waveform_name)

    # define oscillatory waveform.
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
                                waveform_approximant=waveform_name,
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
    
    
    priors2 = copy.copy(priors) # for some reason the priors change after putting it into the likelihood object. 
    # Hence, defining new ones for the second likelihood object.
    
    proposal_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_osc,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = TD_path,
        jitter_time=jitter_time,
        priors = priors,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )
    
    target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_full,
        time_marginalization = time_marginalization,
        distance_marginalization = distance_marginalization,
        distance_marginalization_lookup_table = TD_path,
        jitter_time=jitter_time,
        priors = priors2,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )


    weights_list, weights_sq_list, proposal_ln_likelihood_list, target_ln_likelihood_list, ln_weights_list \
        = reweight_parallel(samples, proposal_likelihood, target_likelihood, priors2, 
                            time_marginalization, distance_marginalization, n_parallel)
        
    print('Reweighting results')

    # Calulate the effective number of samples.
    neff = (np.sum(weights_list))**2 /np.sum(weights_sq_list)
    efficiency = neff/len(weights_list)
    print("effective no. of samples = {}".format(neff))
    print("{} percent efficiency".format(efficiency*100))

    # Calculate the Bayes factor
    bf = 1/(len(ln_weights_list)) * np.exp(logsumexp(ln_weights_list))
    print("Bayes factor = {}".format(bf))
    
    lnbf = logsumexp(ln_weights_list) - np.log(len(ln_weights_list))
    print("Log Bayes factor = {}".format(lnbf))
    
    
    # save weights, proposal and target likelihoods into a .txt file
    np.savetxt(out_folder+"/{0}_a={1}_{2}.csv".format(outfile_name_w, amplitude, waveform_name), 
               weights_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_a={1}_{2}_proposal_likelihood.csv".format(outfile_name_w, amplitude, waveform_name), 
               proposal_ln_likelihood_list, 
               delimiter=",")
    np.savetxt(out_folder+"/{0}_a={1}_{2}_target_likelihood.csv".format(outfile_name_w, amplitude, waveform_name), 
               target_ln_likelihood_list, 
               delimiter=",")

    return weights_list, bf
    

    
def reweighting(data, proposal_likelihood, target_likelihood, priors, time_marginalization, distance_marginalization):
    logger = bilby.core.utils.logger
    ln_weights_list=[]
    weights_list = []
    weights_sq_list = []
    proposal_ln_likelihood_list = []
    target_ln_likelihood_list = []
    
    # if marginalization is turned on, define the reference values.
    reference_dict = {}
    if time_marginalization:
        reference_dict.update({'geocent_time': priors['geocent_time']}),
    if distance_marginalization:
        reference_dict.update({'luminosity_distance': priors['luminosity_distance']})

    
    length = data.shape[0]
    
    for i in range(length):

        posterior = data.iloc[i].to_dict()

        # make sure the values are float and not complex.
        if np.iscomplexobj(posterior['mass_2']):
            for keys in posterior:
                posterior[keys] = float(np.real(posterior[keys]))

        use_stored_likelihood=False
        
        if i % 1000 == 0:
            print("reweighted {0} samples out of {1}".format(i+1, length))
            logger.info("{:0.2f}".format(i / length * 100) + "%")
        
        if use_stored_likelihood:
            proposal_ln_likelihood_value = data['log_likelihood'].iloc[i]
            
        else:
            proposal_likelihood.parameters.update(posterior)
            proposal_likelihood.parameters.update(reference_dict)
            proposal_ln_likelihood_value = proposal_likelihood.log_likelihood_ratio()
            
            #print('difference between stored and calculated log likelihood')
            #print(proposal_ln_likelihood_value - data['log_likelihood'].iloc[i])
            
            
        target_likelihood.parameters.update(posterior)
        target_likelihood.parameters.update(reference_dict)
        target_ln_likelihood_value = target_likelihood.log_likelihood_ratio()
        
        ln_weights = target_ln_likelihood_value-proposal_ln_likelihood_value
        # print(ln_weights)
        # print('target log likelihood', target_ln_likelihood_value)
        
        weights = np.exp(target_ln_likelihood_value-proposal_ln_likelihood_value)
        weights_sq = np.square(weights)
        weights_list.append(weights)
        weights_sq_list.append(weights_sq)
        proposal_ln_likelihood_list.append(proposal_ln_likelihood_value)
        target_ln_likelihood_list.append(target_ln_likelihood_value)
        ln_weights_list.append(ln_weights)


    return weights_list, weights_sq_list, proposal_ln_likelihood_list, target_ln_likelihood_list, ln_weights_list



def reweight_parallel(samples, proposal_likelihood, target_likelihood, priors, 
                      time_marginalization, distance_marginalization, n_parallel=2):
    print("activate multiprocessing")
    p = mp.Pool(n_parallel, maxtasksperchild=200)

    data=samples
    new_data = copy.deepcopy(data)
  
    posteriors = np.array_split(new_data, n_parallel)

    new_results = []
    for i in range(n_parallel):
        res = copy.deepcopy(posteriors[i])
        new_results.append(res)
 
    iterable = [(new_result, proposal_likelihood, target_likelihood, priors, time_marginalization, distance_marginalization) for new_result in new_results]
    
    res = p.starmap(reweighting, iterable)
    
    p.close()
    weights_list_comb = np.concatenate([r[0] for r in res])
    weights_sq_list_comb = np.concatenate([r[1] for r in res])
    proposal_comb = np.concatenate([r[2] for r in res])
    target_comb = np.concatenate([r[3] for r in res])    
    ln_weights_comb = np.concatenate([r[4] for r in res])      
    return weights_list_comb, weights_sq_list_comb, proposal_comb, target_comb, ln_weights_comb



def call_data_GWOSC(logger, args, start_time, end_time, psd_start_time, psd_end_time, psds_array=None):
    
    ifo_list = bilby.gw.detector.InterferometerList([])
    
    # define interferometer objects
    for det in args['detectors']:   
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)

        data = gwpy.timeseries.TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=16384)

        alpha = 2 * args['tukey_roll_off'] / args['duration']     

        # Resampling timeseries to sampling_frequency using lal.
        lal_timeseries = data.to_lal()
        lal.ResampleREAL8TimeSeries(
            lal_timeseries, float(1/args['sampling_frequency'])
        )
        data = TimeSeries(
            lal_timeseries.data.data*tukey(len(lal_timeseries.data.data), alpha=alpha),
            epoch=lal_timeseries.epoch,
            dt=lal_timeseries.deltaT
        )

        # data = TimeSeries(
        #     lal_timeseries.data.data,
        #     epoch=lal_timeseries.epoch,
        #     dt=lal_timeseries.deltaT
        # )

        # define some attributes in ifo
        ifo.strain_data.roll_off = args['tukey_roll_off']
        ifo.maximum_frequency = args['maximum_frequency']
        ifo.minimum_frequency = args['minimum_frequency']
        
        # set data as the strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)

        # compute the psd
        if det in psds_array.keys():
            print("Using pre-computed psd from results file")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psds_array[det][: ,0], psd_array=psds_array[det][:, 1]
            )
        else:
            print('PSD is missing from results file. Generating one.')
            psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=16384)

            psd_lal_timeseries = psd_data.to_lal()
            lal.ResampleREAL8TimeSeries(
                lal_timeseries, float(1/args['sampling_frequency'])
            )
            psd_data = TimeSeries(
                psd_lal_timeseries.data.data,
                epoch=psd_lal_timeseries.epoch,
                dt=psd_lal_timeseries.deltaT
            )

            psd_alpha = 2 * args['tukey_roll_off'] / args['duration']                                      
            psd = psd_data.psd(                                                       
                fftlength=args['duration'], overlap=0.5*args['duration'], window=("tukey", psd_alpha), method="median"
            )

            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=psd.frequencies.value, psd_array=psd.value
            )

        if args['calibration_model'] == 'CubicSpline':
            ifo.calibration_model = bilby.gw.calibration.CubicSpline(f"recalib_{ifo.name}_",
                    minimum_frequency=ifo.minimum_frequency,
                    maximum_frequency=ifo.maximum_frequency,
                    n_points=args['spline_calibration_nodes'])

        ifo_list.append(ifo)

    return ifo_list



