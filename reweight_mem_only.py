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
import glob

from waveforms import *
from utils import *


def reweight_mem_only(event_name, samples, args, priors, out_folder, outfile_name_w, amplitude = 1.0, data_file=None, TD_path="TD.npz", psds = None, n_parallel=2):

    logger = bilby.core.utils.logger
    read_data=True
    _plot_fd_data_and_waveforms = False
    _check_template_fit = False
    _fit_vs_amplitude = True
    _calculate_likelihood = False
    _show_info = False

    # adds in detectors and the specs for the detectors. 
    sampling_frequency = args['sampling_frequency']
    maximum_frequency = args['maximum_frequency']
    minimum_frequency = args['minimum_frequency']
    reference_frequency = args['reference_frequency']
    roll_off = args['tukey_roll_off']
    duration = args['duration']
    trigger_time = args['trigger_time']
    
    # start_time = args['start_time']
    # end_time = args['end_time']


    # define psd parameters in case they don't have the psd in the result file.
    # psd_duration = 32*duration
    # psd_start_time = start_time - psd_duration
    # psd_end_time = start_time

    #args['detectors'] = ['L1']
    if read_data:
        ifo_list = bilby.gw.detector.InterferometerList([])
    
        # define interferometer objects
        for det in args['detectors']:   
            data_path = glob.glob(f'/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/{event_name}_*_{det}.txt')
            psd_path = glob.glob(f'/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/{event_name}_*_{det}_psd.dat')
            print(f'calling data from {data_path}')
            print(f'calling psd from {psd_path}')
            logger.info("Downloading analysis data for ifo {}".format(det))
            ifo = bilby.gw.detector.get_empty_interferometer(det)
            data = TimeSeries.read(data_path[0])
            ifo.strain_data.set_from_gwpy_timeseries(data)
            
            psd = np.genfromtxt(psd_path[0])
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd[:, 0], psd_array=psd[:, 1]
            )

            ifo_list.append(ifo)
            
    else:
        ifo_list = call_data_GWOSC(logger, args, start_time, end_time, psd_start_time, psd_end_time, psds_array=None)
    
    waveform_name = args['waveform_approximant']
    
    print('waveform used: ', waveform_name)

    # define oscillatory waveform.
    waveform_generator_osc = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model= no_CBC_signal,
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
                                waveform_approximant=waveform_name,
                                amplitude=amplitude)

    )

    if _fit_vs_amplitude:
        m = 24
        amplitudes = [1, 75, 150]
        
        posterior = samples.iloc[m].to_dict()
        pre_trigger_time = data_path[0].split(f'{event_name}_')
        trigger_time = float(pre_trigger_time[-1].split('_')[0]) + 2
        print(trigger_time)
        print(type(trigger_time))
        waveform_arguments = dict(duration=duration,
                                    roll_off=roll_off,
                                    minimum_frequency=minimum_frequency,
                                    maximum_frequency=maximum_frequency,
                                    sampling_frequency=sampling_frequency,
                                    reference_frequency=reference_frequency,
                                    waveform_approximant=waveform_name,
                                    amplitude=amplitude)

        fit_vs_amplitude(amplitudes, 
                        ifo_list[0], 
                        posterior, 
                        [mem_freq_XPHM_only], 
                        waveform_arguments, 
                        trigger_time, 
                        event_name,
                        ['memory_only'],
                        f'sample{m}',
                        f'/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/results/{event_name}/')
        exit()
    
    args['time_marginalization']=False
    args['distance_marginalization']=False
    args['jitter_time']=False
    
    priors2 = copy.copy(priors) # for some reason the priors change after putting it into the likelihood object. 
    # Hence, defining new ones for the second likelihood object.

    
    proposal_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_osc,
        time_marginalization = args['time_marginalization'],
        distance_marginalization = args['distance_marginalization'],
        #distance_marginalization_lookup_table = TD_path,
        jitter_time=args['jitter_time'],
        priors = priors,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )
    
    target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_mem,
        time_marginalization = args['time_marginalization'],
        distance_marginalization = args['distance_marginalization'],
        #distance_marginalization_lookup_table = TD_path,
        jitter_time=args['jitter_time'],
        priors = priors2,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )


    weights_list, weights_sq_list, proposal_ln_likelihood_list, target_ln_likelihood_list, ln_weights_list \
        = reweight_parallel(samples, proposal_likelihood, target_likelihood, priors2, 
                            args['time_marginalization'], args['distance_marginalization'], n_parallel)
        
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
            
            
        target_likelihood.parameters.update(posterior)
        target_likelihood.parameters.update(reference_dict)
        target_ln_likelihood_value = target_likelihood.log_likelihood_ratio()
        
        ln_weights = target_ln_likelihood_value-proposal_ln_likelihood_value
        
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
    p = mp.Pool(n_parallel)

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

        # Resampling timeseries to sampling_frequency using lal.
        lal_timeseries = data.to_lal()
        lal.ResampleREAL8TimeSeries(
            lal_timeseries, float(1/args['sampling_frequency'])
        )
        data = TimeSeries(
            lal_timeseries.data.data,
            epoch=lal_timeseries.epoch,
            dt=lal_timeseries.deltaT
        )

        # define some attributes in ifo
        ifo.strain_data.roll_off = args['tukey_roll_off']
        ifo.maximum_frequency = args['maximum_frequency']
        ifo.minimum_frequency = args['minimum_frequency']
        
        # set data as the strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)

        # # make my own psds
        # psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, sample_rate=16384)

        # psd_lal_timeseries = psd_data.to_lal()
        # lal.ResampleREAL8TimeSeries(
        #     lal_timeseries, float(1/args['sampling_frequency'])
        # )
        # psd_data = TimeSeries(
        #     psd_lal_timeseries.data.data,
        #     epoch=psd_lal_timeseries.epoch,
        #     dt=psd_lal_timeseries.deltaT
        # )

        # psd_alpha = 2 * args['tukey_roll_off'] / args['duration']                                      
        # psd = psd_data.psd(                                                       
        #     fftlength=args['duration'], overlap=0.5*args['duration'], window=("tukey", psd_alpha), method="median"
        # )

        # plt.figure()
        # plt.title(f"{det} psds")
        # plt.loglog(psds_array[det][:, 0], psds_array[det][:, 1], label='GWOSC psd')
        # plt.loglog(psd.frequencies.value, psd.value, label='my psd')
        # plt.xlim(10, 1024)
        # plt.ylim(1e-48, 1e-39)
        # plt.legend()
        # plt.savefig(f'tests/test_results/check_{det}_psd_stored_vs_self_generated_512s.png')   
   

        # compute the psd
        try:
            print("Using pre-computed psd from results file")
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psds_array[det][: ,0], psd_array=psds_array[det][:, 1]
            )
        except:
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


        ifo_list.append(ifo)

    return ifo_list


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
