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

from waveforms import mem_freq_XPHM, mem_freq_XPHM_only
from utils import *


def reweight_mem_parallel(event_name, samples, args, priors, out_folder, outfile_name_w, amplitude = 1.0, data_file=None, TD_path="TD.npz", psds = None, n_parallel=2):

    logger = bilby.core.utils.logger
    _plot_fd_data_and_waveforms = True
    _check_template_fit = False
    _fit_vs_amplitude = False
    _calculate_likelihood = False
    _show_info = False

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
        minimum_frequency = 20#args['minimum_frequency']
        print('minimum frequency = ', minimum_frequency)
        reference_frequency = args['reference_frequency']
        roll_off = args['tukey_roll_off']
        duration = args['duration']
    else:
        sampling_frequency = args['sampling_frequency']
        maximum_frequency = args['maximum_frequency']
        minimum_frequency = 20#args['minimum_frequency']
        print('minimum frequency = ', minimum_frequency)
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

        #args['detectors'] = ['L1']

        ifo_list = call_data_GWOSC(logger, args, start_time, end_time, psd_start_time, psd_end_time, psds_array=psds)
        # print(f'minimum frequency = {minimum_frequency}')
        # print(f'maximum frequency = {maximum_frequency}')
        # plt.figure()
        # plt.plot(ifo_list[0].frequency_array, ifo_list[0].frequency_mask)
        # plt.xlabel('frequency')
        # plt.savefig('tests/test_results/check_frequency_mask.png')
    
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

    if _check_template_fit:
        check_template_fit(amplitude, 
                            ifo_list[0], 
                            samples, 
                            waveform_generator_full, 
                            trigger_time, 
                            '/home/shunyin.cheung/amp_memory_GWTC3/tests/test_results/')

    if _fit_vs_amplitude:
        amplitudes = np.arange(0, 300, 100)

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
                        samples, 
                        mem_freq_XPHM, 
                        waveform_arguments, 
                        trigger_time, 
                        '/home/shunyin.cheung/amp_memory_GWTC3/tests/test_results/')
    
    mem_strain = waveform_generator_mem.time_domain_strain(samples.iloc[58].to_dict())
    osc_strain = waveform_generator_osc.time_domain_strain(samples.iloc[58].to_dict())
    mem_total = mem_strain['plus'] - 1j* mem_strain['cross']
    osc_total = osc_strain['plus'] - 1j*osc_strain['cross']
    plt.figure()
    #plt.plot(ifo_list[0].time_array, ifo_list[0].time_domain_strain, label='data')
    plt.plot(ifo_list[0].time_array, mem_total, label='mem')
    plt.plot(ifo_list[0].time_array, osc_total, label='osc')
    plt.legend()
    plt.savefig(f'tests/test_results/{event_name}_memory_vs_data_sample58.png')



    if isinstance(args['time_marginalization'], str):
        args['time_marginalization'] = eval(args['time_marginalization'])
    
    if isinstance(args['distance_marginalization'], str):
        args['distance_marginalization'] = eval(args['distance_marginalization'])
    
    if isinstance(args['jitter_time'], str):
        args['distance_marginalization'] = eval(args['distance_marginalization'])
    
    
    priors2 = copy.copy(priors) # for some reason the priors change after putting it into the likelihood object. 
    # Hence, defining new ones for the second likelihood object.

    
    proposal_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_osc,
        time_marginalization = args['time_marginalization'],
        distance_marginalization = args['distance_marginalization'],
        distance_marginalization_lookup_table = TD_path,
        jitter_time=args['jitter_time'],
        priors = priors,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )
    
    target_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator_full,
        time_marginalization = args['time_marginalization'],
        distance_marginalization = args['distance_marginalization'],
        distance_marginalization_lookup_table = TD_path,
        jitter_time=args['jitter_time'],
        priors = priors2,
        reference_frame = args['reference_frame'],
        time_reference = args['time_reference'],
    )

    if _calculate_likelihood:
        max_like = np.argmax(samples['log_likelihood'])
        target_likelihood.parameters.update(samples.iloc[max_like].to_dict())
        target_likelihood.parameters.update({'geocent_time': priors['geocent_time']}),
        target_likelihood.parameters.update({'luminosity_distance': priors['luminosity_distance']})
        log_likelihood=target_likelihood.log_likelihood_ratio()
        print(f'likelihood at A={amplitude}',log_likelihood)

    if _show_info:
        max_like = np.argmax(samples['log_likelihood'])
        network_optimal_snr_osc = calculate_network_snr(ifo_list, 
                                                    waveform_generator_osc, 
                                                    proposal_likelihood, 
                                                    samples.iloc[max_like].to_dict())
        network_optimal_snr_full = calculate_network_snr(ifo_list, 
                                                    waveform_generator_full, 
                                                    target_likelihood, 
                                                    samples.iloc[max_like].to_dict())

        best_fit_params = samples.iloc[max_like].to_dict()
        for key in best_fit_params.keys():
            print(f'{key} = ', best_fit_params[key])
        print('segment duration:', duration)
        print(f'Tukey roll off = {roll_off}')
        print(f'fmin = {minimum_frequency}')
        print('Detectors = ', args['detectors'])
        print(f'sampling frequency = {sampling_frequency}')
        print(f'reference frequency = {reference_frequency}')
        print(f'network optimal snr A=0 = {network_optimal_snr_osc}')
        print(f'network optimal snr A={amplitude} = {network_optimal_snr_full}')

    if _plot_fd_data_and_waveforms:
        best_fit_params = samples.iloc[58].to_dict()
        plt.figure()
        plot_fd_data_and_waveforms(ifo_list[0], waveform_generator_osc, waveform_generator_mem, waveform_generator_full, best_fit_params)
        plt.xlabel('frequency (Hz)')
        plt.xlim(19, 1024)
        plt.ylim(1e-28, 5e-21)
        plt.legend()
        plt.savefig(f'tests/test_results/{event_name}_{ifo_list[0].name}_fd_data_and_waveforms_A{amplitude}_sample58.png')
    

    samples = samples.astype('float64')


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
    # np.savetxt(out_folder+"/{0}_a={1}_{2}.csv".format(outfile_name_w, amplitude, waveform_name), 
    #            weights_list, 
    #            delimiter=",")
    # np.savetxt(out_folder+"/{0}_a={1}_{2}_proposal_likelihood.csv".format(outfile_name_w, amplitude, waveform_name), 
    #            proposal_ln_likelihood_list, 
    #            delimiter=",")
    # np.savetxt(out_folder+"/{0}_a={1}_{2}_target_likelihood.csv".format(outfile_name_w, amplitude, waveform_name), 
    #            target_ln_likelihood_list, 
    #            delimiter=",")

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

    for i in range(1):
        posterior = data.iloc[48].to_dict()
        for key in posterior.keys():
            print(f'{key} = {posterior[key]}')

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
        print('target ln likelihood', target_ln_likelihood_value)
        
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

        if args['calibration_model'] == 'CubicSpline':
            ifo.calibration_model = bilby.gw.calibration.CubicSpline(f"recalib_{ifo.name}_",
                    minimum_frequency=ifo.minimum_frequency,
                    maximum_frequency=ifo.maximum_frequency,
                    n_points=args['spline_calibration_nodes'])

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
