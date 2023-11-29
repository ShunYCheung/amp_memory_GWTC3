import h5py
from bilby.core.utils.io import recursively_load_dict_contents_from_group, decode_from_hdf5, decode_bilby_json
import pandas as pd
from bilby.core.prior import Prior, PriorDict, DeltaFunction, ConditionalDeltaFunction
import json
import ast
import numpy as np
import os
import sys


def create_post_dict(file_name, waveform):
    
    # open data and convert the <closed hdf5 group> into readabel data types.
    with h5py.File(file_name, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')
    
    # access relevant info in the result file.
    posterior_samples = pd.DataFrame(data[waveform]['posterior_samples'])
    meta = data[waveform]['meta_data']['meta_data']
    
    if 'config' in data[waveform]['config_file'].keys():
        config = data[waveform]['config_file']['config']
    else:
        config = data[waveform]['config_file']
        print('No analytic priors found. Create time and distance priors to marginalize over.')
    psds = data[waveform]['psds']
    calibration = data[waveform]['calibration_envelope']
    
    if 'analytic' in data[waveform]['priors'].keys():
        priors = data[waveform]['priors']['analytic']
        # get rid of the annoying problem where all the prior entries are wrapped in a list.
        for key in list(priors.keys()):
            val = priors[key][0]
            priors[key] = val

        # complete some prior names so that bilby can recognise them and recover the appropriate function.
        val = data[waveform]['priors']['analytic']['chirp_mass']
        cl = val.split("(")
        if cl[0] == "UniformInComponentsChirpMass":
            complete_cl = "bilby.gw.prior.UniformInComponentsChirpMass("
            cl[0] = complete_cl
            string = ''.join(cl)
            data[waveform]['priors']['analytic']['chirp_mass']=string
            #print(string)

        val = data[waveform]['priors']['analytic']['mass_ratio']
        cl = val.split("(")
        if cl[0] == "UniformInComponentsMassRatio":
            complete_cl = "bilby.gw.prior.UniformInComponentsMassRatio("
            cl[0] = complete_cl
            string = ''.join(cl)
            data[waveform]['priors']['analytic']['mass_ratio']=string
            #print(string)
        val = data[waveform]['priors']['analytic']['mass_ratio']
        cl = val.split("(")
        if cl[0] == "UniformInComponentsMassRatio":
            complete_cl = "bilby.gw.prior.UniformInComponentsMassRatio("
            cl[0] = complete_cl
            string = ''.join(cl)
            data[waveform]['priors']['analytic']['mass_ratio']=string
            #print(string)
        # use bilby to convert the dict of prior names into PriorDict.
        priors = PriorDict(data[waveform]["priors"]['analytic'])
    else:
        print('No analytic priors found. Create time and distance priors to marginalize over.')
        priors = dict()
        priors['luminosity_distance'] = "PowerLaw(alpha=2, minimum=10, maximum=10000, name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None)"
        priors['geocent_time'] = "Uniform(minimum={0}, maximum={1}, name='geocent_time', latex_label='$t_c$', unit='$s$', boundary=None)".format(end_time-2-0.1, end_time-2+0.1)
        priors = PriorDict(priors)

    return posterior_samples, meta, config, priors, psds, calibration


def extract_relevant_info(meta, config):
    """
    I need a function to extract info as everybody stores the info in their result files differently.
    """
    # extract sampling frequency
    if 'sampling_frequency' in meta.keys():
        sampling_frequency = meta['sampling_frequency'][0]
    elif 'srate' in meta.keys():
        sampling_frequency = meta['srate'][0]
    elif 'sample_rate' in meta.keys():
        sampling_frequency = meta['sample_rate'][0]
    elif 'sampling-frequency' in config.keys():
        sampling_frequency = float(config['sampling-frequency'][0])
    else:
        print("unable to extract the sampling_frequency")
    
    # extract duration
    if 'duration' in meta.keys():
        duration = meta['duration'][0]
    elif 'segment_length' in meta.keys():
        duration = meta['segment_length'][0]
    elif 'duration' in config.keys():
        duration = float(config['duration'][0])
    else:
        print("unable to extract duration")
        
    # extract minimum frequency
    if 'f_low' in meta.keys():
        minimum_frequency = meta['f_low'][0]
    else:
        print("unable to extract minimum frequency")
    
    # extract maximum frequency
    if 'maximum-frequency' in config.keys():
        try:
            maximum_frequency = float(config['maximum-frequency'][0])
        except:
            str_dict = config['maximum-frequency'][0]
            max_freq_dict = ast.literal_eval(str_dict)
            key = list(max_freq_dict.keys())[0]
            maximum_frequency = max_freq_dict[key]
    else:
        print("unable to extract maximum frequency")

    # extract reference frequency
    if 'reference-frequency' in config.keys():
        reference_frequency = float(config['reference-frequency'][0])
    elif 'f_ref' in meta.keys():
        reference_frequency = meta['f_ref'][0]
    else:
        print('unable to extract reference frequency')
    
    # extract waveform name 
    waveform_approximant = meta['approximant'][0]
    
    # extract triggers
    if 'trigger-time' in config:
        trigger_time = float(config['trigger-time'][0])
        segment_start = None
        post_trigger_duration = float(config['post-trigger-duration'][0])
        end_time = None
    elif 'segment_start' in meta:
        trigger_time = None
        segment_start = meta['segment_start']
        post_trigger_duration = None
        end_time = segment_start + duration
    else:
        print('unable to extract trigger_time/start_time')
    
    # extract detectors
    detectors = ast.literal_eval(config['detectors'][0])
    
    # extract roll off
    if 'tukey-roll-off' in config:
        tukey_roll_off = float(config['tukey-roll-off'][0])
    else:
        print('Unable to extract tukey roll off settigns. Use default roll off of 0.4.')
        tukey_roll_off = 0.4
    
    # extract marginalisation settings
    if 'distance-marginalization' in config:
        distance_marginalization = config['distance-marginalization'][0]
    else:
        print('Cannot find time marginalization settings. Use general distance marginalization settings')
        distance_marginalization = True
    
    if 'time-marginalization' in config:
        time_marginalization = config['time-marginalization'][0]
    else:
        print('Cannot find distance marginalization settings. Use general time marginalization settings')
        time_marginalization = True
    
    # extract reference
    if 'reference-frame' in config:
        reference_frame = config['reference-frame'][0]
    else:
        print("Reference frame setting cannot be found. Use default setting.")
        ifos = config['analysis']['ifos'][0]
        res = ifos.replace("'", "").strip('][').split(', ')
        reference_frame = ''
        for string in res:
            reference_frame+=string
            
    if 'time-reference' in config:
        time_reference = config['time-reference'][0]
    else:
        print("Time reference setting cannot be found. Use default setting.")
        time_reference = 'geocent'
    
    if 'jitter_time' in config:
        jitter_time = config['jitter-time'][0]
    else:
        print("Jitter time setting cannot be found. Use default setting.")
        jitter_time = True
    
    # extract channel-dict
    if 'channel-dict' in config:
        string = config['channel-dict'][0]

        res = []

        if string[0:2] == '{ ':
            string = string[2:]
        if string[-1] == '}':
            string = string[:-1]
        for sub in string.split(','):
            for i in range(len(sub)):
                if sub[i] == ' ':
                    sub[i] == ''
            if ':' in sub:
                    res.append(map(str.strip, sub.split(':', 1)))
        channel_dict = dict(res)
    else:
        print('Unable to extract channel dict.')

    # extract path to distance lookup table
    outdir= config['outdir'][0]
    TD_name = config['distance-marginalization-lookup-table'][0]

    outdir_split = outdir.split('/')
    path_TD = ''
    for i in range(len(outdir_split)-1):
        path_TD += outdir_split[i]
        path_TD += '/'

    if os.path.isfile(path_TD + TD_name):
        lookup_table_path = path_TD + TD_name
    elif os.path.isfile(path_TD + "'TD.npz'.npz"):
        lookup_table_path = path_TD + "'TD.npz'.npz"
    else:
        print('Error: Unable to find distance marginalization lookup table')
        lookup_table_path = None
    
    # extract calibration model
    calibration_model = config['calibration-model'][0]
    spline_calibration_nodes = int(config['spline-calibration-nodes'][0])
    
    # combine all into a dict
    args = dict(duration=duration,
               sampling_frequency=sampling_frequency,
               maximum_frequency=maximum_frequency,
               minimum_frequency=minimum_frequency,
               reference_frequency=reference_frequency,
               waveform_approximant=waveform_approximant,
               trigger_time = trigger_time,
               detectors=detectors,
               start_time = segment_start,
               end_time=end_time,
               post_trigger_duration=post_trigger_duration,
               tukey_roll_off = tukey_roll_off,
               distance_marginalization=distance_marginalization,
               time_marginalization=time_marginalization,
               reference_frame=reference_frame,
               time_reference=time_reference,
               jitter_time=jitter_time,
               channel_dict=channel_dict,
               distance_marginalization_lookup_table = lookup_table_path,
               calibration_model = calibration_model,
               spline_calibration_nodes = spline_calibration_nodes)
    
    return args


def process_bilby_result(meta):
    
    if '{' in meta['minimum_frequency']:
        min_dict= ast.literal_eval(meta['minimum_frequency'])
        minimum_frequency = np.min(
                    [xx for xx in min_dict.values()]
                ).item()

        meta['minimum_frequency'] = minimum_frequency
    
    if '{' in meta['maximum_frequency']:
        max_dict = ast.literal_eval(meta['maximum_frequency'])
        maximum_frequency = np.max(
                    [xx for xx in max_dict.values()]
                ).item()

        meta['maximum_frequency'] = maximum_frequency
    
    meta['minimum_frequency'] = float(meta['minimum_frequency'])
    meta['maximum_frequency'] = float(meta['maximum_frequency'])
    return meta