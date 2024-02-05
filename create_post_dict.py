import h5py
from bilby.core.utils.io import recursively_load_dict_contents_from_group
import pandas as pd
from bilby.core.prior import PriorDict
import ast
import numpy as np


def create_post_dict(file_name, wf):
    # open data and convert the <closed hdf5 group> into readable data types.
    with h5py.File(file_name, "r") as ff:
        data = recursively_load_dict_contents_from_group(ff, '/')
    
    # access the right waveform in the result file.
    for key in data.keys():
        if wf in key:
            waveform = key
            continue

    posterior_samples = pd.DataFrame(data[waveform]['posterior_samples'])
    # extract posterior
    try:
        posterior_samples = pd.DataFrame(data[waveform]['posterior_samples'])
    except:
        raise Exception('Waveform not valid')
    
    # extract meta data
    meta = data[waveform]['meta_data']['meta_data']
    
    # extract config data
    try:
        config = data[waveform]['config_file']['config']
    except:
        raise Exception('Cannot recognise file struture of the result file.')
    
    # extract psds
    psds = data[waveform]['psds']
    
    # extract priors
    try:
        priors = data[waveform]['priors']['analytic']
        # get rid of the annoying problem where all the prior entries are wrapped in a list.
        for key in list(priors.keys()):
            val = priors[key][0]
            priors[key] = val

        # complete some prior names so that bilby can recognise them and recover the appropriate function.
        cl = priors['chirp_mass'].split("(")
        if cl[0] == "UniformInComponentsChirpMass":
            complete_cl = "bilby.gw.prior.UniformInComponentsChirpMass("
            cl[0] = complete_cl
            string = ''.join(cl)
            priors['chirp_mass']=string

        cl = priors['mass_ratio'].split("(")
        if cl[0] == "UniformInComponentsMassRatio":
            complete_cl = "bilby.gw.prior.UniformInComponentsMassRatio("
            cl[0] = complete_cl
            string = ''.join(cl)
            priors['mass_ratio']=string
       
        # use bilby to convert the dict of prior names into PriorDict.
        prior_dict = PriorDict(priors)
    except:
        raise Exception('No analytic priors found.')

    return posterior_samples, meta, config, prior_dict, psds


def extract_relevant_info(meta, config):
    """
    I need a function to extract info as everybody stores the info in their result files differently.
    """

    # extract sampling frequency
    if 'sampling-frequency' in config.keys():
        sampling_frequency = float(config['sampling-frequency'][0])
    else:
        print("unable to extract the sampling_frequency")
    
    # extract duration
    if 'duration' in config.keys():
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
               calibration_model = calibration_model,
               spline_calibration_nodes = spline_calibration_nodes)
    return args


def process_bilby_result(meta):
    if meta['minimum_frequency'] is not None:
        if '{' in meta['minimum_frequency']:
            min_dict= ast.literal_eval(meta['minimum_frequency'])
            minimum_frequency = np.min(
                        [xx for xx in min_dict.values()]
                    ).item()

            meta['minimum_frequency'] = minimum_frequency
        meta['minimum_frequency'] = float(meta['minimum_frequency'])
    
    if meta['maximum_frequency'] is not None:
        if '{' in meta['maximum_frequency']:
            max_dict = ast.literal_eval(meta['maximum_frequency'])
            maximum_frequency = np.max(
                        [xx for xx in max_dict.values()]
                    ).item()

            meta['maximum_frequency'] = maximum_frequency
        
        meta['maximum_frequency'] = float(meta['maximum_frequency'])
    else:
        meta['maximum_frequency'] = meta['sampling_frequency']/2
    return meta