import sys
sys.path.append("/home/shunyin.cheung/amp_memory_GWTC3/")

from create_post_dict import create_post_dict, extract_relevant_info
from pathfinder import extract_files
from injection_studies.data_generation import find_data, generate_data

file_path, data_file = extract_files('GW170818')
print(f"opening {file_path}")

waveform='IMRPhenomXPHM'

samples_dict, meta_dict, config_dict, priors_dict, psds = create_post_dict(file_path, waveform)
args = extract_relevant_info(meta_dict, config_dict)

# Using O2 data segment.
start_segment = 1164556817
end_segment = 1187733618
args['detectors'] = ['H1', 'L1']

for i in range(11, 100):
    args['start_time'], args['end_time'], args['psd_start_time'], args['psd_end_time']\
        =find_data(start_segment, end_segment, args['detectors'],args['duration'], 128)
    print('start time = ', args['start_time'])
    print('end time = ', args['end_time'])

    outdir='/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/'

    print(args['trigger_time'])

    label=f'run{i}_'+str(args['start_time'])

    print(args['start_time'])

    generate_data(args, outdir, label)

# outdir='/home/shunyin.cheung/amp_memory_GWTC3/memory_only_run/data/'

# print(args['trigger_time'])
# args['end_time'] = args['trigger_time']+ 2
# args['start_time'] = args['end_time'] - args['duration']
# args['psd_start_time'] = args['start_time'] - 128
# args['psd_end_time'] = args['start_time']

# label=str(args['start_time'])
# print(args['start_time'])

# generate_data(args, outdir, label)