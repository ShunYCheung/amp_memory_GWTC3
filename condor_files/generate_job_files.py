"""
Makes the .sh, .sub and .dag files to run reweighting.
"""

import os
import numpy as np

import sys
sys.path.append("..")

analysis_template = """universe = vanilla 
executable = {sh_filename}
log = {condor_root}{event}/{label}_{event}_amp{amp}.log
error = {condor_root}{event}/{label}_{event}_amp{amp}.err
output = {condor_root}{event}/{label}_{event}_amp{amp}.out

request_cpus = {cpus}
request_disk = {disk_usage}

should_transfer_files = Yes
when_to_transfer_output = ON_EXIT_OR_EVICT
getenv = True
environment = "HDF5_USE_FILE_LOCKING=FALSE"

accounting_group = ligo.dev.o4.cbc.waveforms.bilby
notification = error 
request_memory = {memory_usage}
queue 1
"""

def make_sh_files(args):
    bash_str = f"""#!/bin/bash
{args['command']} {args['cpus']} {args['event']} {args['amp']} {args['outdir']}
    """
    with open(args['sh_filename'], 'w') as ff:
        ff.write(bash_str)

        
def make_sub_files(args):
    with open(args['submit_filename'], 'w') as ff:
        ff.write(analysis_template.format(**args))

        
def make_dag_files(args):
    dag_str = (f"JOB 0 /home/shunyin.cheung/memory_GWTC3/condor_files/sub_files/dummy.sub\n")
    for i, amp in enumerate(args['amplitude']):
        for j, event in enumerate(args['event_list']):
            job_name = len(args['event_list'])*i+j+1
            dag_str += (
                #f"JOB {job_name} {args['sub_root']}{event}_amp{amp}.sub\n"
                f"JOB {job_name} {args['submit_filename_list'][job_name-1]}\n"
            )
    dag_str += ("PARENT 0 CHILD")
    for j in range(1, len(args['event_list'])*len(args['amplitude'])+1):
        dag_str += f" {j}"
    with open(args['dag_filename'], 'w') as ff:
        ff.write(dag_str)


def generate_GWTC3_job(args):
    submit_filename_list = []
    for amp in args['amplitude']:
        for event in args['event_list']:
            args['sh_filename'] = args['sh_root'] + event + "_amp" + str(amp) + ".sh"
            args['submit_filename'] = args['sub_root'] + event + "_amp" + str(amp) + ".sub"
            args['event'] = event
            args['amp'] = amp
            make_sh_files(args)
            make_sub_files(args)
            submit_filename_list.append(args['submit_filename'])

    args['submit_filename_list'] = submit_filename_list
    make_dag_files(args)


def generate_injection_job(args, events):

    labels = []
    event_number = []

    for amp in args['amplitude']:
        for j, label in enumerate(labels):
            args['sh_filename']= args['sh_root'] + label + "_amp" + str(amp) + ".sh"
            args['submit_filename'] = args['sub_root'] + label + "_amp" + str(amp) + ".sub"
            args['number'] = event_number[j]
            args['label'] = label

            make_sh_files(args)
            make_sub_files(args)
    
    make_dag_files(args)


root_dir = "/home/shunyin.cheung/amp_memory_GWTC3"

small_a = np.arange(0.1, 2, 0.1)
mid_a = np.arange(2, 8, 1)
large_a = np.arange(8, 64, 2)
e_large_a = np.arange(80, 420, 20)
# ee_large_a = np.arange(400, 1100, 100)

amplitude = np.concatenate((small_a, mid_a, large_a, e_large_a))

# small_a = np.arange(1, 10, 1)
# mid_a = np.arange(10, 100, 10)
# large_a = np.arange(100, 550, 50)

# amplitude = np.concatenate((small_a, mid_a, large_a,))

# full_list = list(range(50))

# unfinished_list = [9, 13, 16, 21, 33, 36, 44]

# set1 = set(full_list)
# set2 = set(unfinished_list)
# result_set = set1 - set2
# result_list = list(result_set)

args = dict(
    #command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_GWTC3.py",
    command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_GW170818_fmin30.py",
    #command = "python /home/shunyin.cheung/amp_memory_GWTC3/find_memory_burst.py",
    # command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_GWTC3_fmin.py",
    # command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_injection_simulated_noise.py",
    # command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_injection_simulated_noise_estimated_psd.py",
    # title
    label= 'fmin30',

    # events 
    # event_list = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190924', 'GW190925', 'GW190926', 
    #                'GW190929', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322'],
    
    #event_list = ['GW170608', 'GW190707', 'GW190720', 'GW190814', 'GW191113', 'GW191126', 'GW191216'],
    event_list = ['GW170818'],
    # amplitudes
    amplitude = amplitude,

    # resource allocation
    cpus = 4,
    memory_usage = '16G',
    disk_usage = '16G',

    # define paths
    root_dir = root_dir,
    sub_root = root_dir + '/condor_files/sub_files/GW170818_fmin30/',
    sh_root = root_dir + '/condor_files/sh_files/GW170818_fmin30/',
    condor_root = root_dir + '/output_condor/GW170818_fmin30/',
    dag_filename = root_dir + '/condor_files/reweight_fmin20_no_memory_to_fmin30_memory.submit',
    outdir = "/home/shunyin.cheung/amp_memory_GWTC3/test_run/GW170818_fmin30/",
)

generate_GWTC3_job(args)