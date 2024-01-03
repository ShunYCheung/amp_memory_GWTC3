"""
Makes the .sh, .sub and .dag files to run reweighting.
"""

import os
import numpy as np

analysis_template = """universe = vanilla 
executable = {sh_filename}
log = {condor_root}log_file/{label}_amp{amp}.log
error = {condor_root}err_file/{label}_amp{amp}.err
output = {condor_root}out_file/{label}_amp{amp}.out

request_cpus = {cpus}
request_disk = 32000

should_transfer_files = Yes
when_to_transfer_output = ON_EXIT_OR_EVICT
getenv = True
environment = "HDF5_USE_FILE_LOCKING=FALSE"

accounting_group = ligo.dev.o4.cbc.waveforms.bilby
notification = error 
request_memory = 8000 
queue 1
"""

def make_sh_files(args, sh_filename):
    number = args['number']
    cpus = args['cpus']
    bash_str = f"""#!/bin/bash
{command} {cpus} {number} {amp}
    """
    with open(sh_filename, 'w') as ff:
        ff.write(bash_str)
    

        
def make_sub_files(args, submit_filename):
    with open(submit_filename, 'w') as ff:
        ff.write(analysis_template.format(**args))

        
def make_dag_files(sub_root, dag_filename):
    dag_str = (f"JOB 0 /home/shunyin.cheung/memory_GWTC3/condor_files/sub_files/dummy.sub\n")
    for i, amp in enumerate(amplitude):
        for j, label in enumerate(events[4:]):
            job_name = len(events[4:])*i+j+1
            dag_str += (
                f"JOB {job_name} {sub_root}{label}_amp{amp}.sub\n"
            )
    dag_str += ("PARENT 0 CHILD")
    for j in range(1, len(events[4:])*len(amplitude)+1):
        dag_str += f" {j}"
    with open(dag_filename, 'w') as ff:
        ff.write(dag_str)
        

###################################
command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_injection_simulated_noise.py"
root_dir = "/home/shunyin.cheung/amp_memory_GWTC3"
sub_root = root_dir + '/condor_files/sub_files/injections_simulated_noise/'
sh_root = root_dir + '/condor_files/sh_files/injections_simulated_noise/'
condor_root = root_dir + '/output_condor/injections_simulated_noise/'

small_a = np.arange(0.1, 2, 0.1)
mid_a = np.arange(2, 8, 1)
large_a = np.arange(8, 64, 2)
e_large_a = np.arange(80, 420, 20)
ee_large_a = np.arange(400, 1100, 100)

amplitude = np.concatenate((small_a, mid_a, large_a, e_large_a, ee_large_a))

events = ['zero_noise', 'gaussian_noise_run1', 'gaussian_noise_run3' ,'gaussian_noise_run4', 'gaussian_noise_run5', 'gaussian_noise_run6', 'gaussian_noise_run8']


for amp in amplitude:
    for i, label in enumerate(events):
        args = dict(label = label,
                number = i,
                amp = amp,
                cpus = 4,
                condor_root = condor_root,
        ) 
        sh_filename = sh_root + args['label'] + "_amp" + str(args['amp']) + ".sh"
        submit_filename = sub_root + args['label'] + "_amp" + str(args['amp']) + ".sub"
        args['sh_filename'] = sh_filename
        
        make_sh_files(args, sh_filename)
        make_sub_files(args, submit_filename)

dag_filename = root_dir + "/condor_files/reweight_injections_simulated_noise_2.submit"
make_dag_files(sub_root, dag_filename)

    