"""
Makes the .sh, .sub and .dag files to run reweighting.
"""

import os
import numpy as np

import sys
sys.path.append("..")

from event_table import call_event_table

analysis_template = """universe = vanilla 
executable = {log_dir}/condor_files/sh_files/{label}_amp{amp}.sh
log = {log_dir}/output_condor/{label}/run2_{label}_amp{amp}.log
error = {log_dir}/output_condor/{label}/run2_{label}_amp{amp}.err
output = {log_dir}/output_condor/{label}/run2_{label}_amp{amp}.out

request_cpus = {cpus}
request_disk = 8000

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
python /home/shunyin.cheung/amp_memory_GWTC3/reweight_multi.py {cpus} {number} {amp}
    """
    with open(sh_filename, 'w') as ff:
        ff.write(bash_str)

        
def make_sub_files(args, submit_filename):
    with open(submit_filename, 'w') as ff:
        ff.write(analysis_template.format(**args))

        
def make_dag_files(args, dag_filename):
    dag_str = (f"JOB 0 /home/shunyin.cheung/memory_GWTC3/condor_files/sub_files/dummy.sub\n")
    for i, amp in enumerate(amplitude):
        for j, label in enumerate(events_remaining):
            job_name = len(events_remaining)*i+j+1
            dag_str += (
                f"JOB {job_name} {os.path.join(condor_dir, f'{label}_amp{amp}.sub')}\n"
            )
    dag_str += ("PARENT 0 CHILD")
    for j in range(1, len(events_remaining)*len(amplitude)+1):
        dag_str += f" {j}"
    with open(dag_filename, 'w') as ff:
        ff.write(dag_str)
        

###################################

event_table = call_event_table()

events_wanted = np.array(['GW170104', 'GW170818', 'GW170729', 'GW190413_052954', 'GW190426_190462', 'GW190521', 
                          'GW190602', 'GW190720', 'GW191109', 'GW191127', 'GW191127', 'GW191204_171526', 
                          'GW200128', 'GW200129', 'GW200202', 'GW190728', 'GW190924'])


events_remaining = []
event_number = []

for event_num, event in enumerate(event_table):
    event_name, file_path, trigger_time, durations, waveform, data =event
    if event_name not in events_wanted:
        events_remaining.append(event_name)
        event_number.append(event_num)


print('events remaining', events_remaining)
print('no. of remaining events', len(events_remaining))

amplitude = [200, 300, 400, 500, 600, 700, 800, 900, 1000]


for amp in amplitude:
    for j, label in enumerate(events_remaining):
        args = dict(log_dir = "/home/shunyin.cheung/amp_memory_GWTC3",
                label = label,
                number = event_number[j],
                amp = amp,
                cpus = 4,
        ) 
        sh_filename = args['log_dir'] + "/condor_files/sh_files/" + args['label'] + "_amp" + str(args['amp']) + ".sh"
        submit_filename = args['log_dir'] + "/condor_files/sub_files/" + args['label'] + "_amp" + str(args['amp']) + ".sub"
        make_sub_files(args, submit_filename)
        make_sh_files(args, sh_filename)

condor_dir = "/home/shunyin.cheung/amp_memory_GWTC3/condor_files/sub_files"
dag_filename = args['log_dir'] + "/condor_files/extend_remaining_a1100.submit"
make_dag_files(args, dag_filename)

    