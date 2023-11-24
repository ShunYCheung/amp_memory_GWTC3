"""
Makes the .sh, .sub and .dag files to run reweighting.
"""

import os
import numpy as np

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
{command} {cpus} {number} {amp}
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
command = "python /home/shunyin.cheung/amp_memory_GWTC3/reweight_injection.py"

amplitude = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 4, 8, 16, 32, 64, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


for amp in amplitude:
    for i in range (1, 101):
        args = dict(log_dir = "/home/shunyin.cheung/amp_memory_GWTC3",
                label = f'injection_run{j}',
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

    