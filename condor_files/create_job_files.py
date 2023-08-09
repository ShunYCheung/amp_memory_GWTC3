"""
Makes the .sh, .sub and .dag files to run reweighting.
"""

import os

analysis_template = """universe = vanilla 
executable = {log_dir}/condor_files/sh_files/{label}.sh
log = {log_dir}/output_condor/run3_{label}.log
error = {log_dir}/output_condor/run3_{label}.err
output = {log_dir}/output_condor/run3_{label}.out

request_cpus = 4
request_disk = 5000

should_transfer_files = Yes
when_to_transfer_output = ON_EXIT_OR_EVICT
getenv = True
environment = "HDF5_USE_FILE_LOCKING=FALSE"

accounting_group = ligo.dev.o4.cbc.waveforms.bilby
notification = error 
request_memory = 5000 
queue 1
"""

def make_sh_files(args, sh_filename):
    number = args['job_number'] - 1
    bash_str = f"""#!/bin/bash
python /home/shunyin.cheung/memory_GWTC3/reweight_multi.py {number}
    """
    with open(sh_filename, 'w') as ff:
        ff.write(bash_str)

        
def make_sub_files(args, submit_filename):
    with open(submit_filename, 'w') as ff:
        ff.write(analysis_template.format(**args))

        
def make_dag_files(args, dag_filename):
    dag_str = (f"JOB 0 /home/shunyin.cheung/memory_GWTC3/condor_files/sub_files/dummy.sub\n")
    for i, label in enumerate(event_label2):
        job_name = i+1
        dag_str += (
            f"JOB {job_name} {os.path.join(condor_dir, f'{label}.sub')}\n"
        )
    dag_str += ("PARENT 0 CHILD")
    for j in range(1, len(event_label2)+1):
        dag_str += f" {j}"
    with open(dag_filename, 'w') as ff:
        ff.write(dag_str)
        

###################################


event_label = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']
"""
event_label2 = ['GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190924', 'GW190925', 'GW190926', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_130117', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200311_115853', 'GW200316', 'GW200322']
"""

for i, label in enumerate(event_label):
    args = dict(log_dir = "/home/shunyin.cheung/memory_GWTC3",
            label = label,
            job_number = i+1,
    ) 
    sh_filename = args['log_dir'] + "/condor_files/sh_files/" + args['label'] + ".sh"
    submit_filename = args['log_dir'] + "/condor_files/sub_files/" + args['label'] + ".sub"
    make_sub_files(args, submit_filename)
    #make_sh_files(args, sh_filename)

condor_dir = "/home/shunyin.cheung/memory_GWTC3/condor_files/sub_files"
"""
args = dict(log_dir = "/home/shunyin.cheung/memory_GWTC3",
            label = None,
            job_number = 0)
"""
#dag_filename = args['log_dir'] + "/condor_files/reweight2.dag"
#make_dag_files(args, dag_filename)

    