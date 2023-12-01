from gwosc import datasets
import glob
import numpy as np
import h5py

def call_event_table(events = None):

    event_label = ['GW150914', 'GW151012', 'GW151226', 'GW170104', 'GW170608', 'GW170729', 'GW170809', 'GW170814', 'GW170818', 'GW170823', 'GW190403', 'GW190408', 'GW190412', 'GW190413_134308', 'GW190413_052954', 'GW190421', 'GW190426_190642', 'GW190503', 'GW190512', 'GW190513', 'GW190514', 'GW190517', 'GW190519', 'GW190521', 'GW190521_074359', 'GW190527', 'GW190602', 'GW190620', 'GW190630', 'GW190701', 'GW190706', 'GW190707', 'GW190708', 'GW190719', 'GW190720', 'GW190725', 'GW190727', 'GW190728', 'GW190731', 'GW190803', 'GW190805', 'GW190814', 'GW190828_065509', 'GW190828_063405', 'GW190910', 'GW190915', 'GW190916', 'GW190917', 'GW190924', 'GW190925', 'GW190926', 
                   'GW190929', 'GW190930', 'GW191103', 'GW191105', 'GW191109', 'GW191113', 'GW191126', 'GW191127', 'GW191129', 'GW191204_171526', 'GW191204_110529', 'GW191215', 'GW191216', 'GW191219', 'GW191222', 'GW191230', 'GW200105', 'GW200112', 'GW200115', 'GW200128', 'GW200129', 'GW200202', 'GW200208_222617', 'GW200208_130117', 'GW200209', 'GW200210', 'GW200216', 'GW200219', 'GW200220_124850', 'GW200220_061928', 'GW200224', 'GW200225', 'GW200302', 'GW200306', 'GW200308', 'GW200311_115853', 'GW200316', 'GW200322']

    path_list = glob.glob("/home/shunyin.cheung/GWOSC_PE_runs/posteriors/*.h5")
    s_path_list = sorted(path_list)
    
    waveform_list = []

    # if events is not None:
    #     print('1')
    #     true_path_list = []
    #     for event_num, event in enumerate(event_label):
    #         if event== events:
    #             print(event)
    #             true_path_list.append(s_path_list[event_num])
    # else:
    #     true_path_list = s_path_list

    # The following for loop accesses the durations in the public data. 
    # Because the categories are non-standard between differen events, 
    #it uses a lot of if statements to find the right statements.
    for i, file_name in enumerate(s_path_list):
        #print(i)
        wf_name1 = 'C01:IMRPhenomXPHM'
        wf_name2 = 'C01:IMRPhenomXPHM:LowSpin'
        wf_name3 = 'C01:IMRPhenomXPHM:HighSpin'
        with h5py.File(file_name,'r+') as f1:
            if wf_name1 in f1.keys():
                if 'config' in list(f1['C01:IMRPhenomXPHM']['config_file'].keys()):
                    duration = float(list(f1['C01:IMRPhenomXPHM']['config_file']['config']['duration'])[0])
                elif 'input' in list(f1['C01:IMRPhenomXPHM']['config_file'].keys()):
                    duration = float(list(f1['C01:IMRPhenomXPHM']['config_file']['input']['padding'])[0])
                else: 
                    print('category does not exist')
                max_key = wf_name1
            else:
                evidence_dict = {}
                if wf_name2 in list(f1.keys()):
                    ln_evidence = float(f1[wf_name2]['meta_data']['sampler']['ln_evidence'][0])
                    evidence_dict[wf_name2] = ln_evidence
                    max_key = wf_name2
                if wf_name3 in list(f1.keys()):
                    ln_evidence = float(f1[wf_name3]['meta_data']['sampler']['ln_evidence'][0])
                    evidence_dict[wf_name3] = ln_evidence
                    max_key = wf_name3

                for key in evidence_dict:
                    # If the value of the current key is greater than the value of max_key, update max_key
                    if evidence_dict[key] > evidence_dict[max_key]:
                        max_key = key
        waveform_list.append(max_key)
            

    alt_PE_runs = ['GW170608', 'GW190707', 'GW190720', 'GW190728', 'GW190814', 'GW190924']
    
    data_list = [None]*len(event_label)
    
    for i, event in enumerate(event_label):
        if event in alt_PE_runs:
            s_path_list[i] = f"/home/shunyin.cheung/other_PE_run/{event}.json"
            data_list[i] = f"/home/shunyin.cheung/other_PE_run/data/{event}_data_dump.pickle"
    
    event_table = list(zip(event_label, s_path_list, waveform_list, data_list))
    return event_table
#print(call_event_table())