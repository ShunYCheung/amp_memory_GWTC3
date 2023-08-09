from reweight_mem_parallel import reweight_mem_parallel
import json
from create_post_dict import create_post_dict
import sys

#event_number = int(sys.argv[1])
amplitude = float(sys.argv[1])
print(amplitude)


if __name__ == '__main__':
    events = [('GW150914', 'GWOSC_posteriors/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5', 1126259462.4, ['L1', 'H1'], 4.0)]
    waveform = "IMRPhenomXPHM"
    
    for count, i in enumerate(events):
        event_name, file_path, trigger_time, detectors, duration = i
        samples_dict = create_post_dict("/home/shunyin.cheung/"+file_path)
        print("reweighting {}".format(event_name), "{0}/{1} events reweighted".format(count+1, len(events)))
        weights, bf = reweight_mem_parallel(event_name, samples_dict, 
                                trigger_time, "/home/shunyin.cheung/amp_memory_GWTC3/run1" ,"test_{0}_weights".format(event_name), waveform, 
                                20, detectors, duration, 4096, amplitude = amplitude,
                                n_parallel=1)
        

        
        
        
        
        
        