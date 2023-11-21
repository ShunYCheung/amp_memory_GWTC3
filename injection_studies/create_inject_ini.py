import numpy as np
from gwpy import segments
import gwpy

import sys
sys.path.append("..")

from utils import check_data_quality

template = """
outdir = injection{injection_number}
label=GW170818_a1_{injection_number}
overwrite-outdir=False
detectors = [H1, L1]
duration = 4
sampling-frequency=2048
minimum-frequency=20
maximum-frequency=1024
reference-frequency = 20
tukey-roll-off=0.4
resampling-method=lal

coherence-test = False

deltaT = 0.2
time-marginalization=True
distance-marginalization=True
phase-marginalization=False
distance-marginalization-lookup-table = TD.npz
jitter-time=True
time-reference = geocent
reference-frame = H1L1

prior-file=injection1.prior

sampler = dynesty
nact = 5
nlive = 2000
dynesty-sample = rwalk
n-parallel = 1

disable-hdf5-locking = True

waveform-approximant=IMRPhenomXPHM
likelihood-type=GravitationalWaveTransient
waveform-generator=bilby.gw.waveform_generator.WaveformGenerator

################################################################################
## Injection arguments
################################################################################

injection=True
injection-dict=None
injection-file={{H1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/H1_GW170818_a1_run{injection_number}_frequency_domain_data.dat, L1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/L1_GW170818_a1_run{injection_number}_frequency_domain_data.dat}}
injection-numbers=None
injection-waveform-approximant=None
injection-waveform-arguments=None

################################################################################
## Slurm Settings
################################################################################

nodes = 1
ntasks-per-node = 16
time = 48:00:00
n-check-point = 10000
mem-per-cpu = 800

"""

print("""{H1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/H1_GW170818_a1_frequency_domain_data.dat, L1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/L1_GW170818_a1_frequency_domain_data.dat}""")

def check_data_quality(start, end, det):
    channel_num = 1
    quality_flag = (
        f"{det}:ITF_SCIENCE:{channel_num}"
        if det == "V1"
        else f"{det}:DMT-SCIENCE:{channel_num}"
    )
    try:
        flag = segments.DataQualityFlag.query(
            quality_flag, gwpy.time.to_gps(start), gwpy.time.to_gps(end)
        )

        # compare active duration from quality flag and total duration
        total_duration = end - start
        active_duration = float(flag.livetime)
        inactive_duration = total_duration - active_duration

        # data is not good if there is any period when the IFO is inactive
        if inactive_duration > 0:
            data_is_good = False
            print("Data quality check: FAILED. \n"
                "{det} does not have quality data for "
                "{inactive_duration}s out of {total_duration}s".format(
                    det=det,
                    inactive_duration=inactive_duration,
                    total_duration=total_duration,
                ))
        else:
            data_is_good = True
            print("Data quality check: PASSED.")
    except Exception as e:
        print(f"Error in Data Quality Check: {e}.")
        data_is_good = False
    return data_is_good

# Using O2 data segment.
start_segment = 1164556817
end_segment = 1187733618

# event properties
duration = 4
detectors = ['H1', 'L1']


for i in range(1):
    number = i + 1
    filename = f'GW170818_injection_LIGO_data/inject_GW170818_mem_A1_run{number}.ini'
    bad_data = True
    while bad_data:
        trigger_time = np.random.randint(start_segment, end_segment)
        start_time = trigger_time
        end_time = trigger_time + duration
        print(f'check data quality segement [{start_time}, {end_time}]')
        for det in detectors:
            det_pass = check_data_quality(start_time, end_time, det)
            if det_pass:
                bad_data = False

    
    args = dict(
        injection_number = number,
        trigger_time = start_time,
        duration = duration,
    )

    print(args)
    with open(filename, 'w') as ff:
        ff.write(template.format(**args))