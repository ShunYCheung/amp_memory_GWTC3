import numpy as np
import sys
sys.path.append("..")


template = """
outdir = injection{injection_number}
label=GW170818_a1
overwrite-outdir=False
detectors = [H1, L1]
duration = 4
sampling-frequency=2048
minimum-frequency=20
maximum-frequency=1024
reference-frequency = 20
tukey-roll-off=0.4

data-dict = {{H1:data/GW170818_mem_A1_H1_run{injection_number}.txt, L1:data/GW170818_mem_A1_L1_run{injection_number}.txt}}
psd-dict = {{H1:H1_psd_run{injection_number}.dat, L1:L1_psd_run{injection_number}.dat}}
channel-dict={{H1:DCH-CLEAN_STRAIN_C02, L1:DCH-CLEAN_STRAIN_C02}}

trigger-time={trigger_time}
post-trigger-duration = 2.0

coherence-test = False

deltaT = 0.2
time-marginalization=True
distance-marginalization=True
phase-marginalization=False
distance-marginalization-lookup-table = TD.npz
jitter-time=True
time-reference = geocent
reference-frame = H1L1

prior-file=priors/injection1.prior

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
## Slurm Settings
################################################################################

nodes = 1
ntasks-per-node = 16
time = 48:00:00
n-check-point = 10000
mem-per-cpu = 800
"""


# Using O2 data segment.
start_segment = 1164556817
end_segment = 1187733618

# event properties
duration = 4
detectors = ['H1', 'L1']


for i in range(1):
    number = i + 1
    filename = f'GW170818_injection_LIGO_data/inject_GW170818_mem_A1_test_run{number}.ini'
    trigger_time = np.loadtxt(f'GW170818_injection_LIGO_data/data/trigger_time_run{number}.txt')
    
    args = dict(
        injection_number = number,
        trigger_time = trigger_time[0],
    )

    print(args)
    with open(filename, 'w') as ff:
        ff.write(template.format(**args))