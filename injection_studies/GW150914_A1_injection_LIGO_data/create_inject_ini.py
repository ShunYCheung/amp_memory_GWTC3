import numpy as np
import sys
sys.path.append("..")


template = """
trigger-time = {trigger_time}
outdir = GW150914_A1_real_noise_run{injection_number}
detectors = [H1, L1]
duration = 4
data-dict = {{H1:data/GW150914_mem_A1_H1_run{injection_number}.txt, L1:data/GW150914_mem_A1_L1_run{injection_number}.txt}}
psd-dict = {{H1:data/H1_psd_run{injection_number}.dat, L1:data/L1_psd_run{injection_number}.dat}}
channel-dict = {{H1:DCS-CALIB_STRAIN_C02, L1:DCS-CALIB_STRAIN_C02}}

accounting = ligo.dev.o4.cbc.waveforms.bilby

coherence-test = False

sampling-frequency=2048
waveform-approximant = IMRPhenomXPHM

calibration-model=None

time-marginalization=False
distance-marginalization=False
phase-marginalization=False

default-prior=BBHPriorDict
deltaT=0.2
prior-file=None
prior-dict={{mass_1: Constraint(name='mass_1', minimum=10, maximum=80), mass_2 : Constraint(name='mass_2', minimum=10, maximum=80), mass_ratio :  Uniform(name='mass_ratio', minimum=0.125, maximum=1, latex_label="$q$"), chirp_mass :  Uniform(name='chirp_mass', minimum=25, maximum=40, latex_label="$M_{{c}}$"), a_1 : Uniform(name='a_1', minimum=0, maximum=0.99), a_2 : Uniform(name='a_2', minimum=0, maximum=0.99), tilt_1 : 2.3792185506368795, tilt_2 : 0.8663525482689185, phi_12 : 5.878104994643697, phi_jl : 0.25757481830607415, luminosity_distance : PowerLaw(alpha=2, name='luminosity_distance', minimum=50, maximum=2000, unit='Mpc', latex_label='$d_L$'), dec : -1.2259035208296045, ra : 2.216116171456804, theta_jn : Sine(name='theta_jn'), psi : 0.6912519348265866, phase : 4.000226339478637 , geocent_time: {trigger_time}}}
enforce-signal-duration=True

sampler = dynesty
sampler-kwargs = {{nlive: 2000, sample: rwalk, walks=100, n_check_point=2000, nact=10, resume=True}}

n-parallel = 5

request-cpus = 4
request-memory = 8
request-disk = 8

overwrite-outdir = True
"""

# event properties
duration = 4
detectors = ['H1', 'L1']


for i in range(20):
    filename = f'inject_GW150914_A1_run{i}.ini'
    trigger_time = np.loadtxt(f'data/trigger_time_run{i}.txt')
    
    args = dict(
        injection_number = i,
        trigger_time = trigger_time,
    )

    print(args)
    with open(filename, 'w') as ff:
        ff.write(template.format(**args))