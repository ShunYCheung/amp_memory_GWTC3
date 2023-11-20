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
psd-fractional-overlap=0.5
post-trigger-duration=2.0

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
n-parallel = 5

disable-hdf5-locking = True

waveform-approximant=IMRPhenomXPHM
likelihood-type=GravitationalWaveTransient
waveform-generator=bilby.gw.waveform_generator.WaveformGenerator

################################################################################
## Injection arguments
################################################################################

injection=True
injection-dict=None
injection-file={H1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/H1_GW170818_a1_frequency_domain_data.dat, L1:/home/shunyin.cheung/amp_memory_GWTC3/injection_studies/L1_GW170818_a1_frequency_domain_data.dat}
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