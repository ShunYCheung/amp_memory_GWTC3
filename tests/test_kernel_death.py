import lalsimulation as lalsim
import matplotlib.pyplot as plt
import numpy as np
from gwmemory.utils import combine_modes

solar_mass = 2e30
mpc = 3e22
m1 = 30*solar_mass
m2 = 30*solar_mass

test = np.arange(0, 100, 1)


params = dict(f_min = 20.0,
              f_ref = 20.0,
              phiRef=0.0,
              approximant = lalsim.GetApproximantFromString('IMRPhenomXPHM'),
              LALpars=None,
              m1=m1,
              m2=m2,
              S1x=0.0,
              S1y=0.0,
              S1z=0.0,
              S2x=0.0,
              S2y=0.0,
              S2z=0.0,
              distance= 400*mpc,
              inclination=0.0,
              deltaF = 1/4,
              f_max = 1024,
              )

waveform_modes = lalsim.SimInspiralChooseFDModes(**params)

h_lm = dict()
while waveform_modes is not None:
    mode = (waveform_modes.l, waveform_modes.m)
    data = waveform_modes.mode.data.data
    h_lm[mode] = np.roll(data, 4 * params['f_max']+1)
    #h_lm[mode] = data
    waveform_modes = waveform_modes.next

print('original wf array: ', h_lm)
print('length of original wf array: ', len(h_lm[(2,2)]))
frequency = np.linspace(0, 1024, 4097)
# print(frequency)
# wf = h_lm[(2,2)]
# plt.figure()
# plt.loglog(frequency,  np.abs(wf))
# plt.savefig('tests/fd_mode_wf_22modes.png')

times = np.linspace(0, 4, 8193)

# wf_td = np.fft.ifft(wf)*2048
# print('length of ifft array, ', len(wf_td))

# plt.figure()
# plt.plot(times, wf_td)
# plt.savefig('tests/fd_mode_wf_22modes_time_domain.png')

h_lm_td = {}
for l,m in h_lm:
    mode = (l, m)
    h_lm_td[mode] = np.fft.ifft(h_lm[mode])*2048
print(h_lm_td)
full_waveform = combine_modes(h_lm_td, inc=np.pi/2, phase=5)
print(full_waveform)
plt.figure()
plt.plot(times, full_waveform['plus']-1j*full_waveform['cross'])
plt.savefig('tests/fd_mode_full_wf_time_domain.png')

full_waveform_fd = {}
for key in full_waveform:
    full_waveform_fd[key]= np.fft.rfft(full_waveform[key])/2048

print('length of fft array, ', len(full_waveform_fd['plus']))

plt.figure()
plt.plot(frequency, np.abs(full_waveform_fd['plus']+full_waveform_fd['cross']))
plt.savefig('tests/fd_mode_full_wf_freq_domain.png')