import gwmemory
import bilby
import copy

from scipy.signal import get_window
from scipy.signal.windows import tukey
import utils

import matplotlib.pyplot as plt
import numpy as np

modes = [(2,2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]

def osc_time_XPHM(times, mass_ratio, total_mass, luminosity_distance, iota, phase, 
                  **kwargs):
        trigger_time = kwargs.get('trigger_time')
        minimum_frequency = kwargs.get('minimum_frequency')
        sampling_frequency = kwargs.get('sampling_frequency')

        times2 = times - trigger_time
        
        surr = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', minimum_frequency=minimum_frequency,
                                              sampling_frequency=sampling_frequency, 
                                              distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[0, 0, 0], spin_2=[0, 0, 0], 
                                                      times=times)

        h_lm, surr_times = surr.time_domain_oscillatory()
        oscillatory = gwmemory.utils.combine_modes(h_lm,iota, phase)
        window = tukey(surr_times.size, 0.1)

        plus_new = oscillatory["plus"]
        cross_new = oscillatory['cross']
        plus = plus_new * window
        cross = cross_new * window

        return {"plus": plus, "cross": cross}


def mem_time_XPHM(times, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, iota, phase, 
                  **kwargs):
        trigger_time = kwargs.get('trigger_time')
        minimum_frequency = kwargs.get('minimum_frequency')
        sampling_frequency = kwargs.get('sampling_frequency')

        times2 = times - trigger_time
        surr2 = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', minimum_frequency=minimum_frequency, sampling_frequency=sampling_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y,spin_2z], 
                                                      times=times)
    
        h_lm2, surr_times = surr2.time_domain_oscillatory()
        h_lm_mem2, surr_times = surr2.time_domain_memory()

        window = tukey(surr_times.size, )

        oscillatory2 = gwmemory.utils.combine_modes(h_lm2,iota, phase)
        memory2 = gwmemory.utils.combine_modes(h_lm_mem2, iota,phase)

        #oscillatory2, surr_times = surr2.time_domain_oscillatory(phase=phase, inc=inc)
        #memory2, surr_times = surr2.time_domain_memory(phase=phase, inc=inc)

        window = tukey(surr_times.size, 0.05)

        plus_new2 = oscillatory2["plus"]+memory2["plus"]
        cross_new2 = oscillatory2['cross']+memory2["cross"]
        plus2 = plus_new2 * window
        cross2 = cross_new2 * window
 
        return {"plus": plus2, "cross": cross2}


def mem_freq_XPHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase,**kwargs):
        
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        amplitude = kwargs.get('amplitude')
        
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=-2)
        series.frequency_array = frequencies
        
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', minimum_frequency=minimum_frequency, sampling_frequency=sampling_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], times=series.time_array)
        
        osc, xphm_times = xphm.time_domain_oscillatory(inc=iota, phase=phase)
        mem, xphm_times = xphm.time_domain_memory(inc=iota, phase=phase)
        plus = osc['plus'] + amplitude*mem['plus']
        cross = osc['cross'] + amplitude*mem['cross']

        window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window
        
        
        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)
        
        
        return waveform_fd


def osc_freq_XPHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=-2)
        series.frequency_array = frequencies
        
        waveform = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', minimum_frequency=minimum_frequency, sampling_frequency=sampling_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], 
                                                      times=series.time_array)
        
        osc, surr_times = waveform.time_domain_oscillatory(inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        window = tukey(surr_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        
        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd


def mem_freq_XHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase,**kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=-1.5)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], times=series.time_array)
        
        #inc = 0.4

        osc, xhm_times = xhm.time_domain_oscillatory(modes = modes, inc=iota, phase=phase)
        mem, xhm_times = xhm.time_domain_memory(modes = modes, inc=iota, phase=phase)          # no gamma_lmlm argument is needed, as it is deprecated.
        plus = osc['plus'] + mem['plus']
        cross = osc['cross'] + mem['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd

def osc_freq_XHM(frequencies, mass_ratio, total_mass, luminosity_distance, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z, 
                  iota, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')

        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=-1.5)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[spin_1x, spin_1y, spin_1z], spin_2=[spin_2x, spin_2y, spin_2z], 
                                                      times=series.time_array)
        

        osc, xhm_times = xhm.time_domain_oscillatory(modes = modes, inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)

        return waveform_fd

def osc_test(frequencies, mass_ratio, total_mass, luminosity_distance, chi_1, chi_2,
                  iota, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')

        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[0, 0, chi_1], spin_2=[0, 0, chi_2], 
                                                      times=series.time_array)
        
        osc, xhm_times = xhm.time_domain_oscillatory(inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        _, shift = utils.wrap_at_maximum(waveform=osc)

        waveform_fd = utils.nfft_and_time_shift(kwargs, series, shift, waveform)

        return waveform_fd


def mem_test(frequencies, mass_ratio, total_mass, luminosity_distance, chi_1, chi_2, 
                  iota, phase,**kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[0, 0, chi_1], spin_2=[0, 0, chi_2], times=series.time_array)
        
        osc, xhm_times = xhm.time_domain_oscillatory(inc=iota, phase=phase)
        mem, xhm_times = xhm.time_domain_memory(inc=iota, phase=phase)          # no gamma_lmlm argument is needed, as it is deprecated.
        plus = osc['plus'] + mem['plus']
        cross = osc['cross'] + mem['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        _, shift = utils.wrap_at_maximum(waveform=osc)

        waveform_fd = utils.nfft_and_time_shift(kwargs, series, shift, waveform)

        return waveform_fd

def osc_test2(frequencies, mass_ratio, total_mass, luminosity_distance, chi_1, chi_2,
                  iota, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')

        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[0, 0, chi_1], spin_2=[0, 0, chi_2], 
                                                      times=series.time_array)
        
        osc, xhm_times = xhm.time_domain_oscillatory(inc=iota, phase=phase)
        plus = osc['plus']
        cross = osc['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        #_, shift = utils.wrap_at_maximum(waveform=osc)

        waveform_fd = utils.nfft(waveform, sampling_frequency)

        return waveform_fd


def mem_test2(frequencies, mass_ratio, total_mass, luminosity_distance, chi_1, chi_2, 
                  iota, phase,**kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=0)
        series.frequency_array = frequencies
        
        xhm = gwmemory.waveforms.Approximant(name='IMRPhenomXHM', minimum_frequency=minimum_frequency, 
                                               distance=luminosity_distance, q= mass_ratio, total_mass=total_mass, 
                                                      spin_1=[0, 0, chi_1], spin_2=[0, 0, chi_2], times=series.time_array)
        
        osc, xhm_times = xhm.time_domain_oscillatory(inc=iota, phase=phase)
        mem, xhm_times = xhm.time_domain_memory(inc=iota, phase=phase)          # no gamma_lmlm argument is needed, as it is deprecated.
        plus = osc['plus'] + mem['plus']
        cross = osc['cross'] + mem['cross']

        window = tukey(xhm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}

        #_, shift = utils.wrap_at_maximum(waveform=osc)

        waveform_fd = utils.nfft(waveform, sampling_frequency)

        return waveform_fd