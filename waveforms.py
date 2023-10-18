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
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
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

        
        #waveform_fd, freq = bilby.core.utils.series.nfft(waveform, sampling_frequency)
        waveform_fd = utils.nfft(waveform, sampling_frequency=sampling_frequency)
        
       
        
        
        return waveform_fd


def osc_freq_XPHM(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        reference_frequency = kwargs.get('reference_frequency')
        waveform_generator = kwargs.get('bilby_generator')
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies
        
        parameters = dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
                     a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, theta_jn=theta_jn, phase=phase)

        SOLAR_MASS = 1.988409870698051e30

        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
                theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1*SOLAR_MASS, mass_2=mass_2*SOLAR_MASS,
                reference_frequency=reference_frequency, phase=phase)
        print('iota', iota)
        print('theta_jn', theta_jn)
        # Create a generator
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                                minimum_frequency=minimum_frequency,
                                                sampling_frequency=sampling_frequency, 
                                                distance=luminosity_distance, 
                                                q= mass_1/mass_2, 
                                                total_mass=mass_1+mass_2, 
                                                spin_1=[spin_1x, spin_1y, spin_1z], 
                                                spin_2=[spin_2x, spin_2y, spin_2z], 
                                                times=series.time_array,
                                                iota=iota,
                                                phiRef=phase,)
        
        osc_bilby = waveform_generator.time_domain_strain(parameters = parameters)
        osc_gw, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=0)

        
        _, shift = utils.wrap_at_maximum(osc_gw)
        _, shift2 = utils.wrap_at_maximum(osc_bilby)
        new_shift = shift - shift2

        plus = osc_gw['plus']
        cross = osc_gw['cross']

        window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))

        new_plus = plus * window
        new_cross = cross * window

        waveform = {'plus': new_plus, 'cross': new_cross}
        waveform_fd = utils.nfft_and_time_shift(kwargs, series, new_shift, waveform)
        #waveform_fd = utils.nfft(waveform, sampling_frequency)
        plt.figure()
        plt.loglog(frequencies, np.abs(waveform_fd['plus']+waveform_fd['cross']))
        plt.xlim(15, 1000)
        plt.ylim(1e-26, 1e-22)
        plt.savefig('tests/check_end_gwmemory_osc_wf_fd.png')
        
        return waveform_fd


def mem_freq_XPHM_v2(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
    """
    Generates the frequency domain strain of the oscillatory + memory waveform using the approximant IMRPhenomXPHM.
    """
    
    # retrieve the key arguments
    duration = kwargs.get('duration')
    roll_off = kwargs.get('roll_off')
    minimum_frequency = kwargs.get("minimum_frequency")
    maximum_frequency = kwargs.get("maximum_frequency")
    sampling_frequency = kwargs.get('sampling_frequency')
    reference_frequency = kwargs.get("reference_frequency")
    waveform_generator = kwargs.get("bilby_generator")
    amplitude = kwargs.get('amplitude')

    # define the time series based on the frequencies.
    series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
    series.frequency_array = frequencies
    
#     print('bilby frequency array, ', series.frequency_array)
#     print('length of bilby frequency array: ', len(series.frequency_array))

    parameters = dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
                     a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, theta_jn=theta_jn, phase=phase)

    SOLAR_MASS = 1.988409870698051e30

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1*SOLAR_MASS, mass_2=mass_2*SOLAR_MASS,
        reference_frequency=reference_frequency, phase=phase)

    # Create a generator
    xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                          minimum_frequency=minimum_frequency,
                                          sampling_frequency=sampling_frequency, 
                                          distance=luminosity_distance, 
                                          q= mass_1/mass_2, 
                                          total_mass=mass_1+mass_2, 
                                          spin_1=[spin_1x, spin_1y, spin_1z], 
                                          spin_2=[spin_2x, spin_2y, spin_2z], 
                                          times=series.time_array,
                                          iota= iota,
                                          phiRef=phase,)

    # call the time domain oscillatory and memory components. 
    osc = waveform_generator.frequency_domain_strain(parameters = parameters)
    osc_td = waveform_generator.time_domain_strain(parameters = parameters)
    osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=0)
    mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=0)
    
    _, shift = utils.wrap_at_maximum(osc_ref)
    _, shift2 = utils.wrap_at_maximum(osc_td)
    new_shift = shift - shift2

    plus = amplitude*mem['plus']
    cross = amplitude*mem['cross']
    
    window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))
    new_plus = plus * window
    new_cross = cross * window
    
    waveform = {'plus': new_plus, 'cross': new_cross}

    # perform nfft to obtain frequency domain strain
    waveform_fd = utils.nfft_and_time_shift(kwargs, series, new_shift, waveform)
    for mode in waveform_fd:    # add the osc and mem waveforms together in the frequency domain.
        waveform_fd[mode] = waveform_fd[mode] + osc[mode]
    
    print('length of waveform array, ', len(waveform_fd['plus']))
    return waveform_fd


def mem_freq_XPHM_only(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
        
        duration = kwargs.get('duration')
        roll_off = kwargs.get('roll_off')
        minimum_frequency = kwargs.get("minimum_frequency")
        sampling_frequency = kwargs.get('sampling_frequency')
        amplitude = kwargs.get('amplitude')
        reference_frequency = kwargs.get("reference_frequency")
        waveform_generator = kwargs.get("bilby_generator")
                               
        series = bilby.core.series.CoupledTimeAndFrequencySeries(start_time=2-duration)
        series.frequency_array = frequencies

        SOLAR_MASS = 1.988409870698051e30

        parameters = dict(mass_1=mass_1, mass_2=mass_2, luminosity_distance=luminosity_distance,
                     a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, phi_jl=phi_jl, theta_jn=theta_jn, phase=phase)

        iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby.gw.conversion.bilby_to_lalsimulation_spins(
                theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
                phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1*SOLAR_MASS, mass_2=mass_2*SOLAR_MASS,
                reference_frequency=reference_frequency, phase=phase)
        
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                              minimum_frequency=minimum_frequency, 
                                              sampling_frequency=sampling_frequency, 
                                              distance=luminosity_distance, 
                                              q= mass_1/mass_2, 
                                              total_mass=mass_1+mass_2, 
                                              spin_1=[spin_1x, spin_1y, spin_1z], 
                                              spin_2=[spin_2x, spin_2y, spin_2z], 
                                              times=series.time_array,
                                              iota = iota,
                                              phiRef = phase)
        
        osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=0)
        osc_td = waveform_generator.time_domain_strain(parameters = parameters)
        mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=0)
        _, shift = utils.wrap_at_maximum(osc_ref)
        _, shift2 = utils.wrap_at_maximum(osc_td)
        new_shift = shift - shift2

        plus = amplitude*mem['plus']
        cross = amplitude*mem['cross']
        
        window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))
        new_plus = plus * window
        new_cross = cross * window
        
        waveform = {'plus': new_plus, 'cross': new_cross}

        # perform nfft to obtain frequency domain strain
        waveform_fd = utils.nfft_and_time_shift(kwargs, series, new_shift, waveform)
        
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
