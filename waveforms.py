import gwmemory
import bilby
from scipy.signal.windows import tukey
import utils
import matplotlib.pyplot as plt
import numpy as np


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

        # Create a generator
        xphm = gwmemory.waveforms.Approximant(name='IMRPhenomXPHM', 
                                                minimum_frequency=minimum_frequency,
                                                sampling_frequency=sampling_frequency, 
                                                distance=luminosity_distance, 
                                                q= mass_1/mass_2, 
                                                total_mass=mass_1+mass_2, 
                                                spin_1=[spin_1x, spin_1y, spin_1z], 
                                                spin_2=[spin_2x, spin_2y, spin_2z], 
                                                times=series.time_array)
        
        osc_bilby = waveform_generator.time_domain_strain(parameters = parameters)
        osc_gw, xphm_times = xphm.time_domain_oscillatory(inc=iota, phase=phase)

        
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


def mem_freq_XPHM(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
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
                                          )
    

    # call the time domain oscillatory and memory components. 
#     osc_test = bilby.gw.source.lal_binary_black_hole(frequency_array=frequencies, 
#                                                      mass_1=mass_1,
#                                                      mass_2=mass_2,
#                                                      luminosity_distance=luminosity_distance,
#                                                      a_1=a_1,
#                                                      a_2=a_2)
    osc = waveform_generator.frequency_domain_strain(parameters = parameters)
    osc_td = waveform_generator.time_domain_strain(parameters = parameters)
    osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=phase)
    mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=phase)
    
    _, shift = utils.wrap_at_maximum(osc_ref)
    _, shift2 = utils.wrap_at_maximum(osc_td)
    new_shift = shift - shift2
    print('shift', shift)
    print('shift2', shift2)

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
    waveform_approximant = kwargs.get('waveform_approximant')
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
                                          )
    

    # call the time domain oscillatory and memory components. 
    osc = bilby.gw.source.lal_binary_black_hole(frequency_array=frequencies, 
                                                     mass_1=mass_1,
                                                     mass_2=mass_2,
                                                     luminosity_distance=luminosity_distance,
                                                     a_1=a_1,
                                                     a_2=a_2, 
                                                     tilt_1=tilt_1, 
                                                     phi_12=phi_12, 
                                                     tilt_2=tilt_2, 
                                                     phi_jl=phi_jl, 
                                                     theta_jn=theta_jn, 
                                                     phase=phase, 
                                                     waveform_approximant=waveform_approximant,
                                                     reference_frequency=reference_frequency,
                                                     minimum_frequency=minimum_frequency,
                                                     maximum_frequency=maximum_frequency,)
    
    osc_td = bilby.core.series.utils.infft(osc, sampling_frequency)
    osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=phase)
    mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=phase)
    
    _, shift = utils.wrap_at_maximum(osc_ref)
    _, shift2 = utils.wrap_at_maximum(osc_td)
    new_shift = shift - shift2
    print('shift', shift)
    print('shift2', shift2)

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
    
    return waveform_fd

def mem_freq_XPHM_only(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
        
    duration = kwargs.get('duration')
    roll_off = kwargs.get('roll_off')
    minimum_frequency = kwargs.get("minimum_frequency")
    maximum_frequency = kwargs.get('maximum_frequency')
    sampling_frequency = kwargs.get('sampling_frequency')
    amplitude = kwargs.get('amplitude')
    reference_frequency = kwargs.get("reference_frequency")
    waveform_approximant = kwargs.get('waveform_approximant')
    waveform_generator = kwargs.get("bilby_generator")
    activate_shift = kwargs.get('activate_shift')
                            
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
                                    #       iota = iota,
                                    #       phiRef = phase
                                            )

    osc = bilby.gw.source.lal_binary_black_hole(frequency_array=frequencies, 
                                                    mass_1=mass_1,
                                                    mass_2=mass_2,
                                                    luminosity_distance=luminosity_distance,
                                                    a_1=a_1,
                                                    a_2=a_2, 
                                                    tilt_1=tilt_1, 
                                                    phi_12=phi_12, 
                                                    tilt_2=tilt_2, 
                                                    phi_jl=phi_jl, 
                                                    theta_jn=theta_jn, 
                                                    phase=phase, 
                                                    waveform_approximant=waveform_approximant,
                                                    reference_frequency=reference_frequency,
                                                    minimum_frequency=minimum_frequency,
                                                    maximum_frequency=maximum_frequency,)
    
    osc_td_test=dict()
    for key in osc:
        osc_td_test[key] = bilby.core.series.utils.infft(osc[key], sampling_frequency)
        
    osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=phase)
    osc_td = waveform_generator.time_domain_strain(parameters = parameters)
    mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=phase)
    _, shift = utils.wrap_at_maximum(osc_ref)
    _, shift2 = utils.wrap_at_maximum(osc_td)
    

    # testing waveform
    plt.figure()
    plt.plot(series.time_array, osc_td['plus']-1j*osc_td['cross'], label='waveform generator')
    plt.plot(series.time_array, osc_td_test['plus']-1j*osc_td_test['cross'], label='source model')
    plt.legend()
    plt.xlim(1.6, 2.0)
    plt.savefig('test_results/wf_generator_vs_freq_source_model.png')

    # print('shift', shift)
    # print('shift2', shift2)

    if activate_shift:
            new_shift = shift - shift2
    else:
            new_shift = shift

    plus = amplitude*mem['plus']
    cross = amplitude*mem['cross']
    
    window = tukey(xphm_times.size, utils.get_alpha(roll_off, duration))
    new_plus = plus * window
    new_cross = cross * window
    
    waveform = {'plus': new_plus, 'cross': new_cross}

    # perform nfft to obtain frequency domain strain
    waveform_fd = utils.nfft_and_time_shift(kwargs, series, new_shift, waveform)
    
    return waveform_fd


def mem_freq_XPHM_only_v2(frequencies, mass_1, mass_2, luminosity_distance, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, phase, **kwargs):
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
    waveform_approximant = kwargs.get('waveform_approximant')
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
                                          )
    

    # call the time domain oscillatory and memory components. 
    osc = bilby.gw.source.lal_binary_black_hole(frequency_array=frequencies, 
                                                     mass_1=mass_1,
                                                     mass_2=mass_2,
                                                     luminosity_distance=luminosity_distance,
                                                     a_1=a_1,
                                                     a_2=a_2, 
                                                     tilt_1=tilt_1, 
                                                     phi_12=phi_12, 
                                                     tilt_2=tilt_2, 
                                                     phi_jl=phi_jl, 
                                                     theta_jn=theta_jn, 
                                                     phase=phase, 
                                                     waveform_approximant=waveform_approximant,
                                                     reference_frequency=reference_frequency,
                                                     minimum_frequency=minimum_frequency,
                                                     maximum_frequency=maximum_frequency,)
    
    
    osc_td_test=dict()
    for key in osc:
        osc_td_test[key] = bilby.core.series.utils.infft(osc[key], sampling_frequency)

    osc_ref, xphm_times = xphm.time_domain_oscillatory(inc=theta_jn, phase=phase)
    mem, xphm_times = xphm.time_domain_memory(inc=theta_jn, phase=phase)
    
    _, shift = utils.wrap_at_maximum(osc_ref)
    _, shift2 = utils.wrap_at_maximum(osc_td_test)
    new_shift = shift - shift2
    print('shift', shift)
    print('shift2', shift2)

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
