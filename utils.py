"""
Script for bunch of useful functions for reweighting code.
"""

import bilby
import copy
import numpy as np
from gwpy import segments
import gwpy
import matplotlib.pyplot as plt
from bilby.core.utils.series import infft


def nfft(time_domain_strain, sampling_frequency):
    frequency_domain_strain = dict()
    for mode in time_domain_strain:
        frequency_domain_strain[mode] = np.fft.rfft(time_domain_strain[mode])
        frequency_domain_strain[mode] /=sampling_frequency
    return frequency_domain_strain


def ifft(frequency_domain_strain, sampling_frequency):
    time_domain_strain = dict()
    for mode in frequency_domain_strain:
        time_domain_strain[mode] = np.fft.irfft(frequency_domain_strain[mode])
        time_domain_strain[mode] *= sampling_frequency
    
    return time_domain_strain

def nfft_and_time_shift(kwargs, series, shift, waveform):
    time_shift = shift* (series.time_array[1]-series.time_array[0])    # ?
    waveform_fd = nfft(waveform, series.sampling_frequency)             # ?
    # for mode in waveform:
    #     indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
    #     waveform_fd[mode][indexes] = 0
    waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                                    duration=series.duration, shift=time_shift)
    return waveform_fd


def get_alpha(roll_off, duration):
    return 2*roll_off/duration


def apply_time_shift_frequency_domain(waveform, frequency_array, duration, shift):
    wf = copy.deepcopy(waveform)
    #print('shift', shift)
    for mode in wf:
        wf[mode] = wf[mode] * np.exp(-2j * np.pi * (duration + shift) * frequency_array)
    return wf


def wrap_at_maximum(waveform):
    max_index = np.argmax(np.real(waveform['plus'] - 1j * waveform['cross']))
    shift = len(waveform['plus'])- max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=copy.deepcopy(waveform))
    return waveform, shift


def wrap_by_n_indices(shift, waveform):
    for mode in copy.deepcopy(waveform):
        waveform[mode] = np.roll(waveform[mode], shift=shift)
    return waveform

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


def plot_fd_data_and_waveforms(ifo, osc_model, mem_model, full_model, parameters):

    osc_fd_strain = osc_model.frequency_domain_strain(parameters)
    mem_fd_strain = mem_model.frequency_domain_strain(parameters)
    full_fd_strain = full_model.frequency_domain_strain(parameters)

    osc_response = ifo.get_detector_response(osc_fd_strain, parameters)
    mem_response = ifo.get_detector_response(mem_fd_strain, parameters)
    full_response = ifo.get_detector_response(full_fd_strain, parameters)

    fd_data = ifo.frequency_domain_strain
    frequencies = ifo.frequency_array
    det = ifo.name

    plt.loglog(frequencies, np.abs(fd_data), label=f'{det} data')
    plt.loglog(frequencies, np.abs(osc_response), label='osc waveform')
    plt.loglog(frequencies, np.abs(mem_response), label='mem_waveform')
    plt.loglog(frequencies, np.abs(full_response), label='osc+mem waveform')


def fit_vs_amplitude(amplitudes, ifo, samples, source_model, waveform_arguments, trigger_time, outdir):

    det = ifo.name

    whitened = np.array(copy.copy(ifo.whitened_frequency_domain_strain))
    print(ifo.frequency_array)
    print(whitened[ifo.frequency_array>19.5])
    whitened[ifo.frequency_array>300] = 0

    td_strain = infft(whitened, sampling_frequency = 2048)

    time = ifo.time_array

    plt.figure()
    plt.plot(time, td_strain, label=f'whitened {det} data')

    max_like = np.argmax(samples['log_likelihood'])
    posterior = samples.iloc[max_like].to_dict()

    for amplitude in amplitudes:
        waveform_arguments['amplitude'] = amplitude
        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=ifo.duration,
            sampling_frequency=ifo.sampling_frequency,
            frequency_domain_source_model= source_model,
            parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,

        )

        frequency_domain_strain = ifo.get_detector_response(waveform_generator.frequency_domain_strain(posterior), 
                                                                        posterior)

        fd_strain = frequency_domain_strain/ifo.power_spectral_density_array**0.5
        time_domain_strain= bilby.core.utils.series.infft(fd_strain, sampling_frequency=2048)

        plt.plot(time, time_domain_strain, label=f'A = {amplitude}')

    plt.xlim(trigger_time-0.1, trigger_time+0.1)
    plt.xlabel('time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+f'data_with_full_waveform_different_A.png')


def check_template_fit(amplitude, ifo, samples, waveform_generator, trigger_time, outdir):
    from bilby.core.utils.series import infft
    import matplotlib.pyplot as plt

    det = ifo.name

    whitened = np.array(copy.copy(ifo.whitened_frequency_domain_strain))
    whitened[ifo.frequency_array>300] = 0

    td_strain = infft(whitened, sampling_frequency = 2048)

    time = ifo.time_array

    max_like = np.argmax(samples['log_likelihood'])
    posterior = samples.iloc[max_like].to_dict()

    frequency_domain_strain = ifo.get_detector_response(waveform_generator.frequency_domain_strain(posterior), 
                                                                    posterior)

    fd_strain = frequency_domain_strain/ifo.power_spectral_density_array**0.5
    time_domain_strain= bilby.core.utils.series.infft(fd_strain, sampling_frequency=2048)

    #time_array = waveform_generator.time_array
    print(time_domain_strain)

    plt.figure()
    plt.title(f'Amplitude = {amplitude}')
    plt.plot(time, td_strain, label=f'whitened {det} data')
    plt.plot(time, time_domain_strain, label='full waveform')
    plt.xlim(trigger_time-0.1, trigger_time+0.1)
    plt.xlabel('time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+f'amplitude{amplitude}_template_fit_on_whitened_data_{det}_sample48.png')