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
            print("Data quality check: FAILED. \n"
                "{det} does not have quality data for "
                "{inactive_duration}s out of {total_duration}s".format(
                    det=det,
                    inactive_duration=inactive_duration,
                    total_duration=total_duration,
                ))
            return False
        else:
            data_is_good = flag.isgood
            if data_is_good:
                print("Data quality check: PASSED.")
            else:
                print("Data quality check: FAILED. \n"
                    "{det} ifo is not in a good state".format(
                        det=det,
                    ))
            return data_is_good

    except Exception as e:
        print(f"Error in Data Quality Check: {e}.")
        return False
    


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
    #plt.loglog(frequencies, np.abs(osc_response + mem_response), label='alt osc+mem')
    plt.loglog(frequencies, np.abs(osc_response), label='osc waveform')
    plt.loglog(frequencies, np.abs(mem_response), label='mem_waveform')
    plt.loglog(frequencies, np.abs(full_response), label='osc+mem waveform')


def fit_vs_amplitude(amplitudes, ifo, posterior, source_model, waveform_arguments, trigger_time, event_name, label, result_label, outdir):
    
    font = {'size'   : 16}
    import matplotlib
    matplotlib.rc('font', **font)

    det = ifo.name

    whitened = np.array(copy.copy(ifo.whitened_frequency_domain_strain))
    whitened[ifo.frequency_array>300] = 0

    td_strain = infft(whitened, sampling_frequency = 2048)

    time = ifo.time_array
    fig, axs = plt.subplots(len(amplitudes), figsize=(13, 11))
    fig.suptitle(f'{event_name}')

    for i, amplitude in enumerate(amplitudes):
        axs[i].set_title(f'A = {amplitude}')
        axs[i].plot(time, td_strain, label=f'whitened {det} data')
        for j, model in enumerate(source_model):
            waveform_arguments['amplitude'] = amplitude
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                duration=ifo.duration,
                sampling_frequency=ifo.sampling_frequency,
                frequency_domain_source_model= model,
                parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=waveform_arguments,

            )

            frequency_domain_strain = ifo.get_detector_response(waveform_generator.frequency_domain_strain(posterior), 
                                                                            posterior)

            fd_strain = frequency_domain_strain/ifo.power_spectral_density_array**0.5
            time_domain_strain= bilby.core.utils.series.infft(fd_strain, sampling_frequency=2048)
            axs[i].plot(time, time_domain_strain, label=f'{label[j]}')
        axs[i].legend()
        axs[i].set_xlim(trigger_time-0.75+2, trigger_time+2-0.45)
        axs[i].set_ylim(-100, 100)
        
    for ax in axs.flat:
        ax.set(xlabel='time (s)')
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(outdir+f'{event_name}_whitened_data_with_waveforms_{result_label}_{det}.png')
    plt.savefig(outdir+f'{event_name}_whitened_data_with_waveforms_{result_label}_{det}.pdf')


def check_template_fit(amplitude, ifo, samples, waveform_generator, memory_only, trigger_time, outdir):
    from bilby.core.utils.series import infft
    import matplotlib.pyplot as plt

    det = ifo.name

    whitened = np.array(copy.copy(ifo.whitened_frequency_domain_strain))
    whitened[ifo.frequency_array>1000] = 0

    td_strain = infft(whitened, sampling_frequency = 2048)

    time = ifo.time_array

    max_like = np.argmax(samples['log_likelihood'])
    posterior = samples.iloc[80].to_dict()

    frequency_domain_strain = ifo.get_detector_response(waveform_generator.frequency_domain_strain(posterior), 
                                                                    posterior)

    fd_strain = frequency_domain_strain/ifo.power_spectral_density_array**0.5
    time_domain_strain= bilby.core.utils.series.infft(fd_strain, sampling_frequency=2048)

    mem_frequency_domain_strain = ifo.get_detector_response(memory_only.frequency_domain_strain(posterior), 
                                                                    posterior)

    mem_fd_strain = mem_frequency_domain_strain/ifo.power_spectral_density_array**0.5
    mem_time_domain_strain= bilby.core.utils.series.infft(mem_fd_strain, sampling_frequency=2048)


    #time_array = waveform_generator.time_array
    print(time_domain_strain)

    plt.figure()
    plt.title(f'Amplitude = {amplitude}')
    plt.plot(time, ifo.time_domain_strain, label=f'{det} data')
    plt.plot(time, time_domain_strain, label='full waveform')
    plt.plot(time, mem_time_domain_strain, label='memory waveform')
    plt.xlim(trigger_time-0.1, trigger_time+0.1)
    plt.xlabel('time (s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir+f'amplitude{amplitude}_template_fit_on_whitened_data_{det}_sample48.png')