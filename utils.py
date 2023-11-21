"""
Script for bunch of useful functions for reweighting code.
"""

import bilby
import copy
import numpy as np
from gwpy import segments
import gwpy


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
