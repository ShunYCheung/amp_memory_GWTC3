"""
Script for bunch of useful functions for reweighting code.
"""


import bilby
import gwmemory
import copy

import numpy as np


def nfft(time_domain_strain, sampling_frequency):
    frequency_domain_strain = dict()
    for mode in time_domain_strain:
        frequency_domain_strain[mode] = np.fft.rfft(time_domain_strain[mode])
        frequency_domain_strain[mode] /=sampling_frequency
    return frequency_domain_strain

def nfft_and_time_shift(kwargs, series, shift, waveform):
    time_shift = kwargs.get('time_shift', 0.)
    time_shift += shift* (series.time_array[1]-series.time_array[0])    # ?
    waveform_fd = nfft(waveform, series.sampling_frequency)             # ?
    for mode in waveform:
        indexes = np.where(series.frequency_array < kwargs.get('minimum_frequency', 20))
        waveform_fd[mode][indexes] = 0
    waveform_fd = apply_time_shift_frequency_domain(waveform=waveform_fd, frequency_array=series.frequency_array,
                                                    duration=series.duration, shift=time_shift)
    return waveform_fd


def get_alpha(roll_off, duration):
    return roll_off/duration


def apply_time_shift_frequency_domain(waveform, frequency_array, duration, shift):
    wf = copy.deepcopy(waveform)
    for mode in wf:
        wf[mode] = wf[mode] * np.exp(-2j * np.pi * (duration + shift) * frequency_array)
    return wf


def wrap_at_maximum(waveform):
    max_index = np.argmax(np.abs(waveform['plus'] - 1j * waveform['cross']))
    shift = len(waveform['plus'])- max_index
    waveform = wrap_by_n_indices(shift=shift, waveform=copy.deepcopy(waveform))
    return waveform, shift


def wrap_by_n_indices(shift, waveform):
    for mode in copy.deepcopy(waveform):
        waveform[mode] = np.roll(waveform[mode], shift=shift)
    return waveform


