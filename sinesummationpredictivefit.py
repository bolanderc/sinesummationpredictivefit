# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:09:24 2018

@author: Christian
"""

import numpy as np
import matplotlib.pyplot as plt


#def sinesummationfit(x, y, num_sine=0, tol=0):
#    

def signal_decomposition(x, y):
    N = len(y)
    dt = (x[-1] - x[0])/N
    num_freqs = N//2
    fourier_transform = np.fft.fft(y)
    fourier_one = fourier_transform[:num_freqs]
    fourier_one[1:num_freqs] = 2*fourier_transform[1:num_freqs]
    amp_spec_one = np.absolute(fourier_one)/N
    phase_spec_one = np.pi/2.0 + np.arctan2(fourier_one.imag,
                                fourier_one.real)
    amplitude = max(amp_spec_one).real
    freq_spec = np.fft.fftfreq(N)/dt
    freq_spec_one = freq_spec[:num_freqs]*2.0*np.pi
    power_spec = (np.abs(fourier_one)**2)*((dt)**2)
    highest_power_index = power_spec.argmax()
    frequency = freq_spec_one[highest_power_index]
    phase = phase_spec_one[highest_power_index]
    offset = np.average(y)
    return np.array([amplitude, frequency, phase, offset])

def down_sample(x, y, num_sines):
    min_period = _find_max_period(x, y, num_sines)
    num_min_periods = (x[-1] - x[0])/min_period
    new_x = np.linspace(x[0], x[-1], num=int(4.0*num_min_periods),
                        endpoint=True)
    new_y = np.interp(new_x, x, y)
    return new_x, new_y

def _find_max_period(x, y, num_sines):
    N = len(y)
    dt = (x[-1] - x[0])/N
    num_freqs = N//2
    fourier_transform = np.fft.fft(y)
    fourier_one = fourier_transform[:num_freqs]
    fourier_one[1:num_freqs] = 2*fourier_transform[1:num_freqs]
    power_spec = (np.abs(fourier_one)**2)*((dt)**2)
    freq_spec = np.fft.fftfreq(N)/dt
    freq_spec_one = freq_spec[:num_freqs]*2.0*np.pi
    peak_indices = np.argpartition(power_spec, -num_sines)[-num_sines:]
    max_freq = freq_spec_one[max(peak_indices)]
    min_period = (2.0*np.pi)/max_freq
    return min_period
    

