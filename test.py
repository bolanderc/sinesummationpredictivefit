# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:32:38 2018

@author: Christian
"""

import pytest
import sinesummationpredictivefit as ssf
import numpy as np
import os.path

def test_decomposition():
    amplitude = 1.5  # m
    omega = 0.5  # rad/s
    offset = 1.0
    phase = np.pi/4.0
    time = np.linspace(0, 100*4*np.pi, 10000, endpoint=True)
    signal = amplitude*np.sin(omega*time + phase) + offset
    harmonic_params = ssf.signal_decomposition(time, signal)
    harmonic_params_test = np.array([amplitude, omega, phase, offset])
    assert np.allclose(harmonic_params, harmonic_params_test, rtol=0.0,
                       atol=10e-2)

def test_downsample():
    amplitude = 1.0
    omega = 2.0
    phase = 0.0
    offset = 0.0
    period = np.pi
    time = np.linspace(0, 100*4*np.pi, 10000, endpoint=True)
    signal = amplitude*np.sin(omega*time)
    min_period = ssf._find_max_period(time, signal, 1)
    downsample_time, downsample_signal = ssf.down_sample(time, signal, 1)
    harmonic_params = ssf.signal_decomposition(downsample_time,
                                               downsample_signal)
    harmonic_params_test = np.array([amplitude, omega, phase, offset])
    test = np.array([True, True])
    test[0] = np.allclose(harmonic_params[1], harmonic_params_test[1], rtol=0.0,
                          atol=10e-3)
    test[1] = np.allclose(min_period, period, rtol=0.0,
                          atol=10e-2)
    assert test.all() == True
    
