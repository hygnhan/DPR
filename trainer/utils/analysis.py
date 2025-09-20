import torch as t
import numpy as np
import matplotlib.pyplot as plt




def gradient_analysis(label, b_label,magnitude,  log):
    major = t.where(label == b_label)
    minor = t.where(label != b_label)

    magnitude /= t.max(magnitude)
    
    magnitude = magnitude.numpy()
    
    mag_maj = np.histogram(magnitude[major], bins=10, range=(0,1))[0]
    mag_min = np.histogram(magnitude[minor], bins=10, range=(0,1))[0]
    
    log('Majority Magnitude histogram')
    log(mag_maj)
    log('Minority Magnitude histogram')
    log(mag_min)
    
def prob_analysis(label, b_label, mag_prob, log):
    major = t.where(label == b_label)
    minor = t.where(label != b_label)

    mag_prob = mag_prob.numpy()
    
    # Sampling probability (Sum)
    log('Majority magnitude Probability : %f '% (np.sum(mag_prob[major]) ))
    log('Minority magnitude Probability : %f '% (np.sum(mag_prob[minor]) ))

    mag_prob /= np.max(mag_prob)
    
    mag_maj = np.histogram(mag_prob[major], bins=10, range=(0,1))[0]
    mag_min = np.histogram(mag_prob[minor], bins=10, range=(0,1))[0]

    log('Majority magnitude probability histogram')
    log(mag_maj)
    log('Minority magnitude probability histogram')
    log(mag_min)
