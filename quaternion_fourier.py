import numpy as np
import quaternion # From package numpy-quaternion

def fqft(signal, mu=None):
    '''
    Calculates the quaternion fourier transform of *signal*.
    Requires mu**2 == -1. 
    '''
    if mu is None:
        mu = np.quaternion(0, 1, 0, 0)

    signal = quaternion.as_float_array(signal)
    red = signal[:, :, 1]
    green = signal[:, :, 2]
    blue = signal[:, :, 3]
    red_ft = np.fft.fft(red)
    green_ft = np.fft.fft(green)
    blue_ft = np.fft.fft(blue)

    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = i*(np.real(red_ft) + mu*np.imag(red_ft)) + \
         j*(np.real(green_ft) + mu*np.imag(green_ft)) + \
         k*(np.real(blue_ft) + mu*np.imag(blue_ft))

    return ft

def fiqft(signal, mu=None):
    '''
    Calculates the quaternion fourier transform of *signal*.
    Requires mu**2 == -1 and for *mu* to be the same value used in the function
    fqft. 
    '''
    if mu is None:
        mu = np.quaternion(0, 1, 0, 0)

    signal = quaternion.as_float_array(signal)
    quat_real = signal[:, :, 0]
    quat_i = signal[:, :, 1]
    quat_j = signal[:, :, 2]
    quat_k = signal[:, :, 3]
    real_ift = np.fft.ifft(quat_real)
    i_ift = np.fft.ifft(quat_i)
    j_ift = np.fft.ifft(quat_j)
    k_ift = np.fft.ifft(quat_k)

    i = np.quaternion(0, 1, 0, 0)
    j = np.quaternion(0, 0, 1, 0)
    k = np.quaternion(0, 0, 0, 1)

    ft = np.real(real_ift) + mu*np.imag(real_ift) + \
         i*(np.real(i_ift) + mu*np.imag(i_ift)) + \
         j*(np.real(j_ift) + mu*np.imag(j_ift)) + \
         k*(np.real(k_ift) + mu*np.imag(k_ift))

    return ft 
