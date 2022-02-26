import numpy as np
import quaternion # From package numpy-quaternion
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from quaternion_fourier import fqft, fiqft

def proj_bandlimit(signal, bandlimit, *, reverse=False, fourier=np.fft.fft, inv_fourier=np.fft.ifft):
    '''
    Projects a discrete signal onto the space of bandlimited signals with
    bandlimit *bandlimit*. 

    Set *reverse=True* to have it keep the middle *bandlimit* components of
    the signal. Otherwise it will keep the middle *bandlimit* componenents.
    '''
    dims = np.array(signal.shape)
    if np.any(bandlimit > dims):
        raise ValueError("Can't have a bandlimit larger than the signals "
                         "shape.")
    if reverse:
        bandlimit_mat = np.zeros([bandlimit for _ in dims], dtype=np.uint8)
    else:
        bandlimit_mat = np.ones([bandlimit for _ in dims], dtype=np.uint8)
    pad_sides_len = dims - bandlimit
    padding_len = list()
    for padding in pad_sides_len:
        if padding % 2 != 0:
            padding_len.append((int(np.floor(padding/2) + 1),
                                int(np.floor(padding/2))))
        else:
            padding_len.append((int(padding/2), int(padding/2)))
    if reverse:
        bandlimit_mat = np.pad(bandlimit_mat, padding_len, mode='constant',
                               constant_values=(1))
    else:
        bandlimit_mat = np.pad(bandlimit_mat, padding_len, mode='constant',
                               constant_values=(0))
    return inv_fourier(bandlimit_mat * fourier(signal))

def proj_timelimit(signal, timelimit, *, reverse=False):
    '''
    Projects a discrete signal onto the space of timelimited signals with
    timelimit *timelimit*. 

    Set *reverse=True* to have it keep the middle *timelimit* components of
    the signal. Otherwise it will keep the middle *timelimit* componenents.
    '''
    dims = np.array(signal.shape)
    if np.any(timelimit > dims/2):
        raise ValueError("Can't have a bandlimit larger than half the "
                         "signals shape.")
    if reverse:
        timelimit_mat = np.zeros([timelimit for _ in dims], dtype=np.uint8)
    else:
        timelimit_mat = np.ones([timelimit for _ in dims], dtype=np.uint8)
    pad_sides_len = dims - timelimit
    padding_len = list()
    for padding in pad_sides_len:
        if padding % 2 != 0:
            padding_len.append((int(np.floor(padding/2) + 1), 
                                int(np.floor(padding/2))))
        else:
            padding_len.append((int(padding/2), int(padding/2)))
    if reverse:
        timelimit_mat = np.pad(timelimit_mat, padding_len, mode='constant',
                               constant_values=(1))
    else:
        timelimit_mat = np.pad(timelimit_mat, padding_len, mode='constant',
                               constant_values=(0))
    return timelimit_mat * signal

def cut_out_region(signal, region, *, reverse=False):
    '''
    Cuts out a region of the signal. The region can either be a tuple of tuples
    (the top left and bottom right corners of rectangles to cut out) or a
    numpy array of zeros and ones with the same shape as *signal*. Entry (i, j)
    will be cut out of the signal if region[i][j] == 0. 

    Set *reverse=True* to have it keep the areas defined by *region* instead of
    removing them. 
    '''
    if isinstance(region, tuple):
        if len(region) % 2 != 0:
            raise ValueError("There must be an even number of tuples.")
        cut_out_mat = np.ones(signal.shape, dtype=np.uint8)
        for i in range(int(len(region)/2)):
            top_left = region[2*i]
            bot_right = region[2*i+1]
            for row in range(top_left[1], bot_right[1]+1):
                for col in range(top_left[0], bot_right[0]+1):
                    if row == signal.shape[0] or col == signal.shape[1]:
                        continue
                    cut_out_mat[row][col] = 0
    elif isinstance(region, np.array):
        cut_out_mat = region
    else:
        raise ValueError("region must be either a tuple of tuples are a numpy "
                         "array.")

    if reverse:
        cut_out_mat = np.mod(cut_out_mat + 1, 2)

    return cut_out_mat * signal


def calc_error(signal1, signal2):
    '''
    Calculates the absolute distance (l1 distance) between two signals. 
    '''
    return np.sum(np.abs(signal1-signal2)**2)

def projection_reconstruction(original_signal, bandlimit, region, iterations,
                              reverse_bandlimit=False, reverse_region=True,
                              true_signal=None, fourier=np.fft.fft,
                              inv_fourier=np.fft.ifft):
    signal = deepcopy(original_signal)
    summed_signal = proj_bandlimit(original_signal, bandlimit, 
                                   reverse=reverse_bandlimit, fourier=fourier,
                                   inv_fourier=inv_fourier)
    prev_term = deepcopy(summed_signal)
    if true_signal is not None:
        errors = list()
    for i in range(iterations):
        temp =  proj_bandlimit(cut_out_region(prev_term, region, 
                                              reverse=reverse_region), 
                               bandlimit, reverse=reverse_bandlimit,
                               fourier=fourier, inv_fourier=inv_fourier)
        prev_term = deepcopy(temp)
        summed_signal += temp
        if true_signal is not None:
            errors.append(calc_error(summed_signal, true_signal))
    if true_signal is not None:
        return summed_signal, errors
    else:
        return summed_signal

def import_image(img_path, greyscale=False):
    '''
    Saves the image located at img_path into a numpy array.

    Set *greyscale=True* to have it convert the image to a greyscale one.
    '''
    img = Image.open(img_path).convert('RGBA')
    background = Image.new('RGBA', img.size, (255))
    img = Image.alpha_composite(background, img).convert('RGB')
    if greyscale:
        img = img.convert('L')
    return np.array(img)

def display_image(img, greyscale=False):
    '''
    Displays an image that is stored as a numpy array.
    '''
    if greyscale:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.show()
    plt.clf()

def rgb_to_quat(img):
    '''
    Converts an MxNx3 numpy array of integers into an MxN numpy array of
    quaternions in the form 0 + REDe1 + GREENe2 + BLUEe1e2. 
    '''
    real_layer = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    new_img = np.concatenate((real_layer, img), axis=2)
    return quaternion.as_quat_array(new_img)

def quat_to_rgb(img):
    '''
    Converts an MxN numpy array of quaternions into an MxNx3 numpy array of
    integers.
    '''
    output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = quaternion.as_float_array(img)
    output[:, :, 0] = img[:, :, 1]
    output[:, :, 1] = img[:, :, 2]
    output[:, :, 2] = img[:, :, 3]
    return output


BANDLIMIT = 220
#REGION = ((50, 50), (53, 200), (200, 25), (200, 25))
REGION = ((50, 50), (60, 150))
#REGION = ((50, 10), (60, 40))
ITERATIONS = 100


if __name__ == "__main__":
    img_path = '../Images/bird.jpg'

    # First perform the reconstruction on a greyscale image using the regular
    # fast fourier transform. 
    original_img = import_image(img_path, greyscale=True)
    original_img = proj_bandlimit(original_img, BANDLIMIT, reverse=True)
    img = cut_out_region(original_img, REGION)
    reconstructed, errors = projection_reconstruction(img, BANDLIMIT, REGION,
                                                      ITERATIONS, 
                                                      reverse_bandlimit=True, 
                                                      true_signal=original_img)
    error = calc_error(reconstructed, original_img)
    print(f"Least Absolute Error: {error}")
    display_image(np.real(img), greyscale=True)
    display_image(np.real(reconstructed), greyscale=True)
    plt.plot([x for x in range(ITERATIONS)], np.log(errors))
    plt.show()
    
    # Now reconstruct a colour image using the fast quaternion fourier
    # transform. 
    img = import_image(img_path)
    img_quat = rgb_to_quat(img)
    img_bandlimit = proj_bandlimit(img_quat, BANDLIMIT, reverse=True, fourier=fqft, inv_fourier=fiqft)
    img_loss = cut_out_region(img_bandlimit, REGION)
    reconstructed_img, errors = projection_reconstruction(img_loss, BANDLIMIT, REGION,
                                                      ITERATIONS, 
                                                      reverse_bandlimit=True,
                                                      true_signal=img_bandlimit,
                                                      fourier=fqft,
                                                      inv_fourier=fiqft)
    error = calc_error(reconstructed_img, img_bandlimit)
    print(f"Least Absolute Error: {error}") 
    display_image(quat_to_rgb(img_loss))
    display_image(quat_to_rgb(reconstructed_img))
    plt.plot([x for x in range(ITERATIONS)], np.log(errors))
    plt.show()

