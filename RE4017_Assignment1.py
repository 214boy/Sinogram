###########
#Authors: Ciaran Hickey (17204267)
#         Edmond Keogh (15144895)
#         Eoin Fitzgibbon (17229618)
#         Fiachra O' Sullivan (17220688)
###########



import scipy
from scipy import signal
from skimage import transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


# accept array and image and crop to 1:1 aspect ratio
def unpad(array, image):
    hyp = np.shape(image)[1]
    # image side calculation from hypotenuse
    im_side = math.sqrt(hyp ** 2 / 2)
    x_margin = round((np.shape(array)[0] - im_side) / 2)
    y_margin = round((np.shape(array)[1] - im_side) / 2)
    # crop the array to the correct aspect ratio
    cropped = array[x_margin:np.shape(array)[0] - x_margin, y_margin:np.shape(array)[1] - y_margin]
    return cropped


# main procedure for the sinogram
def back_projection(filtered_image):
    # find rows, cols and angular resolution
    rows, cols = np.shape(filtered_image)
    angle = 180 / rows

    # use np.tile to take each row and make copies along hytotenuse length, iterate and sum results
    back_proj = np.tile(filtered_image[0], (round(math.sqrt(rows ** 2 + cols ** 2)), 1))
    for i in range(rows - 1):
        new_row = np.tile(filtered_image[i + 1], (round(math.sqrt(rows ** 2 + cols ** 2)), 1))
        # rotate for each iteration
        rot_row = transform.rotate(new_row, (angle * (i + 1)))
        back_proj = np.add(rot_row, back_proj)
    return back_proj


# Scale and normalise
def to_8bit(channel):
    ch_hi, ch_lo = channel.max(), channel.min()
    ch_norm = 255.999 * (channel - ch_lo) / (ch_hi - ch_lo)
    ch_8bit = np.floor(ch_norm).astype('uint8')
    return ch_8bit


# Perform ramp filtering on fast fourier transform
def ramp_filter_ffts(ffts):
    ramp = np.floor(np.arange(0.5, ffts.shape[1] // 2 + 0.1, 0.5))
    return ffts * ramp


# use window function to remove noise (hamming or hann depends on c arg)
def window(ffts, c=0.54):
    steps = np.arange(0, ffts.shape[1] // 2, .5)
    divisor = math.pi * steps / ffts.shape[1]
    hamming = c + (1 - c) * np.cos(divisor)
    return ffts * hamming


# import image
image = np.array(Image.open("sinogram.png"))  # Open the image and convert to a Numpy array.

# seperate channels
r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

# fft on each channel
r1 = scipy.fft.rfft(r, axis=1)
g1 = scipy.fft.rfft(g, axis=1)
b1 = scipy.fft.rfft(b, axis=1)

#For all calculations each channel is calculated seperately

# ramp filter each channel
red_ramp_signal = ramp_filter_ffts(r1)
green_ramp_signal = ramp_filter_ffts(g1)
blue_ramp_signal = ramp_filter_ffts(b1)

# use hamming window on ramp filtered
red_hamming_signal = window(red_ramp_signal)
green_hamming_signal = window(green_ramp_signal)
blue_hamming_signal = window(blue_ramp_signal)

# use hann window on ramp filtered
red_hann_signal = window(red_ramp_signal, c=.5)
green_hann_signal = window(green_ramp_signal, c=.5)
blue_hann_signal = window(blue_ramp_signal, c=.5)

# inverse fft using only ramp filtering without window
red_ramp_image = scipy.fft.irfft(red_ramp_signal)
green_ramp_image = scipy.fft.irfft(green_ramp_signal)
blue_ramp_image = scipy.fft.irfft(blue_ramp_signal)

# inverse fft hamming image
red_hamming_image = scipy.fft.irfft(red_hamming_signal)
blue_hamming_image = scipy.fft.irfft(blue_hamming_signal)
green_hamming_image = scipy.fft.irfft(green_hamming_signal)

# inverse fft on hann in each seperate channel
red_hann_image = scipy.fft.irfft(red_hann_signal)
green_hann_image = scipy.fft.irfft(green_hann_signal)
blue_hann_image = scipy.fft.irfft(blue_hann_signal)

#perform back_proj on each channel for ramp employing 8_bit and unpad funcs
red_back_proj = unpad(to_8bit(back_projection(red_ramp_image)), image)
green_back_proj = unpad(to_8bit(back_projection(green_ramp_image)), image)
blue_back_proj = unpad(to_8bit(back_projection(blue_ramp_image)), image)

red_back_proj_hamming = unpad(to_8bit(back_projection(red_hamming_image)), image)
green_back_proj_hamming = unpad(to_8bit(back_projection(green_hamming_image)), image)
blue_back_proj_hamming = unpad(to_8bit(back_projection(blue_hamming_image)), image)

red_back_proj_hann = unpad(to_8bit(back_projection(red_hann_image)), image)
green_back_proj_hann = unpad(to_8bit(back_projection(green_hann_image)), image)
blue_back_proj_hann = unpad(to_8bit(back_projection(blue_hann_image)), image)

red_unfilt_back_proj = unpad(to_8bit(back_projection(r)), image)
green_unfilt_back_proj = unpad(to_8bit(back_projection(g)), image)
blue_unfilt_back_proj = unpad(to_8bit(back_projection(b)), image)

ramp_image = np.dstack((red_back_proj, green_back_proj, blue_back_proj))
hamming_image = np.dstack((red_back_proj_hamming, green_back_proj_hamming, blue_back_proj_hamming))
hann_image = np.dstack((red_back_proj_hann, green_back_proj_hann, blue_back_proj_hann))
unfilt_image = np.dstack((red_unfilt_back_proj, green_unfilt_back_proj, blue_unfilt_back_proj))

#plots figures
plt.figure("Red Channel Sinogram")
plt.imshow(r, cmap='gray')

plt.figure("Red Channel FFT Row 0")
plt.plot(np.arange(len(r1[0])), r1[0])


#filtered stem plots for each of the filtering methods
fig, axs = plt.subplots(3)
fig.suptitle('Filtered Stem Plots:')

axs[0].set_title("Ramp:")
axs[0].stem(red_ramp_signal[0])

axs[1].set_title("Hamming:")
axs[1].stem(red_hamming_signal[0])

axs[2].set_title("Hann:")
axs[2].stem(red_hann_signal[0])

#displaying the red channels filtered sinogram
fig, axs = plt.subplots(3)
fig.suptitle('Red  Channel Filtered Sinogram:')
axs[0].axis('off')
axs[0].set_title("Ramp:")
axs[0].imshow(red_ramp_image, cmap='gray_r')
axs[1].axis('off')
axs[1].set_title("Hamming:")
axs[1].imshow(red_hamming_image, cmap='gray_r')
axs[2].axis('off')
axs[2].set_title("Hann:")
axs[2].imshow(red_hann_image, cmap='gray_r')

#displaying all outputs for each filter and unfiltered
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.suptitle('Back Projected Images:')
ax1.axis('off')
ax1.set_title("Ramp")
ax1.imshow(ramp_image)
ax2.axis('off')
ax2.set_title("Hamming")
ax2.imshow(hamming_image)
ax3.axis('off')
ax3.set_title("Hann")
ax3.imshow(hann_image)
ax4.axis('off')
ax4.set_title("Unfiltered")
ax4.imshow(unfilt_image)

plt.show()