# Display of the five rythms + the initial EEG
import numpy as np
import matplotlib.pyplot as plt

# create an image by projecting the information in the XY plan follow by a 2D interpolation between each points.
import math as m

from tqdm import tqdm
from scipy.interpolate import griddata, interp1d
from sklearn.preprocessing import scale

def band_image(frequency_band, electrodes_location, img_size=32):
    locs_2d = elec_proj( electrodes_location )
    frequency_band = frequency_band / np.min( frequency_band )
    frequency_band = frequency_band.reshape( (frequency_band.shape[0], -1) )

    images = image_generation( frequency_band, locs_2d, img_size )
    return images

def elec_proj(loc_3d):
    locs_2d = []
    for l in loc_3d:
        locs_2d.append( azim_proj( l ) )
    return np.asarray( locs_2d )

def azim_proj(pos):
    [r, elev, az] = cart2sph( pos[0], pos[1], pos[2] )

    return pol2cart( az, m.pi / 2 - elev )

def pol2cart(theta, rho):
    return rho * m.cos( theta ), rho * m.sin( theta )

def cart2sph(x, y, z):
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt( x2_y2 + z ** 2 )  # r
    elev = m.atan2( z, m.sqrt( x2_y2 ) )  # Elevation
    az = m.atan2( y, x )  # Azimuth
    return r, elev, az

def image_generation(feature_matrix, electrodes_loc, n_gridpoints):
    n_electrodes = electrodes_loc.shape[0]  # number of electrodes
    n_bands = feature_matrix.shape[1] // n_electrodes  # number of frequency bands considered in the feature matrix
    n_samples = feature_matrix.shape[0]  # number of samples to consider in the feature matrix.

    # Checking the dimension of the feature matrix
    if feature_matrix.shape[1] % n_electrodes != 0:
        print( 'The combination feature matrix - electrodes locations is not working.' )
    assert feature_matrix.shape[1] % n_electrodes == 0
    new_feat = []

    # Reshape a novel feature matrix with a list of array with shape [n_samples x n_electrodes] for each frequency band
    for bands in range( n_bands ):
        new_feat.append( feature_matrix[:, bands * n_electrodes: (bands + 1) * n_electrodes] )

    # Creation of a meshgrid data interpolation
    #   Creation of an empty grid
    grid_x, grid_y = np.mgrid[
                     np.min( electrodes_loc[:, 0] ): np.max( electrodes_loc[:, 0] ): n_gridpoints * 1j,  # along x_axis
                     np.min( electrodes_loc[:, 1] ): np.max( electrodes_loc[:, 1] ): n_gridpoints * 1j  # along y_axis
                     ]

    interpolation_img = []
    #   Interpolation
    #       Creation of the empty interpolated feature matrix
    for bands in range( n_bands ):
        interpolation_img.append( np.zeros( [n_samples, n_gridpoints, n_gridpoints] ) )
    #   Interpolation between the points
    # print('Signals interpolations.')
    for sample in tqdm( range( n_samples ) ):
        for bands in range( n_bands ):
            interpolation_img[bands][sample, :, :] = griddata( electrodes_loc, new_feat[bands][sample, :],
                                                               (grid_x, grid_y), method='cubic', fill_value=np.nan )
    #   Normalization - replacing the nan values by interpolation
    for bands in range( n_bands ):
        interpolation_img[bands][~np.isnan( interpolation_img[bands] )] = scale(
            interpolation_img[bands][~np.isnan( interpolation_img[bands] )] )
        interpolation_img[bands] = np.nan_to_num( interpolation_img[bands] )
    return np.swapaxes( np.asarray( interpolation_img ), 0, 1 )  # swap axes to have [samples, colors, W, H]

sig = np.load('sig.npy')

info = ['Raw EEG Signal', 'Delta Rythm', 'Theta Rythm', 'Alpha Rythm',
        'Beta Rythm', 'Gamma Rythm']

fig, ax = plt.subplots(6, figsize=(6,10))
fig.tight_layout(h_pad=3)
for i in range(6):
  ax[i].plot(sig[i])
  ax[i].set_title(info[i])
  ax[i].grid()

# upload the datsets
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
print('The dimension of the training set is ', X_train.shape)
print('The dimension of the training labels is ', y_train.shape)

X_test = np.load('X_test.npy')
print('The dimension of the testing set is ', X_test.shape)


# Display of 3D position of each electrodes
channel_information = np.load('ChanInfo.npy', allow_pickle=True).all()
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(channel_information['position'][:, 0], channel_information['position'][:, 1], channel_information['position'][:, 2])
plt.suptitle("Electrodes Location in 3D")
plt.show()

# Image Creation
print('Image Creation')
Electrodes_position = channel_information['position']
X_image_train = band_image(X_train, Electrodes_position)
X_image_test = band_image(X_test, Electrodes_position)

#command specific to keras (need to have n_channels x Height x Width as shape)
#implementation made with pytorch wich is a clearly better lib ;)
X_image_train = X_image_train.swapaxes(1,-1)
X_image_test = X_image_test.swapaxes(1,-1)

plt.imshow(X_image_train[584, :, :, 0])
plt.suptitle('EEG under image from corresponding to a '+str(y_train[584]))
plt.show()
