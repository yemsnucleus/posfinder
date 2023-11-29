import tensorflow as tf
import numpy as np
import imutils
import toml
import cv2

from imutils import contours
from skimage import measure

from src.hci import get_positions, cut_patch
from vip_hci.var import fit_2dgaussian
from scipy.optimize import curve_fit
from src.plot import plot_frame

@tf.function
def gauss_funtion(params, mean, scale, amplitude):
    f = tf.exp(-((params[:, 0]-mean[0])**2/(2*scale[0]**2) + (params[:, 1]-mean[1])**2/(2*scale[1]**2)))
    return amplitude*f

def gauss_tf_model(lambda_frame, cropsize=30, init_pos=None):
    
    learning_rate = 1e-1
    n_iters = 500
    if init_pos is not None:
        with open(init_pos, 'r') as f:
            init_conf = toml.load(f)

    frame_size = lambda_frame.shape
    init_scale = 2.
    positions = np.zeros([frame_size[0], 2])
    for i, frame in enumerate(lambda_frame):
        posx, posy = get_positions(lambda_frame, 
                                   init_conf['sep']['values'][i], 
                                   init_conf['theta']['values'][i])
        planet_frame = cut_patch(frame, x=posx, y=posy, cropsize=cropsize)  

        tpsf      = tf.convert_to_tensor(planet_frame)
        indices   = tf.cast(tf.where(tpsf), tf.float64)
        shp_tpsf  = tf.shape(tpsf)
        flat_tpsf = tf.reshape(tpsf, [-1])
        amplitude  = tf.cast(tf.reduce_max(flat_tpsf), tf.float64)
        init_x     = tf.cast(shp_tpsf[-1]//2, tf.float64)
        init_y     = tf.cast(shp_tpsf[-1]//2, tf.float64)
        init_scale = tf.cast(init_scale, tf.float64)
        
        mean       = tf.Variable([init_x, init_y])
        scale      = tf.Variable([init_scale, init_scale])
    
        optimizer = tf.optimizers.Adam(learning_rate)

        losses = []
        for it in range(n_iters):
            with tf.GradientTape() as tape:
                y_pred = gauss_funtion(indices, mean, scale, amplitude)
                loss = tf.keras.metrics.mean_squared_error(flat_tpsf, y_pred)
                losses.append(loss)
                # Compute gradients
                trainable_vars = [mean, scale]
                gradients = tape.gradient(loss, trainable_vars)
                # Update weights
                optimizer.apply_gradients(zip(gradients,trainable_vars))
        

        positions[i][0] =  (posx - cropsize/2) + mean[0].numpy()
        positions[i][1] =  (posy - cropsize/2) + mean[1].numpy()

    return positions

def gaussian_model(lambda_frame, cropsize=30, init_pos=None):
    """
    For each wavelenght fit a Gaussian and calculate its center.
    The frame is supposed to be the median frame from the cube
    Args:
        lambda_frame (numpy array): A cube containing the frames for each wavelenght (n_wavelenght x W x H).
        cropsize (int, optional): The subset cut where the Gaussian is optimized.
    """

    if init_pos is not None:
        with open(init_pos, 'r') as f:
            init_conf = toml.load(f)

    frame_size = lambda_frame.shape
    positions = np.zeros([frame_size[0], 2])
    for i, frame in enumerate(lambda_frame):
        posx, posy = get_positions(lambda_frame, 
                                   init_conf['sep']['values'][i], 
                                   init_conf['theta']['values'][i])


        planet_frame = cut_patch(frame, x=posx, y=posy, cropsize=cropsize)    

        fit = fit_2dgaussian(planet_frame, 
                             crop=False, 
                             debug=False, 
                             full_output=True)

        dx = float(fit.centroid_x.iloc[0])
        dy = float(fit.centroid_y.iloc[0])

        # plot_frame([planet_frame], pos=[[dx, dy]])

        positions[i][0] =  (posx - cropsize/2) + dx
        positions[i][1] =  (posy - cropsize/2) + dy

    return positions

def brightest_point(lambda_frame, cropsize=30, init_pos=None):
    """
    For each wavelenght fit a Gaussian and calculate its center.
    The frame is supposed to be the median frame from the cube
    Args:
        lambda_frame (numpy array): A cube containing the frames for each wavelenght (n_wavelenght x W x H).
        cropsize (int, optional): The subset cut where the Gaussian is optimized.
    """

    if init_pos is not None:
        with open(init_pos, 'r') as f:
            init_conf = toml.load(f)
            print(init_conf['sep']['values'])

    frame_size = lambda_frame.shape
    positions = np.zeros([frame_size[0], 2], dtype='float64')
    for i, frame in enumerate(lambda_frame):
        posx, posy = get_positions(lambda_frame, 
                                   init_conf['sep']['values'][i], 
                                   init_conf['theta']['values'][i])


        planet_frame = cut_patch(frame, x=posx, y=posy, cropsize=cropsize)    

        gray = cv2.cvtColor(np.tile(planet_frame[...,None], [1,1,3]), cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, np.zeros_like(gray), 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, 30, blurred.max(), cv2.THRESH_BINARY)[1]
        thresh = np.array(thresh, np.uint8)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]

        for (_, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)

        positions[i][0] =  (posx - cropsize/2) + cX 
        positions[i][1] =  (posy - cropsize/2) + cY 

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(thresh)
        # plt.scatter(cX, cY, marker='.', color='r')
        # plt.show()

    return positions