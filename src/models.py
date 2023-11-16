import numpy as np
import toml

from vip_hci.var import fit_2dgaussian
from src.hci import get_positions, cut_patch
from src.plot import plot_frame

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
            print(init_conf['sep']['values'])
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
        dx = float(fit.fwhm_x.iloc[0])
        dy = float(fit.fwhm_y.iloc[0])

        positions[i][0] = posx +   
        positions[i][1] = 

    return positions