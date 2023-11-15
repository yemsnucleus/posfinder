import numpy as np
import os

from vip_hci.preproc.derotation import cube_derotate
from astropy.io import fits

def get_positions(cube, rad_distance, theta, q=None):
    cube_dim = cube[0].shape
    center_cube_x = cube_dim[-2]/2
    center_cube_y = cube_dim[-1]/2
    if q is not None:
        posy = rad_distance * np.sin(np.deg2rad(theta)-np.deg2rad(q)) + center_cube_y
        posx = rad_distance * np.cos(np.deg2rad(theta)-np.deg2rad(q)) + center_cube_x
    else:
        posy = rad_distance * np.sin(np.deg2rad(theta)) + center_cube_y
        posx = rad_distance * np.cos(np.deg2rad(theta)) + center_cube_x

    return posx, posy

def parse_filter_code(code):
    """ Read the code and write the filters separated.
    
    Args:
        code (str): Code string from X
    
    Returns:
        TYPE: Description
    """
    ftype = code.split('_')[0]
    filters = code.split('_')[1]
    if ftype == 'DB':
        #DUAL BAND
        filter_letter = filters[0]
        filters = [filter_letter+'_'+x for x in filters[1:]]
    return filters
    
def load_fits(path):
    """
    Load cube, PSF, and parallactic angles.

    Args:
        path (str): Directory where the data is located.

    Returns:
        dictionary: A dictionary containing the cube, PSF, parallactic angles, and filters.
    """
    cube, header = fits.getdata(os.path.join(path, 'center_im.fits'), header=True)
    psf  = fits.getdata(os.path.join(path, 'median_unsat.fits'))
    ang  = fits.getdata(os.path.join(path, 'rotnth.fits'))
    filter_name = header['HIERARCH ESO INS COMB IFLT']
    filters = parse_filter_code(filter_name)
    return {
        'cube': cube,
        'psf': psf,
        'parallactic':-ang,
        'filters': filters
    }

def collapse_to_median(lambda_cube, q=None, save=None, overwrite=False, imlib='opencv', interpolation='nearneig'):
    """ Reduce cube frames to its median
    See https://vip.readthedocs.io/en/latest/vip_hci.preproc.html#vip_hci.preproc.derotation.frame_rotate
    for more information about imlib and interpolation parameters.
    Args:
        cube (numpy.array): A numpy array containing frames (Wavelength x n_frames x W x H)
        q (None, optional): Parallactic angles
        imlib (str, optional): Library used for image transformations
        interpolation (str, optional): Interpolation library

    Returns:
        median_frame: A numpy array of the median (W x H)s
    """
    if save is not None and not overwrite:
        outfile = os.path.join(save, 'median_frame.npy')
        if os.path.isfile(outfile):
            median_frame = np.load(outfile)
            return median_frame

    cube_size = lambda_cube.shape
    median_frame = np.zeros([cube_size[0], cube_size[-2], cube_size[-1]], dtype='float32')
    for i, cube in enumerate(lambda_cube):
        print('[INFO] Processing wavelenght {}'.format(i))
        if q is not None:
            cube = cube_derotate(cube, 
                angle_list=q, 
                imlib=imlib, 
                interpolation=interpolation)
        median_frame[i] = np.median(cube, 0)

    if save is not None and overwrite:
        print('[INFO] Saving median frame')
        outfile = os.path.join(save, 'median_frame.npy')
        np.save(outfile, median_frame)

    return median_frame