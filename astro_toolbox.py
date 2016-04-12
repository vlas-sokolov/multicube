import numpy as np
from astropy.io import fits
import os

# TODO: rewrite this to have multiple components generated here.
#       Having a bundle of test filaments would be very nice.

def make_test_cube(shape=(30,9,9), outfile='test.fits', 
                   sigma=None, seed=0, writeSN=False   ):
    """
    Generates a simple gaussian cube with noise of
    given shape and writes it as a fits file.
    """
    from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
    if sigma is None:
        sigma1d, sigma2d = shape[0]/10., np.mean(shape[1:])/5.
    else:
        sigma1d, sigma2d = sigma

    gauss1d = Gaussian1DKernel(stddev=sigma1d, x_size=shape[0])
    gauss2d = Gaussian2DKernel(stddev=sigma2d, x_size=shape[1], 
                                               y_size=shape[2])
    signal_cube = gauss1d.array[:,None,None] * gauss2d.array
    signal_cube=signal_cube/signal_cube.max()
    # adding noise:
    np.random.seed(seed)
    noise_cube = (np.random.random(signal_cube.shape)-.5)* \
                        np.median(signal_cube.std(axis=0))
    test_cube = signal_cube+noise_cube
    true_rms = noise_cube.std()
    
    # making a simple header for the test cube:
    test_hdu = fits.PrimaryHDU(test_cube)
    # the strange cdelt values are a workaround 
    # for what seems to be a bug in wcslib:
    # https://github.com/astropy/astropy/issues/4555
    cdelt1, cdelt2, cdelt3 = -(4e-3+1e-8), 4e-3+1e-8, -0.1
    keylist = {'CTYPE1': 'RA---GLS', 'CTYPE2': 'DEC--GLS', 'CTYPE3': 'VRAD',
               'CDELT1': cdelt1, 'CDELT2': cdelt2, 'CDELT3': cdelt3,
               'CRVAL1': 0, 'CRVAL2': 0, 'CRVAL3': 5, 
               'CRPIX1': 9, 'CRPIX2': 0, 'CRPIX3': 5,
               'CUNIT1': 'deg', 'CUNIT2': 'deg', 'CUNIT3': 'km s-1',
               'BUNIT' : 'K', 'EQUINOX': 2000.0}
    # write out some values used to generate the cube:
    keylist['SIGMA' ] = abs(sigma1d*cdelt3), 'in units of CUNIT3'
    keylist['RMSLVL'] = true_rms
    keylist['SEED'  ] = seed

    test_header = fits.Header()
    test_header.update(keylist)
    test_hdu = fits.PrimaryHDU(data=test_cube, header=test_header)
    test_hdu.writeto(outfile, clobber=True, checksum=True)

    if writeSN:
        signal_hdu = fits.PrimaryHDU(data=signal_cube, header=test_header)
        noise_hdu  = fits.PrimaryHDU(data=noise_cube , header=test_header)
        signame, noiname = [outfile.split('.fits')[0]+'-'+i+'.fits' 
                                            for i in ['signal','noise']]
        signal_hdu.writeto(signame, clobber=True, checksum=True)
        noise_hdu.writeto( noiname, clobber=True, checksum=True)

def download_test_cube(outfile='test.fits'):
    """
    Downloads a sample fits file from Dropbox (325kB).
    """
    from astropy.utils.data import download_file
    test_cube_url = 'https://db.tt/i0jWA7DU'
    tmp_path = download_file(test_cube_url)
    try:
        os.rename(tmp_path, outfile)
    except OSError:
        # os.rename doesn't like cross-device links
        import shutil
        shutil.move(tmp_path, outfile)

def get_ncores():
    """
    Try to get the number of cpu cores
    """
    try:
        import multiprocessing
        ncores = multiprocessing.cpu_count()
    except ImportError:
        ncores = 1

    return ncores
