"""
Exploring ammonia subcubes
"""
import numpy as np
import matplotlib.pylab as plt
from multicube.subcube import SubCube, SubCubeStack
from multicube.astro_toolbox import (make_test_cube, get_ncores,
                                     tinker_ring_parspace)
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
import pyspeckit
from astropy.io import fits
import astropy.units as u

xy_shape = (10, 10)
fittype = 'nh3_restricted_tex'
fitfunc = pyspeckit.spectrum.models.ammonia.ammonia_model_restricted_tex

model_kwargs = {'line_names': ['oneone', 'twotwo']}
npars, npeaks = 6, 1

# generating a dummy (gaussian) FITS file
make_test_cube(
    (600, ) + xy_shape, outfile='foo.fits', sigma=(10, 5), writeSN=True)

def gauss_to_ammoniaJK(xy_pars, lfreq, fname='foo', **model_kwargs):
    """
    Take a cube with a synthetic Gaussian "clump"
    and replace the spectra with an ammonia one.
    """
    spc = SubCube(fname + '.fits')
    spc.specfit.Registry.add_fitter(fittype, fitfunc(**model_kwargs), npars)
    spc.update_model(fittype)
    spc.xarr.refX = lfreq
    # replacing the gaussian spectra with ammonia ones
    for (y, x) in np.ndindex(xy_pars.shape[1:]):
        spc.cube[:, y, x] = spc.specfit.get_full_model(pars=xy_pars[:, y, x])
    # adding noise to the nh3 cube (we lost it in the previous step)
    spc.cube += fits.getdata(fname + '-noise.fits')
    return spc

truepars = [12, 4, 15, 0.3, -25, 0.5, 0]
xy_pars = tinker_ring_parspace(truepars, xy_shape, [0, 2], [3, 1])
cubelst = [
    gauss_to_ammoniaJK(xy_pars, freq_dict[line] * u.Hz, **model_kwargs)
    for line in ['oneone', 'twotwo']
]

# creating a SubCubeStack instance from a list of SubCubes
cubes = SubCubeStack(cubelst)
cubes.update_model(fittype)
cubes.xarr.refX = freq_dict['oneone'] * u.Hz
cubes.xarr.velocity_convention = 'radio'
#cubes.xarr.convert_to_unit('km/s')

# setting up the grid of guesses and finding the one that matches best
minpars = [5, 3, 10.0, 0.1, -30, 0.5, 0]
maxpars = [25, 7, 20.0, 1.0, -20, 0.5, 10]
fixed = [False, False, False, False, False, True, False]
finesse = [5, 3, 5, 4, 4, 1, 1]
cubes.make_guess_grid(minpars, maxpars, finesse, fixed=fixed)
cubes.generate_model(multicore=get_ncores())
cubes.best_guess()

rmsmap = cubes.slice(-37, -27, unit='km/s').cube.std(axis=0)
# fitting the cube with best guesses for each pixel
cubes.fiteach(
    fittype=cubes.fittype,
    guesses=cubes.best_guesses,
    multicore=get_ncores(),
    errmap=rmsmap,
    **cubes.fiteach_args)

# plot ammonia gas temperature
cubes.show_fit_param(0, cmap='viridis')

plt.ion()
plt.show()
