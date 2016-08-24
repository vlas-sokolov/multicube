"""
A working example script: 
* generates a spectral FITS cube
* makes a guess grid based on given parameter ranges
* makes best guess calculatons and goodness-of-fit estimates
"""
import numpy as np
import matplotlib.pylab as plt
import pyspeckit
from subcube import SubCube
from astro_toolbox import make_test_cube
from astro_toolbox import get_ncores

# generating a dummy FITS file
make_test_cube((300,10,10), outfile='foo.fits', sigma=(10,5))
sc = SubCube('foo.fits')

# TODO: move this to astro_toolbox.py
#       as a general synthetic cube generator routine
# let's tinker with the cube a bit!
# this will introduce a radial velocity gradient:
def tinker_ppv(arr):
    scale_roll = 15
    rel_shift  = 30
    rel_str    = 5
    shifted_component = np.roll(arr, rel_shift) / rel_str
    for y,x in np.ndindex(arr.shape[1:]):
        roll  = np.sqrt((x-5)**2 + (y-5)**2) * scale_roll
        arr[:,y,x] = np.roll(arr[:,y,x], int(roll))
    return arr + shifted_component
sc.cube = tinker_ppv(sc.cube)

sc.update_model('gaussian')

main_comp = [[0.1, sc.xarr.min().value, 0.1], # minpars
             [2.0, sc.xarr.max().value, 2.0], # maxpars
             [10 ,                  10,  10], # finesse
             [False]*3, # fixed
             [True]*3,  # limitedmin
             [True]*3]  # limitedmax

sidekick = [[0.1, -10, 0.1], # minpars
            [0.5, -10, 0.5], # maxpars
            [  3,   1,   3], # finesse
            [False]*3, # fixed
            [True]*3,  # limitedmin
            [True]*3]  # limitedmax

total_zero = [[0.0, 5, 1.0], # minpars
              [0.0, 5, 1.0], # maxpars
              [  1, 1,   1], # finesse
              [True]*3, # fixed
              [True]*3,  # limitedmin
              [True]*3]  # limitedmax

unpack = lambda a,b: [i+j for i,j in zip(a,b)]

# Estimating SNR
sc.get_snr_map()

# Making a guess grid based on parameter permutations
sc.make_guess_grid(*unpack(main_comp, total_zero))
sc.expand_guess_grid(*unpack(main_comp, sidekick))

# Generating spectral models for all guesses
sc.generate_model()
# Calculating the best guess on the grid
sc.best_guess()

sc.fiteach(fittype   = sc.fittype,
           guesses   = sc.best_guesses,
           multicore = 1,#get_ncores(),
           errmap    = sc._rms_map,
           **sc.best_fitargs)

# computing chi^2 statistics to judge the goodness of fit:
sc.get_chi_squared(sigma=sc.header['RMSLVL'], refresh=True)
sc.chi_squared_stats()

sc.show_fit_param(1, cmap='coolwarm')
clb = sc.mapplot.FITSFigure.colorbar
clb.set_axis_label_text(sc.xarr.unit.to_string('latex_inline'))

# and overlay the pixels that didn't converge properly:
sc.mark_bad_fits(cut = 1e-40) # voila! all the pixels are fit!
plt.show()
