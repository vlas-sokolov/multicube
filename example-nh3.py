"""
Exploring ammonia subcubes
"""
import numpy as np
import matplotlib.pylab as plt
import pyspeckit
from subcube import SubCube
from astro_toolbox import make_test_cube
from astro_toolbox import get_ncores
import astropy.units as u
from astropy.utils.console import ProgressBar

# generating a dummy FITS file
make_test_cube((600,10,10), outfile='foo.fits', sigma=(10,5))
sc = SubCube('foo.fits')
sc.update_model('ammonia')
sc.xarr.refX = 23.6944955*u.GHz

spec_pars = [10,3.8,15,1.3,-25,0.5]
base_spec = sc.specfit.get_full_model(pars = spec_pars)
sc.cube[:,:,:] = base_spec[:,None,None] * sc.cube.max(axis=0)
sc.cube += (np.random.random(sc.cube.shape)-.5)*np.median(sc.cube.std(axis=0))

minpars = [5 , 3,  8.0, 0.1, -40, 0.5]
maxpars = [25, 7, 22.0,  2., -10, 0.5]
fixed   = [False]*6; fixed[-1]=True
finesse = [ 2, 2,    2,   3,  10,   1]
sc.make_guess_grid(minpars, maxpars, finesse, fixed=fixed)
sc.generate_model()
sc.best_guess()
rmsmap = np.ones(shape=sc.cube.shape[1:]) * sc.header['RMSLVL']
sc.fiteach(fittype   = sc.fittype,
           guesses   = sc.best_guesses,
           multicore = get_ncores(),
           errmap    = rmsmap,
           **sc.fiteach_args)

sc.show_fit_param(3, cmap='viridis')
clb = sc.mapplot.FITSFigure.colorbar
clb.set_axis_label_text(sc.xarr.unit.to_string('latex_inline'))

sc.get_chi_squared(sigma=sc.header['RMSLVL'])
sc.chi_squared_stats()
sc.mark_bad_fits()

plt.show()
