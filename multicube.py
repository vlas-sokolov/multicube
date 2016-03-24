"""
A small OOP-oriented wrapper for pyspeckit,
for extended flexibility with input guesses,
model selection, and multiple component fits.
"""
import numpy as np
import pyspeckit

# TODO: make informative log/on-screen messages
#       about what's being done to the subcubes

class SubCube(pyspeckit.Cube):
    """
    An extention of Cube, tinkered to be an instance of MultiCube, from which 
    it receives references to instances of pyspeckit.Cube that do not depend 
    on a spectral model chosen (so that parent MultiCube doesn't weigh so much)

    Is designed to have methods that operate within a single spectral model.
    """
    def __init__(self, *args, **kwargs):
        super(SubCube, self).__init__(*args, **kwargs)

        # because that UnitConversionError pops up way too often
        if self.xarr.velocity_convention is None:
            self.xarr.velocity_convention = 'radio'
        
        # so I either define some things as `None` 
        # or I'll have to call hasattr or them...
        # TODO: which is a more Pythonic approach?
        # A: probably the hasattr method, see here:  
        # http://programmers.stackexchange.com/questions/
        # 254576/is-it-a-good-practice-to-declare-instance
        # -variables-as-none-in-a-class-in-python
        self.guess_grid = None
        self.model_grid = None

    def update_model(self, fit_type='gaussian'):
        """
        Tie a model to a SubCube. Didn't test it
        on anything but gaussian fitter so far.
        """
        try:
            allowed_fitters = self.specfit.Registry.multifitters
            self.specfit.fitter = allowed_fitters[fit_type]
        except KeyError:
            raise ValueError('Unknown fit type: %s\n'
                             'Choose one from %s' 
                             % (fit_type, allowed_fitters.keys()))
        self.specfit.fittype = fit_type
        self.fittype = fit_type

    def generate_model(self, guess_grid=None):
        """
        Generates a grid of spectral models matching the
        shape of the input guess_grid array. Can take the
        following numpy arrays as an input:

        Parameters
        ----------
        guess_grid : numpy.array
                     A grid of input parameters. 
                     Can be one of the following:
                     1) An (M,)-shaped 1d array of model parameters
                        (see pyspeckit docs for details)
                     2) An (N, M) array to compute N models 
                        for M sets of parameters
                     3) A guess cube of (Y, X, M) size
                     4) An (N, Y, X, M)-shaped array, to 
                        iterate over cubes of guesses.

        WARNING: (3) and (4) aren't implemented yet.

        Returns
        -------
        model_grid : a grid of spectral models, following the 
                     shape of guess_grid. Also saved as an 
                     instance under SubCube.model_grid
        """
        # NOTE: 2016.03.22: currently works for (a) and (c), see below

        # TODO: test for all posible model grid sizes,
        #       and write up proper error handling!
        #       things you should be able receive:
        #        a) a single par-iterable
        #        b) an XY collection of pars
        #        c) an N*par collection of different guesses
        #        d) an N*XY*par mess: should not proceed without
        #           vectorization! it will simply be too slow :/
        if guess_grid is None:
            # TODO: what to do if the guess grid is not in place?
            guess_grid = self.guess_grid

        # TODO: this if/elif Christmas tree is ugly, 
        #       there should be a smarter way to do this
        if len(guess_grid.shape)==4 \
           and self.cube.shape[1] in guess_grid.shape \
           and self.cube.shape[1] in guess_grid.shape:
            # FIXME: this allows for [100,2,4] guess_grid to pass
            #        for an  [xarr.size, 100, 100] cube. Do it right!
            #
            # TODO: implement cube-like guessing grids!

            # this is a good place to re-implement get_modelcube
            # as a method that takes modelcube as a function,
            # or just bait and switch guesses into inherited 
            # Cube.parcube and call get_modelcube . . .

            # Will do the former for now, but it may clash later
            # when I will finish up MultiCube.multiplot()

            raise NotImplementedError('Someone should remind me '
                                      'to write this up. Please?')
        if len(guess_grid.shape)==3 \
           and self.cube.shape[1] in guess_grid.shape \
           and self.cube.shape[1] in guess_grid.shape:
            # FIXME: this allows for [100,2,4] guess_grid to pass
            #        for an [xarr.size, 100, 100] cube. Do it right!
            #
            yy, xx = np.indices(self.cube.shape[1:])
            model_grid = np.zeros_like(self.cube)

            # TODO: vectorize this please please please?
            for x,y in zip(xx.flat,yy.flat):
                model_grid[:,y,x] = \
                       self.specfit.get_full_model(pars=self.guess_grid[:,y,x])

        elif len(guess_grid.shape)==2:
            # set up the modelled spectrum grid
            model_grid = np.empty(shape=(guess_grid.shape[0], 
                                         self.xarr.size      ))
            for i, par in enumerate(guess_grid):
                model_grid[i] = self.specfit.get_full_model(pars=par)
        elif len(guess_grid.shape)==1:
            par = guess_grid
            model_grid = self.specfit.get_full_model(pars=par)
        else:
            raise IndexError('Guess grid size can not be matched'
                             ' to either cube or spectrum size. ')

        self.model_grid = model_grid
        return model_grid

    def best_guess(self, model_grid=None, xy_list=None, sn_cut=None):
        """
        For a grid of intitial guesses, determine the optimal one based 
        on the preliminary residual of the specified spectral model.

        Parameters
        ----------
        model_grid : numpy.array
                     A model grid to choose from.

        use_cube : boolean
                   If true, every xy-slice of a cube will be
                   compared to every model from the model_grid.
                   sn_cut (see below) is still applied.

        xy_list : iterable
                  A collection of positions on the data cube
                  which to check for the lowest residuals.
                  Ignored if use_cube was set to `True`
                  Actually, I guess I should use a mask for this...

        sn_cut : float
                 Ignore items on xy_list if the corresponding
                 spectra have too low signal-to-noise ratios.

        """
        if model_grid is None:
            if self.model_grid is None:
                raise TypeError('sooo the model_grid is empty, '
                                'did you run generate_model()?')
            model_grid = self.model_grid
        # TODO: scale this up later for MultiCube.judge() method
        #       to include the deviance information criterion, DIC
        #       (c.f Kunz et al. 2006 and Sebastian's IMPRS slides)

        # TODO break-down:
        #
        # + First milestone  : make the blind iteration over the spectra,
        #                      have some winner-takes-it-all selection in place
        # - Second milestone : implement the (still broken) snr mask
        # - Third milestone  : xy_list functionality, pixel coordinates only
        # - Fourth milestone : extend to all possible model_grid shapes
        #
        if model_grid.shape!=self.cube.shape:
            raise NotImplementedError
        else:
            # commence the invasion!
            residual = (self.cube - model_grid).std(axis=0)
            if ~np.isnan(residual).any():
                # FIXME: throw warning for all-NaN case
                return
            self.residual = residual
            vmin, (xmin,ymin) = residual.min(), np.where(residual==residual.min())
            print "Best guess at %.2f on (%i, %i)" % (vmin, xmin, ymin)
            return vmin, (xmin, ymin)

    def get_snr_map(self):
        """
        Calculates S/N ratio for the cube.

        Returns
        -------
        snr_map : numpy.array
                  Also stored under SubCube.snr_map
        """
        snr_map = self.get_signal_map(self) / self.get_rms_map(self)
        self._snr_map = snr_map
        return snr_map

    def get_rms_map(self, noise_heads=None, noise_tails=None, unit='km/s'):
        """
        Make an rms estimate, will try to find the noise channels in
        the input values or in class instances. If noise channels are
        not identified, defaults to calculating rms of all channels.
        """
        # TODO: implement sigal masks!
        #       for a gaussian, they can be 
        #       calculated from guess_grids!

        rms_map = self.cube.std(axis=0) # this is so wroooooooong I can't even
        self._rms_map = rms_map
        return rms_map

    def get_signal_map(self, signal_heads=None, 
                       signal_tails=None, unit='km/s'):
        """
        Make a signal strength estimate. Currently just selects maximum 
        value of the all channels, mimicking pyspeckit approach. 
        Will try to improve it soon.
        """
        # TODO: implement sigal masks!
        #       for a gaussian, they can be 
        #       calculated from guess_grids!
        signal_map = self.cube.max(axis=0)
        self._signal_map = signal_map
        return signal_map

class MultiCube:
    def __init__(self, *args):
        """
        A collection of Specfit objects mapped to SubCubes
        by the mapper method. Includes* methods to fit multiple
        guess grids for different models, and the means to
        decide between the results of those fits.

        *In a distant[citation needed] future.

        Input parameters: see ~pyspeckit.Cube
        """

        # parent cube, used for attribute propagation
        self.supercube = pyspeckit.Cube(*args)

        # making a bunch of references to make our life easier
        self.cube = self.SuperCube.cube
        self.xarr = self.SuperCube.xarr
        self.header = self.SuperCube.header

        # FIXME: rewrite mapplot to include the mapper/judge methods!
        #        doesn't work in its current implementation, will need
        #        to rewire different params and errors, for variable
        #        number of parameters across different models
        self.multiplot = self.SuperCube.mapplot

        # MultiCube's own instances:
        self.multigrid = {}
        self.tesseract = {}

    def __repr__(self):
        return ('Parent: MultiCube with TODO models\n'
                'Child: %s' % self.SuperCube.__repr__())

    def spawn(self, model, guesses=None):
        """
        Add a new model and a SubCube for it thorugh Cube()
        The idea is to pass a reference to large data instances 
        of SuperCube to avoid excessive memory usage.

        Not implemented yet.
        """
        self.tesseract[model]=SubCube()
        raise NotImplementedError

    def mapper(model):
        """
        Returns a list of SubCubes for a given model?
        """
        raise NotImplementedError

    def judge_multimodel(subcubes, model, method):
        """
        Decide which model to use.
        First milestone: have one component added only
                         when residual has SNR>3
        Actual goal: proper model selection via DIC.
        """
        raise NotImplementedError

    def multifit(self, multigrid=None):
        """
        Fit the optimized guesses. This should be delegated
        to SubCubes maybe? MultiCube should only call the
        judge function.

        Not really, this approach would allow to juggle all
        the SubCubes defined! In this case, multifit is a
        wrapper for SubCube.fiteach() method. This will do.
        """
        raise NotImplementedError

# NOTE: a working example, generates a grid
#       of spectra form a grid of parameters
#
# TODO: remove this, this isn't one of your km-long scripts!
def main():
    try:
        sc = SubCube('foo.fits')
    except IOError:
        from astro_toolbox import make_test_cube
        make_test_cube((100,10,10), outfile='foo.fits', sigma=(10,5))
        sc = SubCube('foo.fits')

    sc.update_model('gaussian')
    guesses = [0.5, 0.2, 0.8]

    npars = len(guesses)
    parcube_size = sc.cube.size/sc.shape[0]
    parcube_shape = (npars, sc.cube.shape[1], sc.cube.shape[2])
    parflat = np.hstack(np.repeat([guesses],parcube_size,axis=0))
    sc.guess_grid = parflat.reshape(*parcube_shape,order='F')

    sc.generate_model()
    sc.best_guess()
    
    #sc.fiteach(guesses=sc.guess_grid, start_from_pixel=(5,5))
    #sc.mapplot()

if __name__ == "__main__":
	main()
