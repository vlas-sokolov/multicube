"""
A small OOP-oriented wrapper for pyspeckit,
for extended flexibility with input guesses,
model selection, and multiple component fits.
"""
import numpy as np
import astropy.units as u
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

    def info(self):
        print "Shapes of the arrays:\n" \
              "\t--> Data cube:\t{}\n".format(self.cube.shape) + \
              "\t--> Guess grid:\t{}\n".format(self.guess_grid.shape) + \
              "\t--> Model grid:\t{}\n".format(self.model_grid.shape) + \
              "\t--> SNR map:\t{}\n".format(self.snr_map.shape)

    def update_model(self, fit_type='gaussian'):
        """
        Tie a model to a SubCube. Didn't test it
        on anything but gaussian fitter so far.
        Yeap, I don't understand how models work.
        """
        try:
            allowed_fitters = self.specfit.Registry.multifitters
            self.specfit.fitter = allowed_fitters[fit_type]
        except KeyError:
            # TODO: get other models through add_fitter Registry method!
            raise ValueError('Unsupported fit type: %s\n'
                             'Choose one from %s' 
                             % (fit_type, allowed_fitters.keys()))
        self.specfit.fittype = fit_type
        self.fittype = fit_type

    def make_guess_grid(self, minpars, maxpars, finesse, 
            fixed=None, limitedmin=None, limitedmax=None):
        """
        Given parameter ranges and a finesse parameter, generate a grid of 
        guesses in a paramener space to be iterated upon in self.best_guess
        Maybe if parlimits arg is None we can look into parinfo?

        Parameters
        ----------
        minpars : an interable contatining minimal parameter values

        maxpars : an interable contatining maximal parameter values

        finesse : an integer setting the size of cells between minimal
                  and maximal values in the resulting guess grid

        fixed : an iterable of booleans setting whether or not to fix the
                fitting parameters. Will be passed to Cube.fiteach, defaults
                to an array of False-s.

        limitedmin : an iterable of booleans controlling if the fit fixed
                     the minimal boundary of from minpars.

        limitedmax : an iterable of booleans controlling if the fit fixed
                     the maximal boundary of from maxpars.

        Returns
        -------
        guess_grid : a grid of guesses to use for SubCube.generate_model

        In addition, it saves a number of variables under self as a dictionary
        passed later to Cube.fiteach as additional arguments, with keywords:
        ['fixed', 'limitedmin', 'limitedmax', 'minpars', 'maxpars']
        """
        minpars, maxpars = np.asarray([minpars, maxpars])
        truths, falses = np.ones(minpars.shape, dtype=bool), \
                         np.zeros(minpars.shape, dtype=bool)
        
        fixed = falses if fixed is None else fixed
        limitedmin = truths if limitedmin is None else limitedmin
        limitedmax = truths if limitedmax is None else limitedmax
        self.fiteach_args = {'fixed'     : fixed,
                             'limitedmin': limitedmin,
                             'limitedmax': limitedmax,
                             'minpars'   : minpars,
                             'maxpars'   : maxpars    }
 
        # TODO: make sure you return the same shape!
        input_shape = minpars.shape

        minpars, maxpars = minpars.reshape(-1,1), maxpars.reshape(-1,1)
        generator = np.linspace(0,1,finesse)
        guess_grid = minpars + (maxpars - minpars) * generator
        guess_grid = guess_grid.reshape((list(input_shape)+[finesse]))
        self.guess_grid = guess_grid
        return guess_grid

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
                     If not set, SubCube.guess_grid is used.

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
            try:
                guess_grid = self.guess_grid
            except AttributeError:
                raise RuntimeError("Can't find the guess grid to use,")

        # TODO: this if/elif Christmas tree is ugly, 
        #       there should be a smarter way to do this
        # TODO: yup, of course! use np.ndenumerate
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
        # + Second milestone : implement the snr mask
        # - Third milestone  : xy_list functionality, pixel coordinates only
        # - Fourth milestone : extend to all possible model_grid shapes
        #
        if model_grid.shape!=self.cube.shape:
            raise NotImplementedError("Sorry, still working on it!")
        else:
            # commence the invasion!
            self.residual = (self.cube - model_grid).std(axis=0)

            # NOTE: np.array > None returns an all-True mask
            snr_mask = self.get_snr_map() > sn_cut
            try:
                best = self.residual[snr_mask].min()
            except ValueError:
                raise ValueError("Oh gee, something broke. "
                                 "Was SNR cutoff too high?" )
            # TODO: catch edge cases, e.g. no valid points found
            if np.isnan(self.residual).all():
                # FIXME: throw warning for all-NaN case
                raise ValueError("All NaN residual encountered.")
            vmin, (xmin,ymin) = best, np.where(self.residual==best)
            print "Best guess at %.2f on (%i, %i)" % (vmin, xmin, ymin)
            return vmin, (xmin, ymin)

    def get_slice_mask(self, mask2d):
        """
        In case we ever want to apply a 3d mask to a whole cube.
        """
        mask3d = np.repeat([mask2d],self.xarr.size,axis=0)
        return mask3d

    def get_snr_map(self, signal=None, noise=None, unit='km/s', 
                    signal_mask=None, noise_mask=None          ):
        """
        Calculates S/N ratio for the cube. If no information is given on where
        to look for signal and noise channels, a (more-or-less reasonable) rule
        of thirds is used: the outer thirds of the channel range are used to 
        get the root mean square of the noise, and the max value in the inner 
        third is assumed to be the signal strength.
        
        Parameters
        ----------
        signal : 2xN numpy.array, where N is the total number of signal blocks.
                 Should contain channel numbers in `unit` convention, the first
                 subarray for start of the signal block and the second one for
                 the end of the signal block

        noise : 2xN numpy.array, where N is the total number of noise blocks.
                Same as `signal` otherwise.

        unit : a unit for specifying the channels. Defaults to 'km/s'.
               If set to 'pixel', actualy channel numbers are selected.

        signal_mask : dtype=bool numpy.array of SubCube.xarr size
                      If specified, used as a mask to get channels with signal.
                      Overrules `signal`

        noise_mask : dtype=bool numpy.array of SubCube.xarr size
                     If specified, used as a mask to get channels with noise.
                     Overrules `noise`

        Returns
        -------
        snr_map : numpy.array
                  Also stored under SubCube.snr_map
        """
        # will overwrite this lates if no ranges were actually given
        unit = {'signal': unit, 'noise': unit}

        # get rule of thirds signal and noise if no ranges were given
        default_cut = 0.33
        if signal is None:
            # find signal cuts for the current unit?
            # nah let's just do it in pixels, shall we?
            i_low, i_high = int(round(self.xarr.size *    default_cut )),\
                            int(round(self.xarr.size * (1-default_cut)))
            signal = [[i_low+1], [i_high-1]]
            unit['signal'] = 'pixel'

        if noise is None:
            # find signal cuts for the current unit?
            # nah let's just do it in pixels, shall we?
            i_low, i_high = int(round(self.xarr.size *    default_cut )),\
                            int(round(self.xarr.size * (1-default_cut)))
            noise = [[0, i_high], [i_low, self.xarr.size-1]]
            unit['noise'] = 'pixel'

        # setting xarr masks from high / low indices
        if signal_mask is None:
            signal_mask = self.get_mask(*signal, unit=unit['signal'])
        if noise_mask is None:
            noise_mask = self.get_mask(*noise, unit=unit['noise'])
        self._mask_signal = signal_mask
        self._mask_noise = noise_mask

        # no need to care about units at this point
        snr_map = self.get_signal_map(signal_mask) / \
                             self.get_rms_map(noise_mask)
        self.snr_map = snr_map
        return snr_map

    def get_mask(self, low_indices, high_indices, unit):
        """
        Converts low / high indices arrays into a mask on self.xarr
        """
        mask = np.array([False]*self.xarr.size)
        for low, high in zip(low_indices, high_indices):
            # you know this is a hack right?
            # also, undocumented funcitonality is bad and you should feel bad
            if unit not in ['pix','pixel','pixels','chan','channel','channels']:
                # converting whatever units we're given to pixels
                unit_low, unit_high = low*u.Unit(unit), high*u.Unit(unit)
                try:
                    # FIXME: this is too slow, find a better way!
                    unit_bkp = self.xarr.unit
                    self.xarr.convert_to_unit(unit)
                except u.core.UnitConversionError as err:
                    raise type(err)(str(err) + "\nConsider setting, e.g.:\n"
                            "SubCube.xarr.velocity_convention = 'radio'\n"
                            "and\nSubCube.xarr.refX = line_freq*u.GHz")
                index_low  = self.xarr.x_to_pix(unit_low)
                index_high = self.xarr.x_to_pix(unit_high)
                self.xarr.convert_to_unit(unit_bkp)
            else: 
                try:
                    index_low, index_high = int(low.value ),\
                                            int(high.value)
                except AttributeError:
                    index_low, index_high = int(low), int(high)

            # so this also needs to be sorted if the axis goes in reverse
            index_low, index_high = np.sort([index_low, index_high])

            mask[index_low:index_high] = True

        return mask

    def get_rms_map(self, noise_mask=None):
        """
        Make an rms estimate, will try to find the noise channels in
        the input values or in class instances. If noise mask is not
        not given, defaults to calculating rms of all channels.

        Parameters
        ----------
        noise_mask : dtype=bool numpy.array of SubCube.xarr size
                     If specified, used as a mask to get channels with noise.

        Returns
        -------
        rms_map : numpy.array, also stored under SubCube.rms_map
        """
        if noise_mask is None:
            # will calculate rms of all channels
            # TODO: throw a warning here: this will overestimate the rms!
            noise_mask = np.ones(self.xarr.shape, dtype=bool)
        rms_map = self.cube[noise_mask,:,:].std(axis=0)
        self._rms_map = rms_map
        return rms_map

    def get_signal_map(self, signal_mask=None):
        """
        Make a signal strength estimate. If signal mask is not
        not given, defaults to maximal signal on all channels.

        Parameters
        ----------
        signal_mask : dtype=bool numpy.array of SubCube.xarr size
                      If specified, used as a mask to get channels with signal.

        Returns
        -------
        signal_map : numpy.array, also stored under SubCube.signal_map
        """
        if signal_mask is None:
            # will calculate signal strength as a max of all channels
            # TODO: throw a warning here: this might overestimate the signal!
            signal_mask = np.array(self.xarr.shape, dtype=bool)
        signal_map = self.cube[signal_mask,:,:].max(axis=0)
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

    sc.get_snr_map()

    npars = len(guesses)
    parcube_size = sc.cube.size/sc.shape[0]
    parcube_shape = (npars, sc.cube.shape[1], sc.cube.shape[2])
    parflat = np.hstack(np.repeat([guesses],parcube_size,axis=0))
    sc.guess_grid = parflat.reshape(*parcube_shape,order='F')

    sc.generate_model()
    
    sc.info()
    sc.best_guess()

if __name__ == "__main__":
	main()
