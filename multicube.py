"""A skeleton file for a class dealing with different spectral models"""
import numpy as np
import pyspeckit
from subcube import SubCube

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
        Add a new model and a SubCube for it through Cube()
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
