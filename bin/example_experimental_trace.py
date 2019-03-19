import sys; sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
import scipy.interpolate

import pyfrog
import pyfrog.gui
"""
Example file showing the retrieval of an experimental frog trace.

"""

def main():
    # Measured data information
    lambda_c = 518.176784e-9
    dlambda = 0.006491e-9
    dtau = 20e-15
    
    # load trace
    raw_trace = np.loadtxt( '../data/FROGtrace-20fs.dat', skiprows=5 )
    raw_trace /= np.amax( raw_trace )
    raw_trace = raw_trace.transpose()
    raw_trace = np.flipud( raw_trace )

    # grid size of the trace for retrieval
    Nf = 256

    # noise substraction
    raw_trace = pyfrog.full_spectrum_noise_substraction( raw_trace )
    raw_trace = pyfrog.full_spectrum_noise_substraction( raw_trace )
    raw_trace[ raw_trace < 2e-4 ] = 0
    raw_trace = pyfrog.pixel_cleanup_noise_substraction( raw_trace )
    raw_trace = pyfrog.noise.pixel_cleanup( raw_trace )
    raw_trace = pyfrog.grid.center_trace( raw_trace )
    raw_trace = pyfrog.grid.correct_symmetry( raw_trace )
    
    # generate a trace on a grid that fulfills the requirements of the pcgpa algorithm
    trace, f, f_c, tau = pyfrog.grid.raw_trace_to_valid_frog_grid( raw_trace, lambda_c, dlambda, dtau, grid_size=Nf )
    trace = pyfrog.noise.pixel_cleanup( trace )

    # retrieve the fundamental spectrum to have a good initial guess
    ret_spectrum = pyfrog.retrieval.rana_spectrum( trace )

    # retrieve the complex electric field. The various algorithms in pyfrog.retrieval might be used
    #E = pyfrog.retrieve_vanilla( trace, spectrum=ret_spectrum, phase=np.zeros(Nf), max_iter=1e2 )
    E = pyfrog.retrieve_pcgpa( trace , spectrum=ret_spectrum, phase=np.zeros(Nf), g_limit=1e-4, max_iter=5e2, stagnation_count=500 )

    #visualization
    pyfrog.gui.simple_frog_result_plot( f, f_c, tau, E, trace, plotstyle='wavelength' )
    
if __name__ =='__main__':
    main()
