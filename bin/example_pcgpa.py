import sys; sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import pyfrog
import pyfrog.gui
"""
Example file showing the retrieval of an synthetic frog trace using the priciple component generalized projection algorithm.

"""
def main():
    # frequency axis
    f_width = 200e12
    f_c = 800e12
    Nf = 2**8
    f = np.linspace( -f_width/2, f_width/2, Nf )
    f = np.fft.fftshift( f )

    # get the corresponding delay axis from given frequency axis
    tau = pyfrog.tau_from_frequency( f ) 

    # artificial spectrum
    f_FWHM = 10e12 # pulse's spectrum FWHM 
    spectrum = np.exp( - f**2/( 2 * ( f_FWHM/2.35482 )**2 ) )

    # some exemplary phase
    phase = np.zeros( spectrum.shape )
    phase += ( 2 * np.pi * f )**2/2 * 1000e-30
    phase += ( 2 * np.pi * f )**3/6 * 10000e-45
    phase += ( 2 * np.pi * f )**4/24 * 100000e-60

    # generate the trace
    trace = pyfrog.generate_trace( spectrum, phase )

    # retrieve spectrum as an initial guess
    ret_spectrum = pyfrog.retrieval.rana_spectrum( trace )

    # retrieve the complex electric field
    E = pyfrog.retrieve_pcgpa( trace , spectrum=ret_spectrum, phase=np.zeros(Nf), g_limit=1e-6, max_iter=100000 )

    # plotting
    pyfrog.gui.simple_frog_result_plot( f, f_c, tau, E, trace, plotstyle='wavelength' )
    
    plt.show()
    
if __name__ =='__main__':
    main()
