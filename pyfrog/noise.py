import numpy as np
from scipy.ndimage import uniform_filter # pixel cleanup


def lowest_pixel( trace ):
    """
    lowest_pixel( trace ) substracts the lowest pixel from the trace.

    It might be the same as the "Lowest Pixel" method of the original FROG software:
        in those cases where the FROG data is nearly truncated, a Full Spectrum subtraction may yield anomalous results. In these cases, the Lowest Pixel subtraction will look for the pixel with the lowest value (least amount of signal) and subtract that value from every pixel.  This may be useful for removing dark current counts, for example.

    arguments:
        trace : array containing the FROG trace

    returns:
        trace : same trace, where the lowest value in the trace is substracted
    """
    trace -= np.amax( trace )
    return trace

def edge_pixel( trace ):
    """
    edge_pixel( trace ) substracts the mean value of the 4 edges of the trace.

    It might be the same as the "Edge" method of the original FROG software:
    this mode takes an average value of every data value within two pixels of the edge of the data set on all four sides. Again, the real FROG signal should be zero here, so that only noise contributes to the average.  This average value is then subtracted from every pixel in the data set.

    arguments:
        trace : array containing the FROG trace

    returns:
        trace : corrected trace
    """
    mean = 0.25 * ( np.mean( trace[0,:] ) + np.mean( trace[-1,:] ) + np.mean( trace[:,0] ) + np.mean( trace[:,-1] ) )
    return trace - mean

def full_spectrum( trace, average=2 ):
    """
    full_spectrum( trace ) substracts the spectrally dependent mean value of the 2 edges of the trace.

    It might be the same as the "Full Spectrum" method of the original FROG software:
    this method takes the average of the spectra at the two smallest and two largest delays, and uses this average spectrum to subtract from the data at every time delay.  The rationale behind this is that when stray light is entering the spectrometer used to make the FROG measurement, its spectrum will be dispersed across the detector independent of time delay.  The smallest and largest time delays should not have any legitimate FROG data, so that the only signal present there is the stray light.  Subtracting this spectrum from all delay values should increase the accuracy of the data.

    arguments:
        trace : array containing the FROG trace

    returns:
        trace : corrected trace
    """
    Nf, Ntau = trace.shape
    edges = np.concatenate( (trace[:,:average], trace[:,-average:]), axis=1 )
    edges = np.mean( edges, axis=1 )
    return trace - np.reshape( edges, (Nf, 1 ) )

def fourier_low_pass( trace, HWHM=0.5 ):
    """
    fourier_low_pass( trace, HWHM=0.5 ) applies a fourier filter removing high frequency components from the trace.

    It might be the same as the "Fourier Low Pass" method of the original FROG software:
    in this case a low-pass Fourier filter is used.  First, the data are put onto a square M X M grid with a number of elements M that is a power of two (the smallest value of M that will hold all the data is used).  The data are then Fourier transformed, and are multiplied by a circular Gaussian filter.  The half width at half maximum (HWHM) of the filter is input by the user; a value of 1.0 means a HWHM of M/2.

    arguments:
        trace : array containing the FROG trace
        HWHM (optional) : HWHM in units of window size.

    returns:
        trace : corrected trace
    """
    Nf, Ntau = trace.shape
    _trace = np.fft.fftshift( np.fft.fft2( trace ) )
    f = np.arange( Nf )
    f = f - np.mean( f )
    tau = np.arange( Ntau )
    tau = tau - np.mean( tau )
    tt, ff = np.meshgrid( tau, f )
    c1 =  HWHM / 2.35482 * 2 * Nf
    c2 = HWHM / 2.35482 *  2 * Ntau
    _trace *= np.exp( -ff**2/(2*c1**2) ) * np.exp( -tt**2/(2*c2**2) )
    return abs( np.fft.ifft2( np.fft.fftshift( _trace ) ) )
    

def pixel_cleanup( trace, nonzero_neighbours=4 ):
    """
    pixel_cleanup( trace, nonzero_neighbours ) set the pixels that have less than "nonzero_neighbours" neighbours to zero. It is used to remove "lonely" pixels with nonzero signal.

        It might be the same as the "Cleanup Pixel" method of the original FROG software:
    this remarkable filter will hunt down and remove stray pixels.  This will help remove noise in places that are hard to process with the other filters.

    arguments:
        trace : array containing the FROG trace
        nonzero_neighbours (optional) : Nonzero neighbours required (default 4).

    returns:
        trace : corrected trace
    """
    nonzeros = np.zeros( trace.shape )
    nonzeros[ np.nonzero(trace) ] = 1
    neighbours = uniform_filter( nonzeros, mode='constant' )
    trace[ neighbours <= (9-nonzero_neighbours)/9 ] = 0
    return trace
