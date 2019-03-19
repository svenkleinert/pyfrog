import numpy as np
import scipy.interpolate as interp
import scipy.constants as cst

def tau_from_frequency( f ):
    """
    tau_from_frequency( f ) calculates the delay axis from a given frequency axis. The returned delay axis is the axis required by the principle components generalized projection algorithm!

    arguments:
        f : vector containing the frequency values

    returns: 
        tau : vector containing the delay values
    """
    dtau = 1./(np.amax(f)-np.amin(f))
    tau = np.arange( len(f) ) * dtau
    tau -= np.mean( tau )
    return tau


def remove_group_delay( E_t ):
    """
    remove_group_delay( E_t ) removes the group delay from an electric field. This means, that the center of gravity of the intensity is between the last and the first index of the array. This function only corrects integer delay, so it makes sure that the signal is not shifted more than 1 pixel.

    arguments:
        E_t : vector containing the complex electric field

    returns:
        E_t : vector containing the shifted complex electric field
    """
    Nt = len( E_t )
    ind = np.arange( Nt )
    ind = ind - np.mean( ind )
    E_t = np.roll( E_t, int( Nt/2 ) )
    cog = np.sum(ind * abs( E_t )**2) / np.sum( abs( E_t )**2 )
    if np.isnan( cog ):
        return E_t
    E_t = np.roll( E_t, int( -cog ) )
    E_t = np.roll( E_t, int( Nt/2 ) )
    return E_t

def raw_trace_to_valid_frog_grid( trace, lambda_c, dlambda, dtau, grid_size=64, df=None ):
    """
    raw_trace_to_valid_frog_grid( trace, lambda_c, dlambda, dtau, grid_size=64, df=None ) convertes a equidistant wavelength frog trace to a trace equidistant in frequency domain. Addiotionally, it makes sure that the delay and frequency axis fulfill the conditions to apply the principle component generalized projection algorithm.

    arguments:
        trace - FROG trace
        lambda_c - central wavelength of the measured trace
        dlambda - wavelength difference in the wavelength axis
        dtau - delay step in the measurement
        grid_size - grid size of the output trace
        df (optional) - frequency spacing for the output trace. If no df is given (default) then df is chosen to be 1/dtau/grid_size, which preserves the delay step size.

    returns:
        trace - FROG on an equidistant frequency axis
        f - frequency axis of the trace (centered at zero!)
        f_c - central frequency of the frequency axis
        tau - delay axis of the interpolated trace
    """


    if df is None:
        df = 1 / grid_size / dtau
    Nlambda, Ntau = trace.shape
    lambda_orig = np.arange( Nlambda ) * dlambda
    lambda_orig -=  np.mean( lambda_orig )
    lambda_orig += lambda_c
    tau_orig = np.arange( Ntau ) * dtau
    tau_orig -= np.mean( tau_orig )

    freq_marginal = np.sum( trace, 1 ) * lambda_orig**2
    cog_lambda = np.sum(lambda_orig * freq_marginal) / np.sum( freq_marginal )
    trace_fun = interp.interp2d( tau_orig, lambda_orig, trace, kind='cubic', fill_value=0 )
    
    f_c = cst.c / cog_lambda
    f = np.arange( grid_size ) * df
    f -= np.mean( f )
    tau = tau_from_frequency( f )
    trace = trace_fun( tau, cst.c / (f+f_c) )/(f+f_c)**2
    f = np.fft.fftshift( f )
    trace = np.fft.fftshift( trace, 0 )
    trace /= np.amax( trace )
    return trace, f, f_c, tau

def center_trace( trace, shifted=False ):
    """
    center_trace( trace, shifted=False ) centers the trace on the delay axis. After this procedure the center of mass of the delay marginal (autocorrelation) will be 0.

    arguments:
        trace : FROG trace
        shifted (optional) : is the trace fft shifted? default is False

    returns:
        trace : corrected trace
    """
    if shifted:
        trace = np.fft.fftshift( trace, axes=0 )
    Nf, Ntau = trace.shape

    delay_marginal = np.sum( trace, 0 )
    tau = np.arange( Ntau )
    cog = np.sum( delay_marginal * tau ) / np.sum( delay_marginal )
    to_shift = Ntau/2 - cog 
    tau_new = tau - to_shift

    trace_fun = interp.interp2d( tau, np.arange( Nf ), trace, kind='linear', fill_value=0 )
    trace_new = trace_fun( tau_new, np.arange( Nf ) )

    return trace_new

def correct_symmetry( trace, mean='geometric' ):
    """
    correct_symmetry( trace, mean='geometric' ) symmetrizes a trace. By definition the delay marginal (autocorrelation) is supposed to be symmetric. This function uses the mean of the delay marginal and the reversed delay marginal to correct the asymmetry. The method for calculating the mean can be 'geometric' or 'arithmetic'.

    USE THIS FUNCTION WITH CARE!!!

    argumnts:
        trace : FROG trace
        mean (optional) : method to calculate the mean, either 'geometric'(default) or 'arithmetic'.

    returns:
        trace : the corrected trace
    """
    Nf, Ntau = trace.shape
    trace = center_trace( trace )
    delay_marginal = np.sum( trace, 0 )
    if mean == 'geometric':
        cor_dm = ( delay_marginal * np.flip( delay_marginal ) )**(0.5)
    elif mean == 'arithmetic':
        cor_dm = 0.5 * (delay_marginal + np.flip( delay_marginal ) )
    old_err_settings = np.seterr( invalid='ignore' )
    trace = trace * (cor_dm/delay_marginal)[np.newaxis,:]
    np.seterr( invalid=old_err_settings['invalid'] )
    trace[ np.isnan( trace ) ] = 0

    return trace
