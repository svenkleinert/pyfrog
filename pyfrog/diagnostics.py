import numpy as np

def rms_pulse_duration( I_t, dt=None ):
    """
    rms_pulse_duration( I_t, dt=None ) calculates the rms pulse duration as defined in Trebino's book.

    arguments:
        I_t : vector containing the temporal intensity
        dt (optional) : time difference between two values of the I_t vector. If none is provided, the pulse duration is given in units of pixels.

    returns:
        tau : the rms pulse duration
    """
    if dt is None:
        dt = 1
    Nt = len( I_t )
    t = np.arange( Nt ) * dt
    t -= np.mean( t )
    _I_t = I_t / np.sum( I_t )
    _I_t = np.roll( _I_t, int( Nt/2 ) - np.argmax( _I_t ) )
    first_order_momentum = np.sum( t * _I_t )
    second_order_momentum = np.sum( t**2 * _I_t )
    return np.sqrt( abs(second_order_momentum - first_order_momentum**2) )

def fwhm_pulse_duration( I_t, dt=None ):
    """
    fwhm_pulse_duration( t, dt=None ) calculates the full width at half maximum pulse duration. The precision is limited by the resolution of the time axis!

    arguments:
        I_t : vector containing the temporal intensity
        dt (optional) : time difference between two values of the I_t vector. If none is provided, the pulse duration is given in units of pixels.

    returns:
        tau : the FWHM pulse duration
    """
    if dt is None:
        dt=1
    Nt = len(I_t)
    _I_t = I_t / np.amax( I_t )
    _I_t = np.roll( _I_t, int( Nt/2 ) - np.argmax( _I_t ) )
    ind = np.arange( Nt )
    ind = np.where( _I_t >= 0.5, ind, 0 )
    ind = ind[ np.nonzero(ind) ]
    return dt * (np.amax(ind) - np.amin(ind) )
