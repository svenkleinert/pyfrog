import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyfrog.diagnostics
from pyfrog.colormap import getFrogColorMap

def find_resonable_axis_limits( trace1, trace2, axis=0, padding=5, clip=1e-5 ):
    marginal = np.sum( trace1 + trace2, axis ) / trace1.shape[ not axis ]
    a = np.arange( len( marginal ) )
    ind_min = max( 0, min( a[marginal > clip] ) - padding )
    ind_max = min( len(marginal), max( a[marginal > clip] ) + padding )
    return ind_min, ind_max


def simple_frog_result_plot( f, f_c, tau, E_f, trace, plotstyle='frequency', **kwargs ):
    cmap = getFrogColorMap( 256 )
    cmap = 'viridis'
    fontsize = 22
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']

    if 'fontsize' in kwargs:
        fontsize = kwargs['fontsize']

    Nf = len( f )
    Ntau = len( tau )

    ret_spectrum = abs( E_f )**2
    ret_spectrum /= np.amax( ret_spectrum )
    ret_phase = np.unwrap( np.fft.fftshift( np.angle( E_f ) ) )
    ret_phase = np.fft.fftshift( ret_phase )
    ret_phase -= ret_phase[0]

    ret_E_t = np.fft.ifft( E_f )
    ret_E_t = np.roll( ret_E_t, int( Nf/2 ) )
    ret_E_t /= np.amax( abs( ret_E_t ) )


    ret_rms_duration = pyfrog.diagnostics.rms_pulse_duration( abs( ret_E_t )**2, tau[1]-tau[0] )
    ret_fwhm_duration = pyfrog.diagnostics.fwhm_pulse_duration( abs( ret_E_t )**2, tau[1] - tau[0] )

    ret_trace = pyfrog.generate_trace( abs( E_f )**2, np.angle( E_f ) )

    f = np.fft.fftshift( f )
    ret_spectrum = np.fft.fftshift( ret_spectrum )
    ret_phase = np.fft.fftshift( ret_phase )
    trace = np.fft.fftshift( trace, 0 )
    trace /= np.amax( trace )
    ret_trace = np.fft.fftshift( ret_trace, 0 )
    ret_trace /= np.amax( ret_trace )

    #f_marginal = np.sum( trace, 1 )
    #delay_marginal = np.sum( trace, 0 )
    ind_f_min, ind_f_max = find_resonable_axis_limits( trace, ret_trace, axis=1 )
    ind_delay_min, ind_delay_max = find_resonable_axis_limits( trace, ret_trace, axis=0 )
    trace[trace<1e-4] = 1e-4
    tau = tau[ind_delay_min:ind_delay_max]
    f = f[ind_f_min:ind_f_max]
    trace = trace[ind_f_min:ind_f_max,ind_delay_min:ind_delay_max]
    ret_trace = ret_trace[ind_f_min:ind_f_max,ind_delay_min:ind_delay_max]
    ret_spectrum = ret_spectrum[ind_f_min:ind_f_max]
    ret_phase = ret_phase[ind_f_min:ind_f_max]
    ret_E_t = ret_E_t[ind_delay_min:ind_delay_max]


    plt.rcParams.update({'font.size':fontsize})
    fig, ax = plt.subplots(nrows=2,ncols=3,squeeze=True,
                           gridspec_kw={'left':0.05,
                                        'right':0.97,
                                        'bottom':0.05,
                                        'top':0.95,
                                        'wspace':0.35})
    ax1 = ax[0,0]
    ax3 = ax[0,1]
    ax5 = ax[0,2]
    ax6 = ax[1,0]
    ax7 = ax[1,1]
    ax8 = ax[1,2]

    if plotstyle == 'frequency':
        lamf = (f+f_c)*1e-12
        lamf_fun = lamf / 2
        ax1.set_xlabel('frequency (THz')
        ax5.set_ylabel('frequency (THz)')
        ax6.set_ylabel('frequency (THz)')
        ax7.set_ylabel('frequency (THz)')
    else:
        lamf = np.flip( cst.c / (f+f_c) ) * 1e9
        lamf_fun = lamf * 2
        ret_spectrum = np.flip( ret_spectrum ) / (lamf*1e-9)**2
        ret_spectrum /= np.amax( ret_spectrum )
        ret_phase = np.flip( ret_phase )
        trace = np.flipud( trace )
        ret_trace = np.flipud( ret_trace )
        ax1.set_xlabel('wavelength (nm)')
        ax5.set_ylabel('wavelength (nm)')
        ax6.set_ylabel('wavelength (nm)')
        ax7.set_ylabel('wavelength (nm)')

    ax1.plot( lamf_fun, ret_spectrum, 'black' )
    ax2 = ax1.twinx()
    ax2.plot( lamf_fun, ret_phase, 'r' )
    ax1.set_ylabel('PSD (arb. u.)')
    ax2.set_ylabel('Phase (rad.)')
    ax2.spines['right'].set_color('red')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax3.plot( tau*1e15, abs(ret_E_t)**2, 'black' )
    ax3.set_xlabel('time (fs)')
    #ax3.get_yaxis().set_ticks([])
    ax3.set_ylabel('intensity (arb. u.)')
    ax3.text( 0.7, 0.9, '|E(t)|²', transform=ax3.transAxes, fontsize=fontsize )
    ax3.text( 0.65, 0.8, 'rms:{:.0f}fs'.format(ret_rms_duration*1e15), transform=ax3.transAxes, fontsize=fontsize )
    ax3.text( 0.65, 0.7, 'fwhm:{:.0f}fs'.format(ret_fwhm_duration*1e15), transform=ax3.transAxes, fontsize=fontsize )
    ax4 = ax3.twinx()
    ax4.plot( tau*1e15, np.unwrap( np.angle( ret_E_t ) ), 'r' )
    ax4.spines['right'].set_color('red')
    ax4.yaxis.label.set_color('red')
    ax4.tick_params(axis='y', colors='red')
    ax4.set_ylabel('Phase (rad.)')

    ax5.text( 0.6, 0.9, 'Difference', transform=ax5.transAxes, color='w', fontsize=fontsize )
    mesh = ax5.pcolormesh( tau*1e15, lamf, ret_trace -  trace, cmap=cmap )
    plt.colorbar(mesh, ax=ax5)
    ax5.set_xlabel('delay (fs)')
    ax6.text( 0.7, 0.9, 'Trace', transform=ax6.transAxes, color='w', fontsize=fontsize )
    ax6.pcolormesh( tau*1e15, lamf, trace, norm=LogNorm(vmin=1e-4,vmax=1,clip=True), cmap=cmap )
    ax6.set_xlabel('delay (fs)')
    ax7.text( 0.7, 0.9, 'Retrieval', transform=ax7.transAxes, color='w', fontsize=fontsize )
    ax7.pcolormesh( tau*1e15, lamf, ret_trace, norm=LogNorm(vmin=1e-4,vmax=1,clip=True), cmap=cmap )
    ax7.set_xlabel('delay (fs)')
    errors = pyfrog.get_errors()
    ax8.semilogy( np.array(errors) )
    ax8.set_xlabel('iteration')
    ax8.set_ylabel('FROG error (%)')

    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    plt.show()
