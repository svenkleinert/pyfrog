import numpy as np
import matplotlib.pyplot as plt
errors = []


""" pyfrog main Module
    Here are the functions defined that should be imported(and used) by the user.
"""

""" Make the noise substraction methods available"""
from pyfrog.noise import lowest_pixel as lowest_pixel_noise_substraction
from pyfrog.noise import edge_pixel as edge_pixel_noise_substraction
from pyfrog.noise import full_spectrum as full_spectrum_noise_substraction
from pyfrog.noise import fourier_low_pass as fourier_low_pass_noise_substraction
from pyfrog.noise import pixel_cleanup as pixel_cleanup_noise_substraction


""" Make the diagnostic functions available """
from pyfrog.diagnostics import *

""" Make retrrievals available """
from pyfrog.retrieval import pcgpa as retrieve_pcgpa
from pyfrog.retrieval import vanilla as retrieve_vanilla


from pyfrog.grid import remove_group_delay
from pyfrog.grid import tau_from_frequency

from pyfrog.frog import generate_trace
from pyfrog.frog import g_error

from pyfrog.colormap import frog as frog_colormap



def get_errors():
    global errors
    return errors

