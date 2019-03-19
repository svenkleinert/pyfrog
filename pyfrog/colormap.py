import numpy as np
from matplotlib.colors import ListedColormap

def getFrogColorMap( steps=256 ):
    """ getFrogColorMap( steps=256 ) generates a colormap for matplotlib similar to the one used by the original FROG software. Color values are taken from the github repository of kenwdelong.

    arguments:
        steps : integer number of steps of the colormap

    returns:
        colormap : instance of matplotlib.colors.ListedColormap
    """
    colors = np.zeros( (steps,3) )
    step1 = steps * 0.15
    step2 = steps * 0.5
    step3 = steps * 0.95
    for i in range( 1, steps ):
        if i <= step1:
            red = 1
            green = 1*i/step1
            if green != 0:
                green = green**0.7
            blue = 0
        elif i <= step2:
            red = (step2 - i)/(step2-step1)
            if red != 0:
                red = red**0.7
            green = 1
            blue = 0
        elif i <= step3:
            red = 0
            green = (step3 - i)/(step3 - step2)
            if green != 0:
                green = green**0.8
            blue = (i-step2)/(step3 - step2)
            if blue != 0:
                blue = blue**0.8
        else:
            red = (i - step3)/(steps-step3)
            green = red
            blue = 1
        colors[i,:] = [ red, green, blue ]
    cmap = ListedColormap( colors, 'frog' )
    return cmap

frog = getFrogColorMap(256) # available as pyfrog.colormap.frog
