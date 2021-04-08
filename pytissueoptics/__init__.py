import matplotlib
import os
# must be before importing matplotlib.pyplot or pylab!
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

""" We import almost everything by default, in the general 
namespace because it is simpler for everyone """

__version__ = "1.0.4"
__author__ = "Daniel Cote <dccote@cervo.ulaval.ca>"

