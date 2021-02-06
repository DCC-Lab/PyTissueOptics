import matplotlib
import os
# must be before importing matplotlib.pyplot or pylab!
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import math

""" We import almost everything by default, in the general 
namespace because it is simpler for everyone """

from .vector import *
from .surface import *
from .material import *
from .photon import *
from .source import *
from .geometry import *
from .detector import *
from .world import *
from .stats import *


__version__ = "1.0.4"
__author__ = "Daniel Cote <dccote@cervo.ulaval.ca>"

