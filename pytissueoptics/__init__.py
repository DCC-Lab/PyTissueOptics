import matplotlib
import os
import numpy as np

import math
import time
import warnings
try:
    import cupy as cp
except:
    cp = np

# must be before importing matplotlib.pyplot or pylab!
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('tkagg')


""" We import almost everything by default, in the general 
namespace because it is simpler for everyone """

from pytissueoptics.vector import *
from pytissueoptics.scalars import *
from pytissueoptics.vectors import *
from pytissueoptics.surface import *
from pytissueoptics.material import *
from pytissueoptics.photon import *
from pytissueoptics.source import *
from pytissueoptics.geometry import *
from pytissueoptics.detector import *
from pytissueoptics.world import *
from pytissueoptics.stats import *


__version__ = "1.0.4"
__author__ = "Daniel Cote <dccote@cervo.ulaval.ca>"

