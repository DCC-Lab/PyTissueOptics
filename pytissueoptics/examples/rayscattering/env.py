# We set up the environment so we do not have to install PyTissueOptics to run the examples.
# By adjusting the path, we use the current version in development.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
