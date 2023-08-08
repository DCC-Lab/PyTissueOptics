# We set up the environment o we do not have to install PyTissueOptics to run the examples.
# By adjusting the path, we use the current version in development 

# append module root directory to sys.path
import sys
import os
sys.path.insert(0,
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__)))))
