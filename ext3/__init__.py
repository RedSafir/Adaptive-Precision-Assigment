"""
ext3.
|- typing.py.   [DEPEND: x]
|- util.        [DEPEND: x]
|- core.
   |- include.  [DEPEND: x]
   |- emodlcls. [DEPEND: core.inclue]
   |- emodlobj. [DEPEND: core.emodlcls]
|- nn.          [DEPEND: core]

TODO.
- mypy: emodlobj.
- train: main.
"""

from . import util, core, nn
