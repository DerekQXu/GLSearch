import os
import sys

# get name of caller (test, train or debug)
caller = sys.argv[0].split(os.sep)[-1].split(".")[0]


# depending on caller, import the correct options
opt = None
if caller == "train":
    from .train_options import TrainOptions
    to = TrainOptions()
    to.initialize()
    opt = to.parse()
elif caller == "test":
    from .test_options import TestOptions
    to = TestOptions()
    to.initialize()
    opt = to.parse()
elif caller == "debug":
    from .debug_options import DebugOptions
    opt = DebugOptions()

if opt is None:
    raise ValueError("opt is None. you should call this script from train.py, test.py or debug.py.")