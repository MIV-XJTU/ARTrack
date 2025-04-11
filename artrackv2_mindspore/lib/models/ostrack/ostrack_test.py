import sys
sys.path.append("/home/djh/python-code/Artrackv2/2stage")
from lib.models.ostrack.vit import *
from lib.test.evaluation.tracker import Tracker
from lib.models.ostrack import *
tracker = Tracker('ostrack', '2stage_256_got', 'got10k_test', None)
param = tracker.get_parameters()
cfg = param.cfg
model = build_ostrack(cfg,training=False)

