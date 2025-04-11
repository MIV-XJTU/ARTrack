import sys
sys.path.append("/home/djh/python-code/Artrackv2/2stage")
from lib.models.ostrack.vit import *
from lib.test.evaluation.tracker import Tracker
from lib.models.layers.mask_decoder import build_maskdecoder
from lib.models.layers.head import DropPathAllocator

tracker = Tracker('ostrack', '2stage_256_got', 'got10k_test', None)
param = tracker.get_parameters()
cfg = param.cfg
patch_start_index = 1
kwargs = {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'drop_path_rate': 0.1}
model = VisionTransformer(**kwargs)
model.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
cross_2_decoder = build_maskdecoder(cfg)
drop_path = cfg.MODEL.DROP_PATH
drop_path_allocator = DropPathAllocator(drop_path)
num_heads = cfg.MODEL.NUM_HEADS
mlp_ratio = cfg.MODEL.MLP_RATIO
qkv_bias = cfg.MODEL.QKV_BIAS
drop_rate = cfg.MODEL.DROP_RATE
attn_drop = cfg.MODEL.ATTN_DROP
#def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
score_mlp = build_score_decoder(cfg)
cover_mlp = build_score_decoder(cfg)

model = OSTrack(
    backbone,
    #decoder,
    cross_2_decoder,
    score_mlp,
    #cover_mlp,
)
for name,param in model.parameters_and_names():
    print (param.name)
