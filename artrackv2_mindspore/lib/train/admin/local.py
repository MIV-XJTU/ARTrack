class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/baiyifan/code/prev_for_2stage'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/baiyifan/code/detrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/baiyifan/code/OSTrack/pretrained_networks'
        self.lasot_dir = '/home/baiyifan/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/home/baiyifan/GOT-10k/train'
        self.got10k_val_dir = '/home/baiyifan/GOT-10k/val'
        self.lasot_lmdb_dir = '/home/baiyifan/LaSOT/LaSOTBenchmark'
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/ssddata/TrackingNet/all_zip'
        self.trackingnet_lmdb_dir = '/ssddata/TrackingNet/all_zip'
        self.coco_dir = '/home/baiyifan/coco'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/baiyifan/code/OSTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/baiyifan/code/OSTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
