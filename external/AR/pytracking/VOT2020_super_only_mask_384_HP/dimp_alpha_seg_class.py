from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot
import sys
import time

'''Refine module & Pytracking base trackers'''
import os
from pytracking.evaluation import Tracker
from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *

''''''
'''DiMP-alpha class'''


class DIMP_ALPHA(object):
    def __init__(self, tracker_name='dimp', para_name='dimp50_vot19',
                 refine_model_name='ARcm_coco_seg', threshold=0.15):
        self.THRES = threshold
        '''create tracker'''
        '''DIMP'''
        tracker_info = Tracker(tracker_name, para_name, None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        self.dimp = tracker_info.tracker_class(params)
        '''Alpha-Refine'''
        project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, img_RGB, mask):
        region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        gt_bbox_torch = torch.from_numpy(gt_bbox_np)
        init_info = {}
        init_info['init_bbox'] = gt_bbox_torch
        _ = self.dimp.initialize(img_RGB, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(img_RGB, np.array(gt_bbox_np))

    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.dimp.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        '''Step1: Post-Process'''
        x1, y1, w, h = pred_bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / self.dimp.base_target_sz.prod())
        ##### update
        self.dimp.pos = new_pos.clone()
        self.dimp.target_sz = new_target_sz
        self.dimp.target_scale = new_scale
        bbox_new = [x1, y1, w, h]
        '''Step2: Mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(bbox_new), vis=True)
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        search_region = search.astype(np.uint8)
        search_mask = (search_mask > self.THRES).astype(np.uint8)
        return bbox_new, final_mask, search_region, search_mask


def run_vot_exp(tracker_name, para_name, refine_model_name, threshold, VIS=False):
    torch.set_num_threads(1)
    # torch.cuda.set_device(CUDA_ID)  # set GPU id
    save_root = os.path.join('<SAVE_DIR>', para_name)
    if VIS and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = DIMP_ALPHA(tracker_name=tracker_name, para_name=para_name,
                         refine_model_name=refine_model_name, threshold=threshold)
    handle = vot.VOT("mask")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if VIS:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, mask)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, m, search, search_m = tracker.track(image)
        handle.report(m)
        if VIS:
            '''Visualization'''
            # original image
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_ori)
            # dimp box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg', '_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
            # search region
            search_bgr = search[:, :, ::-1].copy()
            search_name = image_name.replace('.jpg', '_search.jpg')
            save_path = os.path.join(save_dir, search_name)
            cv2.imwrite(save_path, search_bgr)
            # search region mask
            search_bgr_m = search_bgr.astype(np.float32)
            search_bgr_m[:, :, 1] += 127.0 * search_m
            search_bgr_m[:, :, 2] += 127.0 * search_m
            contours, _ = cv2.findContours(search_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            search_bgr_m = cv2.drawContours(search_bgr_m, contours, -1, (0, 255, 255), 4)
            search_bgr_m = search_bgr_m.clip(0, 255).astype(np.uint8)
            search_name_m = image_name.replace('.jpg', '_search_mask.jpg')
            save_path = os.path.join(save_dir, search_name_m)
            cv2.imwrite(save_path, search_bgr_m)
            # original image + mask
            image_m = image_ori.copy().astype(np.float32)
            image_m[:, :, 1] += 127.0 * m
            image_m[:, :, 2] += 127.0 * m
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_m = cv2.drawContours(image_m, contours, -1, (0, 255, 255), 2)
            image_m = image_m.clip(0, 255).astype(np.uint8)
            image_mask_name_m = image_name.replace('.jpg', '_mask.jpg')
            save_path = os.path.join(save_dir, image_mask_name_m)
            cv2.imwrite(save_path, image_m)
