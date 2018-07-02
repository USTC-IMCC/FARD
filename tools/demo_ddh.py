#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',  # always index 0
                     '1', '2', '3', '4',
                     '5', '6')
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
# groud_truth = '/home/shangzh/pytorch-faster-rcnn/data/VOCdevkit2018/VOC2018/Annotations'
groud_truth = '/home/shangzh/pytorch-faster-rcnn/data/test0701/txt'
def vis_detections(im2show, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        print('not find object')
        return im2show

    #im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')
        # cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 205, 51), 2)
        cv2.circle(im2show, (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), 2, (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (class_name, score), (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2) + 15), cv2.FONT_HERSHEY_PLAIN,
                    4.0, (0, 0, 255), thickness=2)
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)

    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    return im2show

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join('/home/shangzh/pytorch-faster-rcnn/data/VOCdevkit2018/VOC2018/JPEGImages', image_name)
    im_file = os.path.join('/home/shangzh/pytorch-faster-rcnn/data/test0701/original_jpg', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    result_txt = {}
    im2show = np.copy(im)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= 0.5)[0]
        for i in inds:
            bbox = dets[i, :4]
            result_txt[cls] = '%s %s ' % (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))
        im2show = vis_detections(im2show, cls, dets, thresh=0.5)#CONF_THRESH)
    with open('/home/shangzh/pytorch-faster-rcnn/demo/output_%s.txt' % (image_name[:-4]), 'w') as f_txt:
        for i in range(len(CLASSES) - 1):
            try:
                f_txt.write(result_txt[CLASSES[i + 1]])
            except:
                f_txt.write('not find all object')
                print('not find all object !!!!!!!!!!!!!!!!!!/home/shangzh/pytorch-faster-rcnn/demo/output_%s.txt' % (image_name[:-4]))
    with open(os.path.join(groud_truth, ('%s.txt' % image_name[:-4]))) as f:
        content = f.read()
        string = content.split(' ')
        for i in range(6):
          cv2.circle(im2show, (int(round(float(string[2*i]))), int(round(float(string[2*i+1])))), 2, (51, 205, 251), 2)
    cv2.imwrite('/home/shangzh/pytorch-faster-rcnn/demo/output_%s.jpg' % (image_name), im2show)
    #print('save to %s' % '/home/shangzh/pytorch-faster-rcnn/demo/output_%s.jpg' % (image_name))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    # saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
    #                           NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))
    # saved_model = '/home/shangzh/pytorch-faster-rcnn/output/res101/voc_2007_trainval+voc_2012_trainval/default/res101_faster_rcnn_iter_70000.pth'
    saved_model = '/home/shangzh/pytorch-faster-rcnn/output/res101/voc_2018_trainval/default/res101_faster_rcnn_iter_105000.pth'

    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(7,
                          tag='default', anchor_scales=[8, 16, 32])
    # if not torch.cuda.is_available():
    #     net._device = 'cpu'
    # net.to(net._device)
    # net.cuda(0)
    # net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage.cuda()))
    # state_dict = torch.load(saved_model)
    # print(state_dict)
    # net.load_state_dict({k: state_dict[k] for k in list(net.state_dict())})
    net.load_state_dict(torch.load(saved_model, map_location=lambda storage, loc: storage))
    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)


    print('Loaded network {:s}'.format(saved_model))

    # with open('/home/shangzh/pytorch-faster-rcnn/data/VOCdevkit2018/VOC2018/ImageSets/Main/test.txt', 'r') as f:
    #     im_names = f.readlines()
    #     for index, im_name in enumerate(im_names):
    #         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         #print('Demo for /home/shangzh/pytorch-faster-rcnn/data/VOCdevkit2018/VOC2018/JPEGImages/{}.jpg'.format(im_name[:-1]))
    #         demo(net, '%s.jpg' % im_name[:-1])
    list_dirs = os.walk('/home/shangzh/pytorch-faster-rcnn/data/test0701/original_jpg')
    for root, dirs, files in list_dirs:
        for f in files:
            if f[-4:] == '.jpg':
                demo(net, f)