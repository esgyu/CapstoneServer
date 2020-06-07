"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse
import image_proc as ip

import glob
import torch
from torch.autograd import Variable
import cv2
import numpy as np
import crafts.craft_utils
import crafts.imgproc
import crafts.file_utils
from crafts.craft import CRAFT
from collections import OrderedDict
from crafts.refinenet import RefineNet

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.6, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.1, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = crafts.file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = crafts.imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                                 interpolation=cv2.INTER_LINEAR,
                                                                                 mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = crafts.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = crafts.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text,
                                                  poly)

    # coordinate adjustment
    boxes = crafts.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = crafts.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = crafts.imgproc.cvt2HeatmapImg(render_img)

    if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def text_detect(net, refine_net, image):
    t = time.time()

    #print("Test image {:d}/{:d}: {:s}".format(1, len(image_list), image_path), end='\r')
    #image = CRAFTS.imgproc.loadImage(image_path)

    # image, rot = ip.image_resize(image)
    # image = ip.hough_linep(image)
    # print(image.shape)
    # image = ip.edge_detect(image)
    # print(image.shape)

    boxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                         args.cuda, args.poly, refine_net)

    # save score text
    filename = time.strftime("%Y%m%d-%H%M%S")
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    crafts.file_utils.saveResult(mask_file, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

    return boxes

if __name__ == '__main__':
    nets = CRAFT()
    print("[INFO] loading CRAFT text detector...")
    nets.load_state_dict(copyStateDict(torch.load('weights/craft_mlt_25k.pth', map_location='cpu')))

    print("[INFO] loading CRAFT REFINER...")
    refine_nets = RefineNet()
    refine_nets.load_state_dict(
        copyStateDict(torch.load('weights/craft_refiner_CTW1500.pth', map_location='cpu')))

    files = glob.glob('data/*.jpg')
    for file in files:
        image = cv2.imread(file)
        text_detect(nets, refine_nets, image)