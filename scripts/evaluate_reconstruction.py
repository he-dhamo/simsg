#!/usr/bin/python
#
# Copyright 2020 Helisa Dhamo, Iro Laina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import time
import datetime
import os
import yaml
import tqdm
from addict import Dict
from collections import defaultdict

import pickle
import random
import pytorch_ssim
import argparse

from simsg.data import imagenet_deprocess_batch
from simsg.metrics import jaccard
from simsg.model import SIMSGModel
from simsg.loader_utils import build_eval_loader
from simsg.utils import int_tuple, bool_flag

from scripts.eval_utils import bbox_coordinates_with_margin, parse_bool, visualize_imgs_boxes, visualize_scene_graphs

from imageio import imsave

from PerceptualSimilarity import models

GPU = 0
EVAL_ALL = True         # evaluate on all bounding boxes (batch size=1)
IGNORE_SMALL = True

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='./experiments/vg/')
parser.add_argument('--experiment', default="spade_vg", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--dataset', default='vg', choices=['clevr', 'vg'])
parser.add_argument('--with_feats', default=True, type=bool_flag)
parser.add_argument('--generative', default=True, type=bool_flag)
parser.add_argument('--predgraphs', default=True, type=bool_flag)
parser.add_argument('--mode', default='auto_nofeats', type=str)

parser.add_argument('--data_h5', default=None)

parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--print_every', default=500, type=int)
parser.add_argument('--save_every', default=500, type=int)

parser.add_argument('--visualize_imgs_boxes', default=False, type=bool_flag)
parser.add_argument('--visualize_graphs', default=False, type=bool_flag)
parser.add_argument('--save_images', default=False, type=bool_flag)
parser.add_argument('--save_gt_images', default=False, type=bool_flag)

args = parser.parse_args()

if args.dataset == "clevr":
    DATA_DIR = "./dataset/clevr/target"
    args.data_image_dir = DATA_DIR
else:
    DATA_DIR = "./datasets/vg/"
    args.data_image_dir = os.path.join(DATA_DIR, 'images')

if args.data_h5 is None:
    if args.predgraphs:
        args.data_h5 = os.path.join(DATA_DIR, 'test_predgraphs.h5')
    else:
        args.data_h5 = os.path.join(DATA_DIR, 'test.h5')

if args.checkpoint is None:
    ckpt = args.exp_dir + args.experiment
    args.checkpoint = '{}_model.pt'.format(ckpt)

CONFIG_FILE = args.exp_dir + 'args_64_spade_vg.yaml'.format(args.experiment)
IMG_SAVE_PATH = args.exp_dir + 'logs/{}/evaluation/'.format(args.experiment)
RESULT_SAVE_PATH = args.exp_dir + 'logs/{}/evaluation/results/'.format(args.experiment)
RESULT_FILE = RESULT_SAVE_PATH + '{}/test_results_{}.pickle'

USE_GT_BOXES = True     # use ground truth bounding boxes for evaluation
print("feats", args.with_feats)
torch.cuda.set_device(GPU)
device = torch.device(GPU)

def remove_vgg(model_state):
  def filt(pair):
    key, val = pair
    return "high_level_feat" not in key

  return dict(filter(filt, model_state.items()))
def main():

    if not os.path.isfile(args.checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
        return

    # Read config file of the model
    config = Dict(yaml.load(open(CONFIG_FILE)))

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(args.checkpoint, map_location=map_location)

    vocab = checkpoint['model_kwargs']['vocab']

    # initialize model and load checkpoint
    kwargs = checkpoint['model_kwargs']
    model = SIMSGModel(**kwargs)
    new_state = remove_vgg(checkpoint['model_state'])
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)

    # create data loaders
    test_loader = build_eval_loader(args, checkpoint, no_gt=True) #no gt for pairs

    print('Evaluating on test set')
    eval_model(model, test_loader, device, vocab, use_gt_boxes=USE_GT_BOXES, use_feats=args.with_feats,
               filter_box=IGNORE_SMALL)


def eval_model(model, loader, device, vocab, use_gt_boxes=False, use_feats=False, filter_box=False):
    all_boxes = defaultdict(list)
    total_iou = []
    total_boxes = 0
    num_batches = 0
    num_samples = 0
    mae_per_image = []
    mae_roi_per_image = []
    roi_only_iou = []
    ssim_per_image = []
    ssim_rois = []
    rois = 0
    margin = 2

    ## Initializing the perceptual loss model
    lpips_model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)
    perceptual_error_image = []
    # ---------------------------------------

    img_idx = 0

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            num_batches += 1
            # if num_batches > 10:
            #     break
            batch = [tensor.to(device) for tensor in batch]
            masks = None
            #len", len(batch))

            imgs, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = [b.to(device) for b in batch]
            predicates = triples[:, 1]

            #EVAL_ALL = True
            if not args.generative:
                imgs, imgs_in, objs, boxes, triples, obj_to_img, \
                dropimage_indices, dropfeats_indices = [b.to(device) for b in process_batch(
                    imgs, imgs_in, objs, boxes, triples, obj_to_img, triple_to_img, device,
                    use_feats=use_feats, filter_box=filter_box)]

                dropbox_indices = dropimage_indices
            else:
                dropbox_indices = torch.ones_like(objs.unsqueeze(1).float()).to(device)
                dropfeats_indices = torch.ones_like(objs.unsqueeze(1).float()).to(device)
                dropimage_indices = torch.zeros_like(objs.unsqueeze(1).float()).to(device)

            if imgs.shape[0] == 0:
                continue

            if args.visualize_graphs:
                # visualize scene graphs for debugging purposes
                visualize_scene_graphs(obj_to_img, objs, triples, vocab, device)

            if use_gt_boxes:
                model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks, src_image=imgs_in,
                                  keep_box_idx=torch.ones_like(dropimage_indices), keep_feat_idx=dropfeats_indices,
                                  keep_image_idx=dropimage_indices, mode='eval')
            else:
                model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, src_image=imgs_in,
                                  keep_box_idx=dropimage_indices, keep_feats_idx=dropfeats_indices,
                                  keep_image_idx=dropimage_indices, mode='eval')

            # OUTPUT
            imgs_pred, boxes_pred, masks_pred, _, _ = model_out
            # ----------------------------------------------------------------------------------------------------------

            # Save all box predictions
            all_boxes['boxes_gt'].append(boxes)
            all_boxes['objs'].append(objs)
            all_boxes['boxes_pred'].append(boxes_pred)
            all_boxes['drop_targets'].append(dropbox_indices)

            # IoU over all
            total_iou.append(jaccard(boxes_pred, boxes).detach().cpu().numpy())
            total_boxes += boxes_pred.size(0)

            # IoU over targets only
            pred_dropbox = boxes_pred[dropbox_indices.squeeze() == 0, :]
            gt_dropbox = boxes[dropbox_indices.squeeze() == 0, :]
            roi_only_iou.append(jaccard(pred_dropbox, gt_dropbox).detach().cpu().numpy())
            rois += pred_dropbox.size(0)

            num_samples += imgs.shape[0]
            imgs = imagenet_deprocess_batch(imgs).float()
            imgs_pred = imagenet_deprocess_batch(imgs_pred).float()

            if args.visualize_imgs_boxes:
                # visualize images with drawn boxes for debugging purposes
                visualize_imgs_boxes(imgs, imgs_pred, boxes, boxes_pred)

            if args.save_images:
                # save reconstructed images for later FID and Inception computation
                if args.save_gt_images:
                    # pass imgs as argument to additionally save gt images
                    save_images(imgs_pred, img_idx, imgs)
                else:
                    save_images(imgs_pred, img_idx)

            # MAE per image
            mae_per_image.append(torch.mean(
                torch.abs(imgs - imgs_pred).view(imgs.shape[0], -1), 1).cpu().numpy())

            for s in range(imgs.shape[0]):
                # get coordinates of target
                left, right, top, bottom = bbox_coordinates_with_margin(boxes[s, :], margin, imgs)
                if left > right or top > bottom:
                    continue

                # calculate errors only in RoI one by one
                mae_roi_per_image.append(torch.mean(
                    torch.abs(imgs[s, :, top:bottom, left:right] -
                              imgs_pred[s, :, top:bottom, left:right])).cpu().item())

                ssim_per_image.append(
                    pytorch_ssim.ssim(imgs[s:s+1, :, :, :] / 255.0,
                                      imgs_pred[s:s+1, :, :, :] / 255.0, window_size=3).cpu().item())
                ssim_rois.append(
                    pytorch_ssim.ssim(imgs[s:s+1, :, top:bottom, left:right] / 255.0,
                                      imgs_pred[s:s+1, :, top:bottom, left:right] / 255.0, window_size=3).cpu().item())

                # normalize as expected from the LPIPS model
                imgs_pred_norm = imgs_pred[s:s+1, :, :, :] / 127.5 - 1
                imgs_gt_norm = imgs[s:s+1, :, :, :] / 127.5 - 1
                perceptual_error_image.append(
                    lpips_model.forward(imgs_pred_norm, imgs_gt_norm).detach().cpu().numpy())

            if num_batches % args.print_every == 0:
                calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                                 perceptual_error_image)

            if num_batches % args.save_every == 0:
                save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_error_image, all_boxes, num_batches)

            img_idx += 1

    calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                                 perceptual_error_image)
    save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_error_image, all_boxes, 'final')


def calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                     perceptual_image):

    mae_all = np.mean(np.hstack(mae_per_image), dtype=np.float64)
    mae_std = np.std(np.hstack(mae_per_image), dtype=np.float64)
    mae_roi = np.mean(mae_roi_per_image, dtype=np.float64)
    mae_roi_std = np.std(mae_roi_per_image, dtype=np.float64)
    iou_all = np.mean(np.hstack(total_iou), dtype=np.float64)
    iou_std = np.std(np.hstack(total_iou), dtype=np.float64)
    iou_roi = np.mean(np.hstack(roi_only_iou), dtype=np.float64)
    iou_roi_std = np.std(np.hstack(roi_only_iou), dtype=np.float64)
    ssim_all = np.mean(ssim_per_image, dtype=np.float64)
    ssim_std = np.std(ssim_per_image, dtype=np.float64)
    ssim_roi = np.mean(ssim_rois, dtype=np.float64)
    ssim_roi_std = np.std(ssim_rois, dtype=np.float64)
    # percept error -----------
    percept_all = np.mean(perceptual_image, dtype=np.float64)
    percept_all_std = np.std(perceptual_image, dtype=np.float64)
    # ------------------------

    print()
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all, mae_std))
    print('MAE-RoI: Mean {:.6f}, Std {:.6f}: '.format(mae_roi, mae_roi_std))
    print('IoU: Mean {:.6f}, Std {:.6f}'.format(iou_all, iou_std))
    print('IoU-RoI: Mean {:.6f}, Std {:.6f}'.format(iou_roi, iou_roi_std))
    print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all, ssim_std))
    print('SSIM-RoI: Mean {:.6f}, Std {:.6f}'.format(ssim_roi, ssim_roi_std))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all, percept_all_std))


def save_results(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                 perceptual_per_image, all_boxes, iter):

    results = dict()
    results['mae_per_image'] = mae_per_image
    results['mae_rois'] = mae_roi_per_image
    results['iou_per_image'] = total_iou
    results['iou_rois'] = roi_only_iou
    results['ssim_per_image'] = ssim_per_image
    results['ssim_rois'] = ssim_rois
    results['perceptual_per_image'] = perceptual_per_image
    results['data'] = all_boxes

    subdir = parse_bool(args.predgraphs, args.generative, USE_GT_BOXES, args.with_feats)

    if not os.path.exists(RESULT_SAVE_PATH + subdir):
        os.makedirs(RESULT_SAVE_PATH + subdir)
    with open(RESULT_FILE.format(subdir, iter), 'wb') as p:
        pickle.dump(results, p)


def process_batch(imgs, imgs_in, objs, boxes, triples, obj_to_img, triples_to_img, device,
                  use_feats=True, filter_box=False):
    num_imgs = imgs.shape[0]
    imgs_stack = []
    imgs_in_stack = []
    boxes_stack = []
    objs_stack = []
    triples_stack = []
    obj_to_img_new = []
    candidates_stack = []
    previous_idx = 0

    for i in range(num_imgs):
        start_idx_for_img = (obj_to_img == i).nonzero()[0]
        last_idx_for_img = (obj_to_img == i).nonzero()[-1]
        boxes_i = boxes[start_idx_for_img: last_idx_for_img + 1, :]     # this includes the 'image' box!
        objs_i = objs[start_idx_for_img: last_idx_for_img + 1]

        start_idx_for_img = (triples_to_img == i).nonzero()[0]
        last_idx_for_img = (triples_to_img == i).nonzero()[-1]
        triples_i = triples[start_idx_for_img:last_idx_for_img + 1]

        num_boxes = boxes_i.shape[0]  # number of boxes in current image minus the 'image' box

        if filter_box:
            min_dim = 0.05  # about 3 pixels
            keep = [b for b in range(boxes_i.shape[0] - 1) if
                    boxes_i[b, 2] - boxes_i[b, 0] > min_dim and boxes_i[b, 3] - boxes_i[b, 1] > min_dim]

            times_to_rep = len(keep)
            img_indices = torch.LongTensor(keep)
        else:

            times_to_rep = num_boxes - 1
            img_indices = torch.arange(0, times_to_rep)
            keep = img_indices

        # boxes that will be dropped for each sample
        drop_indices = torch.zeros_like(img_indices)
        for j in range(len(keep)):
            drop_indices[j] = num_boxes * j + keep[j]

        # replicate things for current image
        imgs_stack.append(imgs[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        imgs_in_stack.append(imgs_in[i, :, :, :].repeat(times_to_rep, 1, 1, 1))
        objs_stack.append(objs_i.repeat(times_to_rep))     # replicate object ids #boxes times
        boxes_stack.append(boxes_i.repeat(times_to_rep, 1))   # replicate boxes #boxes times

        obj_to_img_new.append(torch.arange(0, times_to_rep).repeat(num_boxes, 1)
                              .transpose(1,0).reshape(-1) + previous_idx)

        previous_idx = obj_to_img_new[-1].max() + 1

        triplet_offsets = num_boxes * torch.arange(0, times_to_rep).repeat(triples_i.size(0), 1)\
            .transpose(1,0).reshape(-1).to(device)

        triples_i = triples_i.repeat(times_to_rep, 1)
        triples_i[:, 0] = triples_i[:, 0] + triplet_offsets     # offset for replicated subjects
        triples_i[:, 2] = triples_i[:, 2] + triplet_offsets     # offset for replicated objects
        triples_stack.append(triples_i)

        # create index to drop for each sample
        candidates = torch.ones(boxes_stack[-1].shape[0], device=device)

        candidates[drop_indices] = 0     # set to zero the boxes that should be dropped
        candidates_stack.append(candidates)

    imgs = torch.cat(imgs_stack)
    imgs_in = torch.cat(imgs_in_stack)
    boxes = torch.cat(boxes_stack)
    objs = torch.cat(objs_stack)
    triples = torch.cat(triples_stack)
    obj_to_img_new = torch.cat(obj_to_img_new)
    candidates = torch.cat(candidates_stack).unsqueeze(1)

    if use_feats:
        feature_candidates = torch.ones((candidates.shape[0], 1), device=device)
    else:
        feature_candidates = candidates

    return imgs, imgs_in, objs, boxes, triples, obj_to_img_new, candidates, feature_candidates


def save_images(imgs_pred, img_idx, imgs=None, first_only=True):

    num_images = imgs_pred.size(0)
    assert num_images > 0

    subdir = parse_bool(args.predgraphs, args.generative, USE_GT_BOXES, args.with_feats)

    img_path = IMG_SAVE_PATH + subdir + '/' + str(img_idx).zfill(4)
    if not os.path.exists(IMG_SAVE_PATH + subdir):
        os.makedirs(IMG_SAVE_PATH + subdir)

    if first_only:
        n = 1
    else:
        n = num_images

    imgs_pred = imgs_pred.detach().cpu().numpy().transpose([0, 2, 3, 1])
    for i in range(n):
        imsave(img_path + '_' + str(i) + '.png', imgs_pred[i].astype('uint8'))

    if imgs is not None:
        img_path_gt = IMG_SAVE_PATH + 'gt/' + str(img_idx).zfill(4) + '.png'
        if not os.path.exists(IMG_SAVE_PATH + 'gt/'):
            os.makedirs(IMG_SAVE_PATH + 'gt/')

        imgs = imgs.detach().cpu().numpy().transpose([0, 2, 3, 1])
        imsave(img_path_gt, imgs[0].astype('uint8'))


if __name__ == '__main__':
    main()
