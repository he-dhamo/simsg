#!/usr/bin/python
#
# Copyright 2020 Azade Farshad
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

"""
This script can be used to test changes for clevr
"""

import argparse, json
import os

import torch
from collections import Counter

from imageio import imsave
import matplotlib.pyplot as plt

from simsg.data import imagenet_deprocess_batch
from simsg.model import SIMSGModel
from simsg.utils import int_tuple, bool_flag

import pytorch_ssim
from simsg.metrics import jaccard
#perceptual error
from PerceptualSimilarity import models

import cv2
import numpy as np

from simsg.loader_utils import build_eval_loader
from scripts.eval_utils import bbox_coordinates_with_margin, makedir, query_image_by_semantic_id, save_graph_json

CLEVR_DIR = os.path.expanduser('./dataset/clevr/target')
SPLIT = "test"

print_every = 100

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='./experiments/clevr/')
parser.add_argument('--experiment', default="spade_64_clevr", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)

parser.add_argument('--save_imgs', default=True, type=bool_flag)

parser.add_argument('--with_query_image', default=False, type=bool)

# used for relationship changes (reposition)
# do we drop the subject, object, or both from the original location?
parser.add_argument('--drop_obj', default=False, type=bool_flag)
parser.add_argument('--drop_subj', default=True, type=bool_flag)

parser.add_argument('--data_h5', default=os.path.join(CLEVR_DIR, SPLIT + '.h5'))
parser.add_argument('--data_image_dir', default=CLEVR_DIR)

args = parser.parse_args()
args.mode = "eval"
args.dataset = "clevr"

vocab_json_t = os.path.join(CLEVR_DIR, 'vocab.json')
with open(vocab_json_t, 'r') as f:
  vocab_t = json.load(f)

CLEVR_SRC_DIR = os.path.join(os.path.split(CLEVR_DIR)[0], 'source')
vocab_json = os.path.join(CLEVR_SRC_DIR, "vocab.json")

with open(vocab_json, 'r') as f:
    vocab_src = json.load(f)

output_file = os.path.join(args.exp_dir ,args.experiment + ".txt")
if args.checkpoint is None:
    args.checkpoint = os.path.join(args.exp_dir, args.experiment + "_model.pt")

def build_model(args, checkpoint):
  model = SIMSGModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model


def tripleToObjID(triples, objs):
  triples_new = []
  for [s,p,o] in triples:
    s2 = int(objs[s].cpu())
    o2 = int(objs[o].cpu())
    triples_new.append([s2, int(p.cpu()), o2])
  return triples_new


def get_triples_names(triples, vocab):
  new_triples = []
  triples = list(triples)
  for i in range(len(triples)):
    s, p, o = triples[i]
    new_triples.append([vocab['object_idx_to_name'][s], vocab['pred_idx_to_name'][p], vocab['object_idx_to_name'][o]])
  return new_triples


def get_def_dict():
  new_dict = {}
  new_dict['replace'] = []
  new_dict['reposition'] = []
  new_dict['remove'] = []
  new_dict['addition'] = []
  return new_dict


def calculate_scores(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                       perceptual_image, perceptual_roi):
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
    # print(perceptual_image, percept_all)
    percept_all_std = np.std(perceptual_image, dtype=np.float64)
    percept_roi = np.mean(perceptual_roi, dtype=np.float64)
    percept_roi_std = np.std(perceptual_roi, dtype=np.float64)
    # ------------------------

    print()
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all, mae_std))
    print('MAE-RoI: Mean {:.6f}, Std {:.6f}'.format(mae_roi, mae_roi_std))
    print('IoU: Mean {:.6f}, Std {:.6f}'.format(iou_all, iou_std))
    print('IoU-RoI: Mean {:.6f}, Std {:.6f}'.format(iou_roi, iou_roi_std))
    print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all, ssim_std))
    print('SSIM-RoI: Mean {:.6f}, Std {:.6f}'.format(ssim_roi, ssim_roi_std))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all, percept_all_std))
    print('LPIPS-RoI: Mean {:.6f}, Std {:.6f}'.format(percept_roi, percept_roi_std))

    with open(output_file, "a+") as f:
        f.write("Mean All\n")
        f.write('MAE: Mean {:.6f}, Std {:.6f}\n'.format(mae_all, mae_std))
        f.write('MAE-RoI: Mean {:.6f}, Std {:.6f}\n'.format(mae_roi, mae_roi_std))
        f.write('IoU: Mean {:.6f}, Std {:.6f}\n'.format(iou_all, iou_std))
        f.write('IoU-RoI: Mean {:.6f}, Std {:.6f}\n'.format(iou_roi, iou_roi_std))
        f.write('SSIM: Mean {:.6f}, Std {:.6f}\n'.format(ssim_all, ssim_std))
        f.write('SSIM-RoI: Mean {:.6f}, Std {:.6f}\n'.format(ssim_roi, ssim_roi_std))
        f.write('LPIPS: Mean {:.6f}, Std {:.6f}\n'.format(percept_all, percept_all_std))
        f.write('LPIPS-RoI: Mean {:.6f}, Std {:.6f}\n'.format(percept_roi, percept_roi_std))


modes = ['replace', 'reposition', 'remove', 'addition']


def calculate_scores_modes(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                     perceptual_image, perceptual_roi):
  mae_all = get_def_dict()
  mae_std = get_def_dict()
  mae_roi = get_def_dict()
  mae_roi_std = get_def_dict()
  iou_all = get_def_dict()
  iou_std = get_def_dict()
  iou_roi = get_def_dict()
  iou_roi_std = get_def_dict()
  ssim_all = get_def_dict()
  ssim_std = get_def_dict()
  ssim_roi = get_def_dict()
  ssim_roi_std = get_def_dict()
  percept_all = get_def_dict()
  percept_all_std = get_def_dict()
  percept_roi = get_def_dict()
  percept_roi_std = get_def_dict()
  for mode in modes:
    mae_all[mode] = np.mean(np.hstack(mae_per_image[mode]), dtype=np.float64)
    mae_std[mode] = np.std(np.hstack(mae_per_image[mode]), dtype=np.float64)
    mae_roi[mode] = np.mean(mae_roi_per_image[mode], dtype=np.float64)
    mae_roi_std[mode] = np.std(mae_roi_per_image[mode], dtype=np.float64)
    iou_all[mode] = np.mean(np.hstack(total_iou[mode]), dtype=np.float64)
    iou_std[mode] = np.std(np.hstack(total_iou[mode]), dtype=np.float64)
    iou_roi[mode] = np.mean(np.hstack(roi_only_iou[mode]), dtype=np.float64)
    iou_roi_std[mode] = np.std(np.hstack(roi_only_iou[mode]), dtype=np.float64)
    ssim_all[mode] = np.mean(ssim_per_image[mode], dtype=np.float64)
    ssim_std[mode] = np.std(ssim_per_image[mode], dtype=np.float64)
    ssim_roi[mode] = np.mean(ssim_rois[mode], dtype=np.float64)
    ssim_roi_std[mode] = np.std(ssim_rois[mode], dtype=np.float64)
    # percept error -----------
    percept_all[mode] = np.mean(perceptual_image[mode], dtype=np.float64)
    # print(perceptual_image, percept_all)
    percept_all_std[mode] = np.std(perceptual_image[mode], dtype=np.float64)
    percept_roi[mode] = np.mean(perceptual_roi[mode], dtype=np.float64)
    percept_roi_std[mode] = np.std(perceptual_roi[mode], dtype=np.float64)
    # ------------------------

    print(mode)
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all[mode], mae_std[mode]))
    print('MAE-RoI: Mean {:.6f}, Std {:.6f}'.format(mae_roi[mode], mae_roi_std[mode]))
    print('IoU: Mean {:.6f}, Std {:.6f}'.format(iou_all[mode], iou_std[mode]))
    print('IoU-RoI: Mean {:.6f}, Std {:.6f}'.format(iou_roi[mode], iou_roi_std[mode]))
    print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all[mode], ssim_std[mode]))
    print('SSIM-RoI: Mean {:.6f}, Std {:.6f}'.format(ssim_roi[mode], ssim_roi_std[mode]))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all[mode], percept_all_std[mode]))
    print('LPIPS-RoI: Mean {:.6f}, Std {:.6f}'.format(percept_roi[mode], percept_roi_std[mode]))

    with open(output_file, "a+") as f:
        f.write("Mode: " + mode)
        f.write('MAE: Mean {:.6f}, Std {:.6f}\n'.format(mae_all[mode], mae_std[mode]))
        f.write('MAE-RoI: Mean {:.6f}, Std {:.6f}\n'.format(mae_roi[mode], mae_roi_std[mode]))
        f.write('IoU: Mean {:.6f}, Std {:.6f}\n'.format(iou_all[mode], iou_std[mode]))
        f.write('IoU-RoI: Mean {:.6f}, Std {:.6f}\n'.format(iou_roi[mode], iou_roi_std[mode]))
        f.write('SSIM: Mean {:.6f}, Std {:.6f}\n'.format(ssim_all[mode], ssim_std[mode]))
        f.write('SSIM-RoI: Mean {:.6f}, Std {:.6f}\n'.format(ssim_roi[mode], ssim_roi_std[mode]))
        f.write('LPIPS: Mean {:.6f}, Std {:.6f}\n'.format(percept_all[mode], percept_all_std[mode]))
        f.write('LPIPS-RoI: Mean {:.6f}, Std {:.6f}\n'.format(percept_roi[mode], percept_roi_std[mode]))


def run_model(args, checkpoint, loader=None):

  output_dir = args.exp_dir
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_eval_loader(args, checkpoint, vocab_t)

  img_dir = makedir(output_dir, 'images_' + SPLIT)
  graph_json_dir = makedir(output_dir, 'graphs_json')

  f = open(output_dir + "/result_ids.txt", "w")

  img_idx = 0
  total_iou_all = []
  total_iou = get_def_dict()
  total_boxes = 0
  mae_per_image_all = []
  mae_per_image = get_def_dict()
  mae_roi_per_image_all = []
  mae_roi_per_image = get_def_dict()
  roi_only_iou_all = []
  roi_only_iou = get_def_dict()
  ssim_per_image_all = []
  ssim_per_image = get_def_dict()
  ssim_rois_all = []
  ssim_rois = get_def_dict()
  rois = 0
  margin = 2

  ## Initializing the perceptual loss model
  lpips_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True)
  perceptual_error_image_all = []
  perceptual_error_image = get_def_dict()
  perceptual_error_roi_all = []
  perceptual_error_roi = get_def_dict()

  for batch in loader:

    imgs, imgs_src, objs, objs_src, boxes, boxes_src, triples, triples_src, obj_to_img, \
        triple_to_img, imgs_in = [x.cuda() for x in batch]

    imgs_gt = imagenet_deprocess_batch(imgs_src)
    imgs_target_gt = imagenet_deprocess_batch(imgs)

    # Get mode from target scene - source scene, or image id, using sets
    graph_set_bef = Counter(tuple(row) for row in tripleToObjID(triples_src, objs_src))
    obj_set_bef = Counter([int(obj.cpu()) for obj in objs_src])
    graph_set_aft = Counter(tuple(row) for row in tripleToObjID(triples, objs))
    obj_set_aft = Counter([int(obj.cpu()) for obj in objs])

    if len(objs) > len(objs_src):
      mode = "addition"
      changes = graph_set_aft - graph_set_bef
      obj_ids = list(obj_set_aft - obj_set_bef)
      new_ids = (objs == obj_ids[0]).nonzero()
    elif len(objs) < len(objs_src):
      mode = "remove"
      changes = graph_set_bef - graph_set_aft
      obj_ids = list(obj_set_bef - obj_set_aft)
      new_ids_src = (objs_src == obj_ids[0]).nonzero()
      new_objs = [obj for obj in objs]
      new_objs.append(objs_src[new_ids_src[0]])
      objs = torch.tensor(new_objs).cuda()
      num_objs = len(objs)
      new_ids = [torch.tensor(num_objs-1)]
      new_boxes = [bbox for bbox in boxes]
      new_boxes.append(boxes_src[new_ids_src[0]][0])
      boxes = torch.stack(new_boxes)
      obj_to_img = torch.zeros(num_objs, dtype=objs.dtype, device=objs.device)
    elif torch.all(torch.eq(objs, objs_src)):
      mode = "reposition"
      changes = (graph_set_bef - graph_set_aft) + (graph_set_aft - graph_set_bef)
      idx_cnt = np.zeros((25,1))
      for [s,p,o] in list(changes):
        idx_cnt[s] += 1
        idx_cnt[o] += 1

      obj_ids = idx_cnt.argmax(0)
      id_src = (objs_src == obj_ids[0]).nonzero()
      box_src = boxes_src[id_src[0]]
      new_ids = (objs == obj_ids[0]).nonzero()
      boxes[new_ids[0]] = box_src

    elif len(objs) == len(objs_src):
      mode = "replace"
      changes = (graph_set_bef - graph_set_aft) + (graph_set_aft - graph_set_bef)
      obj_ids = [list(obj_set_bef - obj_set_aft)[0], list(obj_set_aft - obj_set_bef)[0]]
      new_ids = (objs == obj_ids[1]).nonzero()
    else:
      assert False

    new_ids = [int(new_id.cpu()) for new_id in new_ids]

    show_im = False
    if show_im:
      img_gt = imgs_gt[0].numpy().transpose(1, 2, 0)
      img_gt_target = imgs_target_gt[0].numpy().transpose(1, 2, 0)
      fig = plt.figure()
      fig.add_subplot(1, 2, 1)
      plt.imshow(img_gt)
      fig.add_subplot(1, 2, 2)
      plt.imshow(img_gt_target)
      plt.show(block=True)

    query_feats = None

    if args.with_query_image:
      img, box = query_image_by_semantic_id(new_ids, img_idx, loader)
      query_feats = model.forward_visual_feats(img, box)

      img_filename_query = '%04d_query.png' % (img_idx)
      img = imagenet_deprocess_batch(img)
      img_np = img[0].numpy().transpose(1, 2, 0).astype(np.uint8)
      img_path = os.path.join(img_dir, img_filename_query)
      imsave(img_path, img_np)


    img_gt_filename = '%04d_gt_src.png' % (img_idx)
    img_target_gt_filename = '%04d_gt_target.png' % (img_idx)
    img_pred_filename = '%04d_changed.png' % (img_idx)
    img_filename_noised = '%04d_noised.png' % (img_idx)

    triples_ = triples

    boxes_gt = boxes

    keep_box_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
    keep_feat_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
    keep_image_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)

    subject_node = new_ids[0]
    keep_image_idx[subject_node] = 0

    if mode == 'reposition':
      keep_box_idx[subject_node] = 0
    elif mode == "remove":
      keep_feat_idx[subject_node] = 0
    else:
      if mode == "replace":
        keep_feat_idx[subject_node] = 0
      if mode == 'auto_withfeats':
        keep_image_idx[subject_node] = 0

      if mode == 'auto_nofeats':
        if not args.with_query_image:
          keep_feat_idx[subject_node] = 0

    model_out = model(objs, triples_, obj_to_img,
        boxes_gt=boxes_gt, masks_gt=None, src_image=imgs_in, mode=mode,
        query_feats=query_feats, keep_box_idx=keep_box_idx, keep_feat_idx=keep_feat_idx,
        keep_image_idx=keep_image_idx)

    imgs_pred, boxes_pred_o, masks_pred, noised_srcs, _ = model_out

    imgs = imagenet_deprocess_batch(imgs).float()
    imgs_pred = imagenet_deprocess_batch(imgs_pred).float()

    #Metrics

    # IoU over all
    curr_iou = jaccard(boxes_pred_o, boxes).detach().cpu().numpy()
    total_iou_all.append(curr_iou)
    total_iou[mode].append(curr_iou)
    total_boxes += boxes_pred_o.size(0)

    # IoU over targets only
    pred_dropbox = boxes_pred_o[keep_box_idx.squeeze() == 0, :]
    gt_dropbox = boxes[keep_box_idx.squeeze() == 0, :]
    curr_iou_roi = jaccard(pred_dropbox, gt_dropbox).detach().cpu().numpy()
    roi_only_iou_all.append(curr_iou_roi)
    roi_only_iou[mode].append(curr_iou_roi)
    rois += pred_dropbox.size(0)

    # MAE per image
    curr_mae = torch.mean(
      torch.abs(imgs - imgs_pred).view(imgs.shape[0], -1), 1).cpu().numpy()
    mae_per_image[mode].append(curr_mae)
    mae_per_image_all.append(curr_mae)

    for s in range(imgs.shape[0]):
      # get coordinates of target
      left, right, top, bottom = bbox_coordinates_with_margin(boxes[s, :], margin, imgs)
      if left > right or top > bottom:
        continue
      # print("bboxes with margin: ", left, right, top, bottom)

      # calculate errors only in RoI one by one
      curr_mae_roi = torch.mean(
        torch.abs(imgs[s, :, top:bottom, left:right] - imgs_pred[s, :, top:bottom, left:right])).cpu().item()
      mae_roi_per_image[mode].append(curr_mae_roi)
      mae_roi_per_image_all.append(curr_mae_roi)

      curr_ssim = pytorch_ssim.ssim(imgs[s:s + 1, :, :, :] / 255.0,
                          imgs_pred[s:s + 1, :, :, :] / 255.0, window_size=3).cpu().item()
      ssim_per_image_all.append(curr_ssim)
      ssim_per_image[mode].append(curr_ssim)

      curr_ssim_roi = pytorch_ssim.ssim(imgs[s:s + 1, :, top:bottom, left:right] / 255.0,
                          imgs_pred[s:s + 1, :, top:bottom, left:right] / 255.0, window_size=3).cpu().item()
      ssim_rois_all.append(curr_ssim_roi)
      ssim_rois[mode].append(curr_ssim_roi)

      imgs_pred_norm = imgs_pred[s:s + 1, :, :, :] / 127.5 - 1
      imgs_gt_norm = imgs[s:s + 1, :, :, :] / 127.5 - 1

      curr_lpips = lpips_model.forward(imgs_pred_norm, imgs_gt_norm).detach().cpu().numpy()
      perceptual_error_image_all.append(curr_lpips)
      perceptual_error_image[mode].append(curr_lpips)

    for i in range(imgs_pred.size(0)):

      if args.save_imgs:
        img_gt = imgs_gt[i].numpy().transpose(1, 2, 0).astype(np.uint8)
        img_gt = cv2.resize(img_gt, (128, 128))
        img_gt_path = os.path.join(img_dir, img_gt_filename)
        imsave(img_gt_path, img_gt)

        img_gt_target = imgs_target_gt[i].numpy().transpose(1, 2, 0).astype(np.uint8)
        img_gt_target = cv2.resize(img_gt_target, (128, 128))
        img_gt_target_path = os.path.join(img_dir, img_target_gt_filename)
        imsave(img_gt_target_path, img_gt_target)

        noised_src_np = imagenet_deprocess_batch(noised_srcs[:, :3, :, :])
        noised_src_np = noised_src_np[i].numpy().transpose(1, 2, 0).astype(np.uint8)
        noised_src_np = cv2.resize(noised_src_np, (128, 128))
        img_path_noised = os.path.join(img_dir, img_filename_noised)
        imsave(img_path_noised, noised_src_np)

        img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pred_np = cv2.resize(img_pred_np, (128, 128))
        img_path = os.path.join(img_dir, img_pred_filename)
        imsave(img_path, img_pred_np)

      save_graph_json(objs, triples, boxes, "after", graph_json_dir, img_idx)


    img_idx += 1

    if img_idx % print_every == 0:
      calculate_scores(mae_per_image_all, mae_roi_per_image_all, total_iou_all, roi_only_iou_all, ssim_per_image_all,
                       ssim_rois_all, perceptual_error_image_all, perceptual_error_roi_all)
      calculate_scores_modes(mae_per_image, mae_roi_per_image, total_iou, roi_only_iou, ssim_per_image, ssim_rois,
                       perceptual_error_image, perceptual_error_roi)

    print('Saved %d images' % img_idx)

  f.close()


def main(args):

  got_checkpoint = args.checkpoint is not None

  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint)
  else:
    print('--checkpoint not specified')


if __name__ == '__main__':
  main(args)
