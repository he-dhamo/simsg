#!/usr/bin/python
#
# Copyright 2020 Helisa Dhamo
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

import os
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from simsg.vis import draw_scene_graph
from simsg.data import imagenet_deprocess_batch
from imageio import imsave

from simsg.model import mask_image_in_bbox


def remove_node(objs, triples, boxes, imgs, idx, obj_to_img, triple_to_img):
  '''
  removes nodes and all related edges in case of object removal
  image is also masked in the respective area
  idx: list of object ids to be removed
  Returns:
    updated objs, triples, boxes, imgs, obj_to_img, triple_to_img
  '''

  # object nodes
  idlist = list(range(objs.shape[0]))
  keeps = [i for i in idlist if i not in idx]
  objs_reduced = objs[keeps]
  boxes_reduced = boxes[keeps]

  offset = torch.zeros_like(objs)
  for i in range(objs.shape[0]):
    for j in idx:
      if j < i:
        offset[i] += 1

  # edges connected to removed object
  keeps_t = []
  triples_reduced = []
  for i in range(triples.shape[0]):
    if not(triples[i,0] in idx or triples[i, 2] in idx):
      keeps_t.append(i)
      triples_reduced.append(torch.tensor([triples[i,0] - offset[triples[i,0]], triples[i,1],
                                           triples[i,2] - offset[triples[i,2]]], device=triples.device))
  triples_reduced = torch.stack(triples_reduced, dim=0)

  # update indexing arrays
  obj_to_img_reduced = obj_to_img[keeps]
  triple_to_img_reduced = triple_to_img[keeps_t]

  # mask RoI of removed objects from image
  for i in idx:
    imgs = mask_image_in_bbox(imgs, boxes, i, obj_to_img, mode='removal')

  return objs_reduced, triples_reduced, boxes_reduced, imgs, obj_to_img_reduced, triple_to_img_reduced


def bbox_coordinates_with_margin(bbox, margin, img):
    # extract bounding box with a margin

    left = max(0, bbox[0] * img.shape[3] - margin)
    top = max(0, bbox[1] * img.shape[2] - margin)
    right = min(img.shape[3], bbox[2] * img.shape[3] + margin)
    bottom = min(img.shape[2], bbox[3] * img.shape[2] + margin)

    return int(left), int(right), int(top), int(bottom)


def save_image_from_tensor(img, img_dir, filename):

    img = imagenet_deprocess_batch(img)
    img_np = img[0].numpy().transpose(1, 2, 0)
    img_path = os.path.join(img_dir, filename)
    imsave(img_path, img_np)


def save_image_with_label(img_pred, img_gt, img_dir, filename, txt_str):
    # saves gt and generated image, concatenated
    # together with text label describing the change
    # used for easier visualization of results

    img_pred = imagenet_deprocess_batch(img_pred)
    img_gt = imagenet_deprocess_batch(img_gt)

    img_pred_np = img_pred[0].numpy().transpose(1, 2, 0)
    img_gt_np = img_gt[0].numpy().transpose(1, 2, 0)

    img_pred_np = cv2.resize(img_pred_np, (128, 128))
    img_gt_np = cv2.resize(img_gt_np, (128, 128))

    wspace = np.zeros([img_pred_np.shape[0], 10, 3])
    text = np.zeros([30, img_pred_np.shape[1] * 2 + 10, 3])
    text = cv2.putText(text, txt_str, (0,20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.5, (255, 255, 255), lineType=cv2.LINE_AA)

    img_pred_gt = np.concatenate([img_gt_np, wspace, img_pred_np], axis=1).astype('uint8')
    img_pred_gt = np.concatenate([text, img_pred_gt], axis=0).astype('uint8')
    img_path = os.path.join(img_dir, filename)
    imsave(img_path, img_pred_gt)


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def save_graph_json(objs, triples, boxes, beforeafter, dir, idx):
    # save scene graph in json form

    data = {}
    objs = objs.cpu().numpy()
    triples = triples.cpu().numpy()
    data['objs'] = objs.tolist()
    data['triples'] = triples.tolist()
    data['boxes'] = boxes.tolist()
    with open(dir + '/' + beforeafter + '_' + str(idx) + '.json', 'w') as outfile:
        json.dump(data, outfile)


def query_image_by_semantic_id(obj_id, curr_img_id, loader, num_samples=7):
    # used to replace objects with an object of the same category and different appearance
    # return list of images and bboxes, that contain object of category obj_id

    query_imgs, query_boxes = [], []
    loader_id = 0
    counter = 0

    for l in loader:
        # load images
        imgs, objs, boxes, _, _, _, _ = [x.cuda() for x in l]
        if loader_id == curr_img_id:
            loader_id += 1
            continue

        for i, ob in enumerate(objs):
            if obj_id[0] == ob:
                query_imgs.append(imgs)
                query_boxes.append(boxes[i])
                counter += 1
            if counter == num_samples:
                return query_imgs, query_boxes
        loader_id += 1

    return 0, 0


def draw_image_box(img, box):

    left, right, top, bottom = int(round(box[0] * img.shape[1])), int(round(box[2] * img.shape[1])), \
                               int(round(box[1] * img.shape[0])), int(round(box[3] * img.shape[0]))

    cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 1)
    return img


def draw_image_edge(img, box1, box2):
    # draw arrow that connects two objects centroids
    left1, right1, top1, bottom1 = int(round(box1[0] * img.shape[1])), int(round(box1[2] * img.shape[1])), \
                               int(round(box1[1] * img.shape[0])), int(round(box1[3] * img.shape[0]))
    left2, right2, top2, bottom2 = int(round(box2[0] * img.shape[1])), int(round(box2[2] * img.shape[1])), \
                               int(round(box2[1] * img.shape[0])), int(round(box2[3] * img.shape[0]))

    cv2.arrowedLine(img, (int((left1+right1)/2), int((top1+bottom1)/2)),
             (int((left2+right2)/2), int((top2+bottom2)/2)), (255,0,0), 1)

    return img


def visualize_imgs_boxes(imgs, imgs_pred, boxes, boxes_pred):

    nrows = imgs.size(0)
    imgs = imgs.detach().cpu().numpy()
    imgs_pred = imgs_pred.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    boxes_pred = boxes_pred.detach().cpu().numpy()
    plt.figure()

    for i in range(0, nrows):
        # i = j//2
        ax1 = plt.subplot(2, nrows, i+1)
        img = np.transpose(imgs[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(img)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_gt = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax1.add_patch(bbox_gt)
        plt.axis('off')

        ax2 = plt.subplot(2, nrows, i+nrows+1)
        pred = np.transpose(imgs_pred[i, :, :, :], (1, 2, 0)) / 255.
        plt.imshow(pred)

        left, right, top, bottom = bbox_coordinates_with_margin(boxes_pred[i, :], 0, imgs[i:i+1, :, :, :])
        bbox_pr = patches.Rectangle((left, top),
                                    width=right-left,
                                    height=bottom-top,
                                    linewidth=1, edgecolor='r', facecolor='none')
        # ax2.add_patch(bbox_gt)
        ax2.add_patch(bbox_pr)
        plt.axis('off')

    plt.show()


def visualize_scene_graphs(obj_to_img, objs, triples, vocab, device):
    offset = 0
    for i in range(1):#imgs_in.size(0)):
        curr_obj_idx = (obj_to_img == i).nonzero()

        objs_vis = objs[curr_obj_idx]
        triples_vis = []
        for j in range(triples.size(0)):
            if triples[j, 0] in curr_obj_idx or triples[j, 2] in curr_obj_idx:
                triples_vis.append(triples[j].to(device) - torch.tensor([offset, 0, offset]).to(device))
        offset += curr_obj_idx.size(0)
        triples_vis = torch.stack(triples_vis, 0)

        print(objs_vis, triples_vis)
        graph_img = draw_scene_graph(objs_vis, triples_vis, vocab)

        cv2.imshow('graph' + str(i), graph_img)
    cv2.waitKey(10000)


def remove_duplicates(triples, triple_to_img, indexes):
    # removes duplicates in relationship triples

    triples_new = []
    triple_to_img_new = []

    for i in range(triples.size(0)):
        if i not in indexes:
            triples_new.append(triples[i])
            triple_to_img_new.append(triple_to_img[i])

    triples_new = torch.stack(triples_new, 0)
    triple_to_img_new = torch.stack(triple_to_img_new, 0)

    return triples_new, triple_to_img_new


def parse_bool(pred_graphs, generative, use_gt_boxes, use_feats):
    # returns name of output directory depending on arguments

    if pred_graphs:
        name = "pred/"
    else:
        name = ""
    if generative: # fully generative mode
        return name + "generative"
    else:
        if use_gt_boxes:
            b = "withbox"
        else:
            b = "nobox"
        if use_feats:
            f = "withfeats"
        else:
            f = "nofeats"

        return name + b + "_" + f


def is_background(label_id):

    if label_id in [169, 60, 61, 49, 141, 8, 11, 52, 66]:
        return True
    else:
        return False


def get_selected_objects():

  objs = ["", "apple", "ball", "banana", "beach", "bike", "bird", "bus", "bush", "cat", "car", "chair", "cloud", "dog",
          "elephant", "field", "giraffe", "man", "motorcycle", "ocean", "person", "plane", "sheep", "tree", "zebra"]

  return objs
