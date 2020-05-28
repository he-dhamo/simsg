#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo
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
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np
import h5py
import PIL

from .utils import imagenet_preprocess, Resize


class SceneGraphNoPairsDataset(Dataset):
  def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True,
               mode='train', clean_repeats=True, predgraphs=False):
    super(SceneGraphNoPairsDataset, self).__init__()

    assert mode in ["train", "eval", "auto_withfeats", "auto_nofeats", "reposition", "remove", "replace"]

    self.mode = mode

    self.image_dir = image_dir
    self.image_size = image_size
    self.vocab = vocab
    self.num_objects = len(vocab['object_idx_to_name'])
    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships

    self.evaluating = mode != 'train'
    self.predgraphs = predgraphs

    if self.mode == 'reposition':
      self.use_orphaned_objects = False

    self.clean_repeats = clean_repeats

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

    self.data = {}
    with h5py.File(h5_path, 'r') as f:
      for k, v in f.items():
        if k == 'image_paths':
          self.image_paths = list(v)
        else:
          self.data[k] = torch.IntTensor(np.asarray(v))

  def __len__(self):
    num = self.data['object_names'].size(0)
    if self.max_samples is not None:
      return min(self.max_samples, num)
    return num

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    - objs: LongTensor of shape (num_objs,)
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    img_path = os.path.join(self.image_dir, self.image_paths[index])

    # use for the mix strings and bytes error
    #img_path = os.path.join(self.image_dir, self.image_paths[index].decode("utf-8"))

    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        #print(WW, HH)
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size

    # Figure out which objects appear in relationships and which don't
    obj_idxs_with_rels = set()
    obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
    for r_idx in range(self.data['relationships_per_image'][index]):
      s = self.data['relationship_subjects'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      obj_idxs_with_rels.add(s)
      obj_idxs_with_rels.add(o)
      obj_idxs_without_rels.discard(s)
      obj_idxs_without_rels.discard(o)

    obj_idxs = list(obj_idxs_with_rels)
    obj_idxs_without_rels = list(obj_idxs_without_rels)
    if len(obj_idxs) > self.max_objects - 1:
      if self.evaluating:
        obj_idxs = obj_idxs[:self.max_objects]
      else:
        obj_idxs = random.sample(obj_idxs, self.max_objects)
    if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels))
      if self.evaluating:
        obj_idxs += obj_idxs_without_rels[:num_to_add]
      else:
        obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
    if len(obj_idxs) == 0 and not self.use_orphaned_objects:
      # avoid empty list of objects
      obj_idxs += obj_idxs_without_rels[:1]
    map_overlapping_obj = {}

    objs = []
    boxes = []

    obj_idx_mapping = {}
    counter = 0
    for i, obj_idx in enumerate(obj_idxs):

      curr_obj = self.data['object_names'][index, obj_idx].item()
      x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()

      x0 = float(x) / WW
      y0 = float(y) / HH
      if self.predgraphs:
        x1 = float(w) / WW
        y1 = float(h) / HH
      else:
        x1 = float(x + w) / WW
        y1 = float(y + h) / HH

      curr_box = torch.FloatTensor([x0, y0, x1, y1])

      found_overlap = False
      if self.predgraphs:
        for prev_idx in range(counter):
          if overlapping_nodes(objs[prev_idx], curr_obj, boxes[prev_idx], curr_box):
            map_overlapping_obj[i] = prev_idx
            found_overlap = True
            break
      if not found_overlap:

        objs.append(curr_obj)
        boxes.append(curr_box)
        map_overlapping_obj[i] = counter
        counter += 1

      obj_idx_mapping[obj_idx] = map_overlapping_obj[i]

    # The last object will be the special __image__ object
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))

    boxes = torch.stack(boxes)
    objs = torch.LongTensor(objs)
    num_objs = counter + 1

    triples = []
    for r_idx in range(self.data['relationships_per_image'][index].item()):
      if not self.include_relationships:
        break
      s = self.data['relationship_subjects'][index, r_idx].item()
      p = self.data['relationship_predicates'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      s = obj_idx_mapping.get(s, None)
      o = obj_idx_mapping.get(o, None)
      if s is not None and o is not None:
        if self.clean_repeats and [s, p, o] in triples:
          continue
        if self.predgraphs and s == o:
          continue
        triples.append([s, p, o])

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(num_objs - 1):
      triples.append([i, in_image, num_objs - 1])

    triples = torch.LongTensor(triples)
    return image, objs, boxes, triples


def collate_fn_nopairs(batch):
  """
  Collate function to be used when wrapping a SceneGraphNoPairsDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, 3, H, W)
  - objs: LongTensor of shape (num_objs,) giving categories for all objects
  - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
  - triples: FloatTensor of shape (num_triples, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n]
  - imgs_masked: FloatTensor of shape (N, 4, H, W)
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []

  all_imgs_masked = []

  obj_offset = 0

  for i, (img, objs, boxes, triples) in enumerate(batch):

    all_imgs.append(img[None])
    num_objs, num_triples = objs.size(0), triples.size(0)

    all_objs.append(objs)
    all_boxes.append(boxes)
    triples = triples.clone()

    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset

    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
    all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))

    # prepare input 4-channel image
    # initialize mask channel with zeros
    masked_img = img.clone()
    mask = torch.zeros_like(masked_img)
    mask = mask[0:1,:,:]
    masked_img = torch.cat([masked_img, mask], 0)
    all_imgs_masked.append(masked_img[None])

    obj_offset += num_objs

  all_imgs_masked = torch.cat(all_imgs_masked)

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  return all_imgs, all_objs, all_boxes, all_triples, \
         all_obj_to_img, all_triple_to_img, all_imgs_masked


from simsg.model import get_left_right_top_bottom


def overlapping_nodes(obj1, obj2, box1, box2, criteria=0.7):
  # used to clean predicted graphs - merge nodes with overlapping boxes
  # are these two objects overplapping?
  # boxes given as [left, top, right, bottom]
  res = 100 # used to project box representation in 2D for iou computation
  epsilon = 0.001
  if obj1 == obj2:
    spatial_box1 = np.zeros([res, res])
    left, right, top, bottom = get_left_right_top_bottom(box1, res, res)
    spatial_box1[top:bottom, left:right] = 1
    spatial_box2 = np.zeros([res, res])
    left, right, top, bottom = get_left_right_top_bottom(box2, res, res)
    spatial_box2[top:bottom, left:right] = 1
    iou = np.sum(spatial_box1 * spatial_box2) / \
          (np.sum((spatial_box1 + spatial_box2 > 0).astype(np.float32)) + epsilon)
    return iou >= criteria
  else:
    return False
