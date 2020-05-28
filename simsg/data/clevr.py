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

import os

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np
import h5py, json
import PIL

from .utils import imagenet_preprocess, Resize
sg_task = True

def conv_src_to_target(voc_s, voc_t):
  dic = {}
  for k, val in voc_s['object_name_to_idx'].items():
    dic[val] = voc_t['object_name_to_idx'][k]
  return dic


class SceneGraphWithPairsDataset(Dataset):
  def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True,
               mode='train', clean_repeats=True):
    super(SceneGraphWithPairsDataset, self).__init__()

    assert mode in ["train", "eval", "auto_withfeats", "auto_nofeats", "reposition", "remove", "replace"]

    CLEVR_target_dir = os.path.split(h5_path)[0]
    CLEVR_SRC_DIR = os.path.join(os.path.split(CLEVR_target_dir)[0], 'source')

    vocab_json_s = os.path.join(CLEVR_SRC_DIR, "vocab.json")
    vocab_json_t = os.path.join(CLEVR_target_dir, "vocab.json")

    with open(vocab_json_s, 'r') as f:
      vocab_src = json.load(f)

    with open(vocab_json_t, 'r') as f:
      vocab_t = json.load(f)

    self.mode = mode

    self.image_dir = image_dir
    self.image_source_dir = os.path.join(os.path.split(image_dir)[0], 'source') #Azade

    src_h5_path = os.path.join(self.image_source_dir, os.path.split(h5_path)[-1])
    print(self.image_dir, src_h5_path)

    self.image_size = image_size
    self.vocab = vocab
    self.vocab_src = vocab_src
    self.vocab_t = vocab_t
    self.num_objects = len(vocab['object_idx_to_name'])
    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships

    self.evaluating = mode != 'train'

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

    self.data_src = {}
    with h5py.File(src_h5_path, 'r') as f:
      for k, v in f.items():
        if k == 'image_paths':
          self.image_paths_src = list(v)
        else:
          self.data_src[k] = torch.IntTensor(np.asarray(v))

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
    img_source_path = os.path.join(self.image_source_dir, self.image_paths[index])

    src_to_target_obj = conv_src_to_target(self.vocab_src, self.vocab_t)

    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    with open(img_source_path, 'rb') as f:
      with PIL.Image.open(f) as image_src:
        #WW, HH = image.size
        image_src = self.transform(image_src.convert('RGB'))

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
      obj_idxs = obj_idxs[:self.max_objects]
    if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels))
      obj_idxs += obj_idxs_without_rels[:num_to_add]

    num_objs = len(obj_idxs) + 1

    objs = torch.LongTensor(num_objs).fill_(-1)

    boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(num_objs, 1)
    obj_idx_mapping = {}
    for i, obj_idx in enumerate(obj_idxs):
      objs[i] = self.data['object_names'][index, obj_idx].item()
      x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
      x0 = float(x) / WW
      y0 = float(y) / HH
      x1 = float(x + w) / WW
      y1 = float(y + h) / HH
      boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
      obj_idx_mapping[obj_idx] = i

    # The last object will be the special __image__ object
    objs[num_objs - 1] = self.vocab['object_name_to_idx']['__image__']

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
        triples.append([s, p, o])

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(num_objs - 1):
      triples.append([i, in_image, num_objs - 1])

    triples = torch.LongTensor(triples)

    #Source image

    # Figure out which objects appear in relationships and which don't
    obj_idxs_with_rels_src = set()
    obj_idxs_without_rels_src = set(range(self.data_src['objects_per_image'][index].item()))
    for r_idx in range(self.data_src['relationships_per_image'][index]):
      s = self.data_src['relationship_subjects'][index, r_idx].item()
      o = self.data_src['relationship_objects'][index, r_idx].item()
      obj_idxs_with_rels_src.add(s)
      obj_idxs_with_rels_src.add(o)
      obj_idxs_without_rels_src.discard(s)
      obj_idxs_without_rels_src.discard(o)

    obj_idxs_src = list(obj_idxs_with_rels_src)
    obj_idxs_without_rels_src = list(obj_idxs_without_rels_src)
    if len(obj_idxs_src) > self.max_objects - 1:
      obj_idxs_src = obj_idxs_src[:self.max_objects]
    if len(obj_idxs_src) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs_src)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels_src))
      obj_idxs_src += obj_idxs_without_rels_src[:num_to_add]

    num_objs_src = len(obj_idxs_src) + 1

    objs_src = torch.LongTensor(num_objs_src).fill_(-1)

    boxes_src = torch.FloatTensor([[0, 0, 1, 1]]).repeat(num_objs_src, 1)
    obj_idx_mapping_src = {}
    for i, obj_idx in enumerate(obj_idxs_src):
      objs_src[i] = src_to_target_obj[self.data_src['object_names'][index, obj_idx].item()]
      x, y, w, h = self.data_src['object_boxes'][index, obj_idx].tolist()
      x0 = float(x) / WW
      y0 = float(y) / HH
      x1 = float(x + w) / WW
      y1 = float(y + h) / HH
      boxes_src[i] = torch.FloatTensor([x0, y0, x1, y1])
      obj_idx_mapping_src[obj_idx] = i

    # The last object will be the special __image__ object
    objs_src[num_objs_src - 1] = self.vocab_src['object_name_to_idx']['__image__']

    triples_src = []
    for r_idx in range(self.data_src['relationships_per_image'][index].item()):
      if not self.include_relationships:
        break
      s = self.data_src['relationship_subjects'][index, r_idx].item()
      p = self.data_src['relationship_predicates'][index, r_idx].item()
      o = self.data_src['relationship_objects'][index, r_idx].item()
      s = obj_idx_mapping_src.get(s, None)
      o = obj_idx_mapping_src.get(o, None)
      if s is not None and o is not None:
        if self.clean_repeats and [s, p, o] in triples_src:
          continue
        triples_src.append([s, p, o])

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab_src['pred_name_to_idx']['__in_image__']
    for i in range(num_objs_src - 1):
      triples_src.append([i, in_image, num_objs_src - 1])

    triples_src = torch.LongTensor(triples_src)

    return image, image_src, objs, objs_src, boxes, boxes_src, triples, triples_src


def collate_fn_withpairs(batch):
  """
  Collate function to be used when wrapping a SceneGraphWithPairsDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs, imgs_src: target and source FloatTensors of shape (N, C, H, W)
  - objs, objs_src: target and source LongTensors of shape (num_objs,) giving categories for all objects
  - boxes, boxes_src: target and source FloatTensors of shape (num_objs, 4) giving boxes for all objects
  - triples, triples_src: target and source FloatTensors of shape (num_triples, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (num_triples,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n]
  - imgs_masked: FloatTensor of shape (N, 4, H, W)
  """
  # batch is a list, and each element is (image, objs, boxes, triples)
  all_imgs, all_imgs_src, all_objs, all_objs_src, all_boxes, all_boxes_src, all_triples, all_triples_src = [], [], [], [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  all_imgs_masked = []

  obj_offset = 0

  for i, (img, image_src, objs, objs_src, boxes, boxes_src, triples, triples_src) in enumerate(batch):

    all_imgs.append(img[None])
    all_imgs_src.append(image_src[None])
    num_objs, num_triples = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_objs_src.append(objs_src)
    all_boxes.append(boxes)
    all_boxes_src.append(boxes_src)
    triples = triples.clone()
    triples_src = triples_src.clone()

    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    triples_src[:, 0] += obj_offset
    triples_src[:, 2] += obj_offset
    all_triples_src.append(triples_src)

    all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
    all_triple_to_img.append(torch.LongTensor(num_triples).fill_(i))

    # prepare input 4-channel image
    # initialize mask channel with zeros
    masked_img = image_src.clone()
    mask = torch.zeros_like(masked_img)
    mask = mask[0:1,:,:]
    masked_img = torch.cat([masked_img, mask], 0)
    all_imgs_masked.append(masked_img[None])

    obj_offset += num_objs

  all_imgs_masked = torch.cat(all_imgs_masked)

  all_imgs = torch.cat(all_imgs)
  all_imgs_src = torch.cat(all_imgs_src)
  all_objs = torch.cat(all_objs)
  all_objs_src = torch.cat(all_objs_src)
  all_boxes = torch.cat(all_boxes)
  all_boxes_src = torch.cat(all_boxes_src)
  all_triples = torch.cat(all_triples)
  all_triples_src = torch.cat(all_triples_src)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_imgs_src, all_objs, all_objs_src, all_boxes, all_boxes_src, all_triples, all_triples_src,
         all_obj_to_img, all_triple_to_img, all_imgs_masked)

  return out
