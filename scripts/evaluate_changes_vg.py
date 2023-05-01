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

"""
This script can be used to generate images with changes for Visual Genome for evaluation.
"""

import argparse, json
import os

import torch

from imageio import imsave

from simsg.model import SIMSGModel
from simsg.utils import int_tuple, bool_flag
from simsg.vis import draw_scene_graph

import cv2
import numpy as np

from simsg.loader_utils import build_eval_loader
from scripts.eval_utils import makedir, query_image_by_semantic_id, save_graph_json, \
  remove_duplicates, save_image_from_tensor, save_image_with_label, is_background, remove_node

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', default='./experiments/vg/')
parser.add_argument('--experiment', default="spade_vg", type=str)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--predgraphs', default=True, type=bool_flag)
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)
parser.add_argument('--save_graph_image', default=False, type=bool_flag)
parser.add_argument('--save_graph_json', default=False, type=bool_flag) # before and after
parser.add_argument('--with_query_image', default=False, type=bool)
parser.add_argument('--mode', default='remove',
                    choices=['auto_withfeats', 'auto_nofeats', 'replace', 'reposition', 'remove'])
# fancy image save that also visualizes a text label describing the change
parser.add_argument('--label_saved_image', default=True, type=bool_flag)
# used for relationship changes (reposition)
# do we drop the subject, object, or both from the original location?
parser.add_argument('--drop_obj', default=False, type=bool_flag)
parser.add_argument('--drop_subj', default=True, type=bool_flag)
# used for object replacement
# if True, the position of the original object is kept (gt) while the size (H, W) comes from the predicted box
# recommended to set to True when replacing objects (e.g. bus to car),
# and to False for background change (e.g. ocean to field)
parser.add_argument('--combined_gt_pred_box', default=True, type=bool_flag)
# use with mode auto_nofeats to generate diverse objects when features phi are masked/dropped
parser.add_argument('--random_feats', default=False, type=bool_flag)

VG_DIR = os.path.expanduser('./datasets/vg')
SPLIT = "test"
parser.add_argument('--data_h5', default=os.path.join(VG_DIR, SPLIT + '.h5'))
parser.add_argument('--data_image_dir',
        default=os.path.join(VG_DIR, 'images'))

args = parser.parse_args()
args.dataset = "vg"

if args.predgraphs and SPLIT == "test":
    SPLIT = 'test_predgraphs'
    args.data_h5 = os.path.join(VG_DIR, SPLIT + '.h5')

if args.checkpoint is None:
    args.checkpoint = os.path.join(args.exp_dir, args.experiment + "_model.pt")

def remove_vgg(model_state):
  def filt(pair):
    key, val = pair
    return "high_level_feat" not in key

  return dict(filter(filt, model_state.items()))


def build_model(args, checkpoint):
  model = SIMSGModel(**checkpoint['model_kwargs'])
  new_state = remove_vgg(checkpoint['model_state'])
  model.load_state_dict(new_state, strict=False)
  model.eval()
  model.image_size = args.image_size
  model.cuda()
  return model


def change_relationship(objs, triples, with_auto=False):
  """
  automatically change predicate id
  - objs: Tensor, all objects in scene graph
  - triples: Tensor, all triples in scene graph
  - with_auto: add source id in the list of target ids, to test autoencoding  with masked box coords
  Returns list of [changed index, changed id] pairs
  """

  valid_triples = []
  mapping = {
    31: [2, 15, 22],  # riding -> next to, near, beside
    13: [36, 40],   # sitting in -> standing in, standing on
    22: [1]   # near -> on
  }

  for j in range(triples.size(0)):
    source_id = triples[j,1]

    # if _image_ not one of the objects and predicate is the type we want
    if triples[j,0] != objs.size(0)-1 and triples[j,2] != objs.size(0)-1 \
            and source_id in list(mapping.keys()):

      for target_id in mapping[source_id.item()]:
        valid_triples.append([j, target_id])
      if with_auto:
        valid_triples.append([j, source_id])

  return valid_triples


def change_id_constrained(id, box, num_samples=7):
  """
  automatically change object id, mapping given class to a class with similar size
  uses bbox to contrain the mapping, based on how close the object is
  returns list of changed ids for object replacement, of max size [num_samples]
  """
  objects = [6, 129, 116, 57, 127, 137, 130] # animals, people, etc
  bg = [169, 60, 61, 141] # backgrounds, like ocean, field, road, etc
  vehicles = [100, 19, 70, 143] # bus, car, boat, motorcycle
  sky_obj = [21, 80]

  if (id in objects or id == 20 or id == 3 or id == 58) and (box[2] - box[0] < 0.3):

    if id in objects:
      objects.remove(id)

    new_ids = np.random.choice(objects, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  elif id in bg:
    bg.remove(id)
    new_ids = np.random.choice(bg, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  elif id in vehicles and ((box[2] - box[0]) + (box[3] - box[1]) < 0.5):
    vehicles.remove(id)
    new_ids = np.random.choice(vehicles, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  elif id == 176 and ((box[2] - box[0]) + (box[3] - box[1]) < 0.1):
    new_ids = np.random.choice(sky_obj, num_samples)
    new_ids = list(dict.fromkeys(new_ids))

  else:
    new_ids = []

  return new_ids


def change_id(id, num_samples=7):
  """
  automatically change object id, mapping given class to a class with similar size
  returns list of changed ids for object replacement, of max size [num_samples]
  """
  obj_pool = [75, 58, 127, 165, 159, 129, 116, 57, 35, 3, 20, 6, 9] # animals, people, chairs, etc
  vehicle_pool = [19, 100, 70, 143, 78, 171] # bus, car, boat, motorcycle
  background_pool = [169, 60, 49, 141, 61]
  sky_pool = [21, 80, 176]

  if id in obj_pool:
    new_ids = np.random.choice(obj_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
    print(new_ids)
  elif id in vehicle_pool:
    new_ids = np.random.choice(vehicle_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  elif id in background_pool:
    new_ids = np.random.choice(background_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  elif id in sky_pool:
    new_ids = np.random.choice(sky_pool, num_samples)
    new_ids = list(dict.fromkeys(new_ids))
  else:
    new_ids = []

  # remove entries that map to the original/source id
  if id in new_ids:
    new_ids.remove(id)

  return new_ids


def run_model(args, checkpoint, output_dir, loader=None):
  vocab = checkpoint['model_kwargs']['vocab']
  model = build_model(args, checkpoint)
  if loader is None:
    loader = build_eval_loader(args, checkpoint)

  img_dir = makedir(output_dir, 'images_' + SPLIT)
  graph_json_dir = makedir(output_dir, 'graphs_json')
  graph_dir = makedir(output_dir, 'graphs', args.save_graph_image)

  f = open(output_dir + "/result_ids.txt", "w")

  img_idx = 0

  assert args.mode in ['auto_withfeats', 'auto_nofeats', 'reposition', 'replace', 'remove']

  for batch in loader:

    imgs_gt, objs, boxes, triples, obj_to_img, triple_to_img, imgs_in = [x.cuda() for x in batch]
    objs_before, boxes_before, triples_before = objs.clone(), boxes.clone(), triples.clone()

    subject_node = 0 # node which will be changed

    if args.mode == 'replace':

      if boxes[0, 2] - boxes[0, 0] < 0.1 or boxes[0, 3] - boxes[0, 1] < 0.15:
        img_idx += 1
        continue

      new_ids = change_id_constrained(objs[subject_node], boxes[subject_node])

    elif args.mode == 'reposition':

      new_ids = change_relationship(objs, triples)

    elif args.mode == 'remove':
      idx = [subject_node]

      id_removed = objs[idx[0]].item()
      box_removed = boxes[idx[0]]

      # ignore small boxes - skip image if removed box is too small
      if box_removed[3] - box_removed[1] < 0.2 or \
        box_removed[2] - box_removed[0] < 0.2 or \
        (box_removed[3] - box_removed[1] > 0.8 and box_removed[2] - box_removed[0] > 0.8):

        img_idx += 1
        continue

      # don't remove background or the last remaining node
      if objs[idx[0]] in [169, 60, 49, 141, 61] or objs.shape[0] <= 2:
        continue

      objs, triples, boxes, imgs_in, obj_to_img, triple_to_img = remove_node(objs, triples, boxes, imgs_in, idx,
                                                                          obj_to_img, triple_to_img)
      new_ids = [objs[subject_node]]

    else: # auto mode
      new_ids = [objs[subject_node]]
      if args.mode == 'auto_nofeats' and args.random_feats:
        # generate multiple images to get diverse features
        num_samples = 5
        new_ids = num_samples * new_ids # replicate

    query_feats = None

    if args.with_query_image:
      query_imgs, query_boxes = query_image_by_semantic_id(new_ids, img_idx, loader)
      new_ids = new_ids * len(query_imgs) # replicate

    img_subid = 0

    for new_id in new_ids:

      if args.with_query_image:
        img = query_imgs[img_subid]
        box = query_boxes[img_subid]
        query_feats = model.forward_visual_feats(img, box)

        img_filename_query = '%04d_%d_query.png' % (img_idx, img_subid)
        save_image_from_tensor(img, img_dir, img_filename_query)

      keep_box_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
      keep_feat_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
      keep_image_idx = torch.ones_like(objs.unsqueeze(1), dtype=torch.float)
      combine_gt_pred_box = torch.zeros_like(objs)

      if args.mode == 'reposition':

        triple_idx = new_id[0]
        target_predicate = new_id[1]
        source_predicate = triples[triple_idx, 1]

        triples_changed = triples.clone()
        triple_to_img_changed = triple_to_img.clone()
        triples_changed[triple_idx, 1] = torch.tensor(np.int64(int(target_predicate), dtype=torch.long))
        subject_node = triples_changed[triple_idx, 0]
        object_node = triples_changed[triple_idx, 2]

        indexes = []

        for t_index in range(triples_changed.size(0)):

          if triples_changed[t_index, 1] == source_predicate and (triples_changed[t_index, 0] == subject_node  \
                or triples_changed[t_index, 2] == object_node) and triple_idx != t_index:
            indexes.append(t_index)
        if len(indexes) > 0:
          triples_changed, triple_to_img_changed = remove_duplicates(triples_changed, triple_to_img_changed, indexes)

        img_gt_filename = '%04d_gt.png' % (img_idx)
        if target_predicate == source_predicate:
          img_pred_filename = '%04d_%d_auto.png' % (img_idx, target_predicate)
        else:
          img_pred_filename = '%04d_s%d_t%d_%d.png' % (img_idx, source_predicate, target_predicate, img_subid)

        triples_ = triples_changed
        triple_to_img_ = triple_to_img_changed

        if args.drop_obj:
          keep_box_idx[object_node] = 0
          keep_image_idx[object_node] = 0
        if args.drop_subj:
          keep_box_idx[subject_node] = 0
          keep_image_idx[subject_node] = 0

        txt_str = vocab['object_idx_to_name'][objs[subject_node]] + ' ' + \
                  vocab['pred_idx_to_name'][source_predicate] + ' ' + \
                  vocab['object_idx_to_name'][objs[object_node]]+ ' -> '  + \
                  vocab['pred_idx_to_name'][target_predicate]

      else:

        objs[subject_node] = torch.tensor(np.int64(int(new_id)), dtype=torch.long)

        img_gt_filename = '%04d_gt.png' % (img_idx)
        img_pred_filename = '%04d_%d.png' % (img_idx, img_subid)

        triples_ = triples
        triple_to_img_ = triple_to_img

        if args.mode == 'replace':
          keep_feat_idx[subject_node] = 0
          keep_image_idx[subject_node] = 0
          if args.combined_gt_pred_box and not is_background(objs_before[subject_node]):
            keep_box_idx[subject_node] = 0
            combine_gt_pred_box[subject_node] = 1
          txt_str = 'from ' + vocab['object_idx_to_name'][objs_before[subject_node]] + \
                    " to " + vocab['object_idx_to_name'][new_id]

        if args.mode == 'auto_withfeats':
          keep_image_idx[subject_node] = 0
          # uncomment to drop the box coords
          #keep_box_idx[subject_node] = 0
          txt_str = 'reconstructed ' + vocab['object_idx_to_name'][new_id]

        if args.mode == 'auto_nofeats':
          if not args.with_query_image:
            keep_feat_idx[subject_node] = 0
          keep_image_idx[subject_node] = 0
          # uncomment to drop the box coords
          #keep_box_idx[subject_node] = 0
          txt_str = 'reconstructed ' + vocab['object_idx_to_name'][new_id]

        if args.mode == 'remove':
          txt_str = 'removed ' + vocab['object_idx_to_name'][id_removed]

      model_out = model(objs, triples_, obj_to_img,
          boxes_gt=boxes, masks_gt=None, src_image=imgs_in, mode=args.mode, query_feats=query_feats,
          keep_box_idx=keep_box_idx, keep_feat_idx=keep_feat_idx, keep_image_idx=keep_image_idx,
          combine_gt_pred_box_idx=combine_gt_pred_box, query_idx=subject_node, random_feats=args.random_feats)

      imgs_pred, boxes_pred, masks_pred, noised_srcs, _ = model_out

      if args.label_saved_image:
        save_image_with_label(imgs_pred, imgs_gt, img_dir, img_pred_filename, txt_str)
      else:
        save_image_from_tensor(imgs_pred, img_dir, img_pred_filename)
        save_image_from_tensor(imgs_gt, img_dir, img_gt_filename)

      # save text that describes the change, for each generated image
      f.write(str(img_idx) + "_" + str(img_subid) + " " + txt_str + "\n")

      if args.save_graph_image:
        # graph before changes
        graph_img = draw_scene_graph(objs_before, triples_before, vocab)
        graph_path = os.path.join(graph_dir, img_gt_filename)
        imsave(graph_path, graph_img)
        # graph after changes
        graph_img = draw_scene_graph(objs, triples_, vocab)
        graph_path = os.path.join(graph_dir, img_pred_filename)
        imsave(graph_path, graph_img)

      if args.save_graph_json:
        # graph before changes
        save_graph_json(objs_before, triples_before, boxes_before, "before", graph_json_dir, img_idx)
        # graph after changes
        save_graph_json(objs, triples_, boxes, "after", graph_json_dir, img_idx)

      img_subid += 1

    img_idx += 1
    print('image %d ' % img_idx)

  f.close()


def main(args):

  output_dir = args.exp_dir + "/" + args.experiment + "/" + args.mode
  if args.with_query_image:
    output_dir = output_dir + "_query"

  got_checkpoint = args.checkpoint is not None
  if got_checkpoint:
    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, output_dir)
  else:
    print('--checkpoint not specified')


if __name__ == '__main__':
  main(args)
