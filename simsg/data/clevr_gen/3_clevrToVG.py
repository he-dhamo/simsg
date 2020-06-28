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
Stores the generated CLEVR data in the same format as Visual Genome, for easy loading
"""

import math, sys, random, argparse, json, os
import numpy as np
#import cv2
import itertools


def get_rel(argument):
    switcher = {
        3: "left of",
        2: "behind",
        1: "right of",
        0: "front of"
    }
    return switcher.get(argument, "Invalid label")

def get_xy_coords(obj):
    obj_x, obj_y, obj_w, obj_h = obj['bbox']

    return obj_x, obj_y, obj_w, obj_h

def getLabel(obj, subj):
    diff = np.subtract(obj[:-1], subj[:-1])
    label = int(np.argmax(np.abs(diff)))
    if diff[label] > 0:
        label += 2
    return label


np.random.seed(seed=0)
parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--data_path', default='/media/azadef/MyHDD/Data/MyClevr_postcvpr/source')

args = parser.parse_args()

data_path = args.data_path
im_names = sorted(os.listdir(os.path.join(data_path,'images')))
json_names = sorted(os.listdir(os.path.join(data_path,'scenes')))

labels_file_path = os.path.join(data_path, 'image_data.json')
objs_file_path = os.path.join(data_path, 'objects.json')
rels_file_path = os.path.join(data_path, 'relationships.json')
attrs_file_path = os.path.join(data_path, 'attributes.json')
split_file_path = os.path.join(data_path, 'vg_splits.json')

rel_alias_path = os.path.join(data_path, 'relationship_alias.txt')
obj_alias_path = os.path.join(data_path, 'object_alias.txt')

im_id = 0

num_objs = 0
num_rels = 0
obj_data = []
rel_data = []
scenedata = []
attr_data = []

obj_data_file = open(objs_file_path, 'w')
rel_data_file = open(rels_file_path, 'w')
attr_data_file = open(attrs_file_path, 'w')
split_data_file = open(split_file_path, 'w')

with open(rel_alias_path, 'w'):
  print("Created relationship alias file!")

with open(obj_alias_path, 'w'):
  print("Created object alias file!")

num_ims = len(json_names)

split_dic = {}
rand_perm = np.random.choice(num_ims, num_ims, replace=False)
train_split = int(num_ims * 0.8)
val_split = int(num_ims * 0.9)

train_list = [int(num) for num in rand_perm[0:train_split]]
val_list = [int(num) for num in rand_perm[train_split+1:val_split]]
test_list = [int(num) for num in rand_perm[val_split+1:num_ims]]

split_dic['train'] = train_list
split_dic['val'] = val_list
split_dic['test'] = test_list

json.dump(split_dic, split_data_file, indent=2)
split_data_file.close()

with open(labels_file_path, 'w') as labels_file:
    for json_file in json_names:
        file_name = os.path.join(data_path, 'scenes',json_file)
        new_obj = {}
        new_rel = {}
        new_scene = {}
        new_attr = {}
        
        with open(file_name, 'r') as f:
            properties = json.load(f)

            new_obj['image_id'] = im_id
            new_rel['image_id'] = im_id
            new_scene['image_id'] = im_id
            new_attr['image_id'] = im_id

            rels = []
            objs = properties["objects"]
            attr_objs = []

            for j, obj in enumerate(objs):
                obj['object_id'] = num_objs + j
                obj['x'], obj['y'], obj['w'], obj['h'] = get_xy_coords(obj)
                obj.pop('bbox')
                
                obj['names'] = [obj["color"] + " " + obj['shape']]
                obj.pop('shape')
                obj.pop('rotation')
                attr_obj = obj
                attr_obj["attributes"] = [obj["color"]]
                obj.pop('size')
                attr_objs.append(attr_obj)


            new_obj['objects'] = objs
            new_scene['objects'] = objs
            new_attr['attributes'] = attr_objs

            pairs = list(itertools.combinations(objs, 2))
            indices = list((i,j) for ((i,_),(j,_)) in itertools.combinations(enumerate(objs), 2))
            for ii, (obj, subj) in enumerate(pairs):
                label = getLabel(obj['3d_coords'], subj['3d_coords'])
                predicate = get_rel(label)
                if predicate == 'Invalid label':
                    print("Invalid label!")
                rel = {}
                rel['predicate'] = predicate
                rel['name'] = predicate
                rel['object'] = obj
                rel['subject'] = subj
                rel['relationship_id'] = num_rels + ii
                rel['object_id'] = obj['object_id']
                rel['subject_id'] = subj['object_id']
                rels.append(rel)

            new_rel['relationships'] = rels
            new_scene['relationships'] = rels
            im_path = os.path.join(data_path,'images',json_file.strip('.json') + '.png')


            new_obj['url'] = im_path

            new_scene['url'] = im_path
            new_scene['width'] = 320
            new_scene['height'] = 240
            im_id += 1
            num_objs += len(objs)
            num_rels += len(rels)
            
        scenedata.append(new_scene)
        obj_data.append(new_obj)
        rel_data.append(new_rel)
        attr_data.append(new_attr)

    json.dump(scenedata, labels_file, indent=2)

json.dump(obj_data, obj_data_file, indent=2)
json.dump(rel_data, rel_data_file, indent=2)
json.dump(attr_data, attr_data_file, indent=2)

obj_data_file.close()
rel_data_file.close()
attr_data_file.close()
print("Done processing!")

