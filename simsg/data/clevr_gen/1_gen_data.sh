#!/bin/bash
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

mode="$1" #removal, replacement, addition, rel_change
blender_path="$2" #"/home/azadef/Downloads/blender-2.79-cuda10-x86_64"
merge="$3"
if [ $merge ]
then 
out_folder="./output";
else
out_folder="./output_$mode";
fi
for i in {1..6}
do 
if [ $merge ]
then 
start=`expr $(ls -1 "$out_folder/images/" | wc -l)`;
else
start=`expr $(ls -1 "$out_folder/../output_rel_change/images/" | wc -l) + $(ls -1 "$out_folder/../output_remove/images/" | wc -l) + $(ls -1 "$out_folder/../output_replacement/images/" | wc -l) + $(ls -1 "$out_folder/../output_addition/images/" | wc -l)`;
start=$((start/2));
fi
echo $start
"$blender_path"/blender --background --python render_clevr.py -- --num_images 800 --output_image_dir "$out_folder/images/" --output_scene_dir "$out_folder/scenes/" --output_scene_file "$out_folder/CLEVR_scenes.json" --start_idx $start --use_gpu 1 --mode "$mode"
done
