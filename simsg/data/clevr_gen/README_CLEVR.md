## Setup
You will need to download and install [Blender](https://www.blender.org/); code has been developed and tested using Blender version 2.78c but other versions may work as well.

## Data Generation
Run 1_gen_data.sh to generate source/target pairs for each of the available modes. The first argument is the change mode, the second argument is the path to blender directory and the third argument is the option to merge all of the modes into one folder. The manipulation modes are: removal, replacement, addition, rel_change
Sample command:
```bash
sh 1_gen_data.sh removal /home/user/blender2.79-cuda10-x86_64/ 1
```
Run this file for all of the modes to generate all variations of change.

## Merging all the folders
If you set the merge argument to 1, you can skip this set.

Otherwise, move all of the generated images for each mode to a single folder so that you have all images in the "images" folder and all the scenes in "scenes" folder.

## Arranging the generated image into separate folder
To arrange the the data, run 2_arrange_data.sh. The script will move the previously generated data to MyClevr directory in the required format.

```bash
sh 2_arrange_data.sh
```

## Converting clevr format to VG format
This step converts the generated data to the format required by SIMSG similar to the Visual Genome dataset. The final scene graphs and the required files are generated in this step.

```bash
python 3_clevrToVG.py
```

## Preprocessing

Run scripts/preprocess_vg.py on the generated CLEVR data for both source and target folders to preprocess the data for SIMSG. Set VG_DIR inside the file to point to CLEVR source or target directory.

To make sure no image is removed by the preprocess filtering step so that we have corresponding source/target pairs, set the following arguments:

```bash
python scripts/preprocess_vg.py --min_object_instances 0 --min_attribute_instances 0 --min_object_size 0 --min_objects_per_image 1 --min_relationship_instances 1 --max_relationships_per_image 50
```
## Acknowledgment

Our CLEVR generation code is based on <a href="https://github.com/facebookresearch/clevr-dataset-gen"> this repo</a>.
