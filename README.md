# SIMSG

This is the code accompanying the paper

**Semantic Image Manipulation Using Scene Graphs | <a href="https://arxiv.org/pdf/2004.03677.pdf">arxiv</a>** <br/>
Helisa Dhamo*, Azade Farshad*, Iro Laina, Nassir Navab, Gregory D. Hager, Federico Tombari, Christian Rupprecht <br/>
**CVPR 2020**

The work for this paper was done at the Technical University of Munich.

In our work, we address the novel problem of image manipulation from scene graphs, in which a user can edit images by 
merely applying changes in the nodes or edges of a semantic graph that is generated from the image.

We introduce a spatio-semantic scene graph network that does not require direct supervision for constellation changes or 
image edits. This makes it possible to train the system from existing real-world datasets with no additional annotation 
effort.

If you find this code useful in your research, please cite
```
@inproceedings{dhamo2020_SIMSG,
  title={Semantic Image Manipulation Using Scene Graphs},
  author={Dhamo, Helisa and Farshad, Azade, and Laina, Iro and Navab, Nassir and
          Hager, Gregory D., and Tombari, Federico and Rupprecht, Christian},
  booktitle={CVPR},
  year={2020}
```

**Note:** The project page has been updated with new links for model checkpoints and data.


## Setup

We have tested it on Ubuntu 16.04 with Python 3.7 and PyTorch 1.2.

### Setup code
You can setup a conda environment to run the code like this:

```bash
# clone this repository and move there
git clone --recurse-submodules https://github.com/he-dhamo/simsg.git
cd simsg
# create a conda environment and install the requirments
conda create --name simsg_env python=3.7 --file requirements.txt 
conda activate simsg_env          # activate virtual environment
# install pytorch and cuda version as tested in our work
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
# more installations
pip install opencv-python tensorboardx grave addict
# to add current directory to python path run
echo $PWD > <path_to_env>/lib/python3.7/site-packages/simsg.pth 
```
`path_to_env` can be found using `which python` (the resulting path minus `/bin/python`) while the conda env is active.

### Setup Visual Genome data
(instructions from the <a href="https://github.com/google/sg2im"> sg2im </a> repository)

Run the following script to download and unpack the relevant parts of the Visual Genome dataset:

```bash
bash scripts/download_vg.sh
```

This will create the directory `datasets/vg` and will download about 15 GB of data to this directory; after unpacking it 
will take about 30 GB of disk space.

After downloading the Visual Genome dataset, we need to preprocess it. This will split the data into train / val / test 
splits, consolidate all scene graphs into HDF5 files, and apply several heuristics to clean the data. In particular we 
ignore images that are too small, and only consider object and attribute categories that appear some number of times in 
the training set; we also ignore objects that are too small, and set minimum and maximum values on the number of objects 
and relationships that appear per image.

```bash
python scripts/preprocess_vg.py
```

This will create files `train.h5`, `val.h5`, `test.h5`, and `vocab.json` in the directory `datasets/vg`.


## Training

To train the model, you can either set the right options in a `args.yaml` file and then run the `run_train.py` script 
(**highly recommended**):
```
python scripts/run_train.py args.yaml
``` 
Or, alternatively run the `train.py` script with the respective arguments. For example:
```
python scripts/train.py --checkpoint_name myckpt --dataset vg --spade_blocks True
```

We provide the configuration files for the experiments presented in the paper, represented in the format 
`args_{resolution}_{decoder}_{dataset}.yaml`.

Please set the dataset path `DATA_DIR` in `train.py` before running.

Most relevant arguments:

`--checkpoint_name`: Base filename for saved checkpoints; default is 'checkpoint', so the filename for the checkpoint 
with model parameters will be 'checkpoint_model.pt' <br/>
`--selective_discr_obj`: If True, only apply the object discriminator in the reconstructed RoIs (masked image areas). 
<br/>
`--feats_in_gcn`: If True, feed the RoI visual features in the scene graph network (SGN). <br/> 
`--feats_out_gcn`: If True, concatenate RoI visual features with the node feature (result of SGN). <br/>
`--is_baseline`: If True, run cond-sg2im baseline. <br/>
`--is_supervised`: If True, run fully-supervised baseline for CLEVR. <br/>
`--spade_gen_blocks`: If True, use SPADE blocks in the decoder architecture. Otherwise, use CRN blocks.

## Evaluate reconstruction

To evaluate the model in reconstruction mode, you need to run the script ```evaluate_reconstruction.py```. The MAE, SSIM 
and LPIPS will be printed on the terminal with frequency ```--print_every``` and saved on pickle files with frequency 
```--save_every```. When ```--save_images``` is set to ```True```, the code also saves one reconstructed sample per 
image, which can be used for the computation of <a href="https://github.com/bioinf-jku/TTUR" >FID</a> and 
<a href="https://github.com/openai/improved-gan/tree/master/inception_score">Inception score</a>.

Other relevant arguments are:

`--exp_dir`: path to folder where all experiments are saved <br/>
`--experiment`: name of experiment, e.g. spade_vg_64 <br/>
`--checkpoint`: checkpoint path <br/>
`--with_feats`: (bool) using RoI visual features <br/>
`--generative`: (bool) fully-generative mode, i.e. the whole input image is masked out

If `--checkpoint` is not specified, the checkpoint path is automatically set to ```<exp_dir>/<experiment>_model.pt```.

To reproduce the results from Table 2, please run:
1. for the fully generative task:
 ```
 python scripts/evaluate_reconstruction.py --exp_dir /path/to/experiment/dir --experiment spade --with_feats True 
--generative True
```
 2. for the RoI reconstruction with visual features:
 ```
 python scripts/evaluate_reconstruction.py --exp_dir /path/to/experiment/dir --experiment spade --with_feats True 
--generative False
```
 3. for the RoI reconstruction without visual features:
  ```
  python scripts/evaluate_reconstruction.py --exp_dir /path/to/experiment/dir --experiment spade --with_feats False 
--generative False
  ```

To evaluate the model with ground truth (GT) and predicted (PRED) scene graphs, please set `--predgraphs`  namely to 
```False``` or ```True```. 
Before running with predicted graphs (PRED), make sure you have downloaded the respective predicted graphs 
`test_predgraphs.h5` and placed it at the same directory as ```test.h5```.

Please set the dataset path `DATA_DIR` before running.

For the LPIPS computation, we use PerceptualSimilarity cloned from the 
<a href="https://github.com/richzhang/PerceptualSimilarity">official repo</a> as a git submodule. If you did not clone our
repository recursively, run:
```
git submodule update --init --recursive
```

## Evaluate changes on Visual Genome

To evaluate the model in semantic editing mode on Visual Genome, you need to run the script 
```evaluate_changes_vg.py``` . The script automatically generates edited images, without a user interface.

The most relevant arguments are:

`--exp_dir`: path to folder where all experiments are saved <br/>
`--experiment`: name of experiment, e.g. spade_vg_64 <br/>
`--checkpoint`: checkpoint path <br/>
`--mode`: choose from options ['replace',  'reposition',  'remove',  'auto_withfeats', 'auto_nofeats'], for namely 
object replacement, relationship change, object removal, RoI reconstruction with visual features, RoI reconstruction 
without features. <br/>
`--with_query_image`: (bool) in case you want to use visual features from another image (query image). used in 
combination with `mode='auto_nofeats'`. <br/>

If `--checkpoint` is not specified, the checkpoint path is automatically set to ```<exp_dir>/<experiment>_model.pt```.

Example run of object replacement changes using the spade model:
```
python scripts/evaluate_changes_vg.py --exp_dir /path/to/experiment/dir --experiment spade --mode replace
```

Please set the dataset path `VG_DIR` before running.

## Evaluate changes on CLEVR

To evaluate the model in semantic editing mode on CLEVR, you need to run the script ```evaluate_changes_clevr.py```. 
The script automatically generates edited images, without a user interface.

The most relevant arguments are:

`--exp_dir`: path to folder where all experiments are saved <br/>
`--experiment`: name of experiment, e.g. spade_clevr_64 <br/>
`--checkpoint`: checkpoint path <br/>
`--image_size`: size of the input image, can be (64,64) or (128,128) based on the size used in training <br/>

If `--checkpoint` is not specified, the checkpoint path is automatically set to ```<exp_dir>/<experiment>_model.pt```.

Example run of object replacement changes using the spade model:
```
python scripts/evaluate_changes_clevr.py --exp_dir /path/to/experiment/dir --experiment spade
```

Please set the dataset path `CLEVR_DIR` before running.

## User Interface

To run a simple user interface that supports different manipulation types, such as object addition, removal, replacement 
and relationship change run:
```
python scripts/SIMSG_gui.py 
```

Relevant options:

`--checkpoint`: path to checkpoint file <br/>
`--dataset`: visual genome or clevr <br/>
`--predgraphs`:(bool) specifies loading either ground truth or predicted graphs. So far, predicted graphs are only 
available for visual genome. <br/>
`--data_h5`: path of h5 file used to load a certain data split. If not excplicitly set, it uses `test.h5` for GT graphs 
and `test_predgraphs.h5` for predicted graphs. <br/>
`--update_input`: (bool) used to control a sequence of changes in the same image. If `True`, it sets the input as the 
output of the previous generation. Otherwise, all consecutive changes are applied on the original image.
Please set the value of DATA_DIR in SIMSG_gui.py to point to your dataset.

Once you click on "Load image" and "Get graph" a new image and corresponding scene graph will appear. The current 
implementation loads GT or predicted graph. Then you can apply one of the following manipulations:

- For **object replacement** or **relationship change**:
  1. Select the node you want to change 
  2. Choose a new category from namely the "Replace object" and "Change Relationship" menu.
- For **object addition**: 
  1. Choose a node where you want to connect the new object to (from "Connect to object") 
  2. Select the category of the new object and relationship ("Add new node", "Add relationship"). 
  3. Specify the direction of the connection. Click on "Add as subject" for a `new_node -> predicate -> existing_node` 
  direction, and click "Add as object" for `existing_node -> predicate -> new_node`.
- For **object removal**: 
  1. Select the node you want to remove
  2. Click on "Remove node" <br/>
  Note that the current implementation only supports object removal (cannot remove relationships); though the model 
  supports this and the GUI implementation can be extended accordingly.

After you have completed the change, click on "Generate image". Alternatively you can save the image. 

## Download

Visit the <a href="https://he-dhamo.github.io/SIMSG/#download">project page</a> to download model checkpoints, 
predicted scene graphs and the CLEVR data with edit pairs.

We also provide the code used to generate the CLEVR data with change pairs. Please follow the instructions from 
[here](simsg/data/clevr_gen/README_CLEVR.md).

## Acknoledgement

This code is based on the <a href="https://github.com/google/sg2im"> sg2im repository </a>. 

The following directory is taken from the <a href="https://github.com/NVlabs/SPADE"> SPADE </a> repository:
- simsg/SPADE/
