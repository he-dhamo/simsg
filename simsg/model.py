#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo, Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except impliance with the License.
# You may obtain a copy of the License
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from simsg.graph import GraphTripleConv, GraphTripleConvNet
from simsg.decoder import DecoderNetwork
from simsg.layout import boxes_to_layout, masks_to_layout
from simsg.layers import build_mlp

import random
import torchvision as T

import cv2
from simsg.data import imagenet_deprocess_batch

from sklearn.decomposition import PCA

import torch.distributions as tdist
from simsg.feats_statistics import get_mean, get_std


class SIMSGModel(nn.Module):
    """
    SIMSG network. Given a source image and a scene graph, the model generates
    a manipulated image that satisfies the scene graph constellations
    """
    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 decoder_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mask_size=None, mlp_normalization='none', layout_noise_dim=0,
                 img_feats_branch=True, feat_dims=128, is_baseline=False, is_supervised=False,
                 feats_in_gcn=False, feats_out_gcn=True, layout_pooling="sum",
                 spade_blocks=False, **kwargs):

        super(SIMSGModel, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab
        self.image_size = image_size
        self.layout_noise_dim = layout_noise_dim

        self.feats_in_gcn = feats_in_gcn
        self.feats_out_gcn = feats_out_gcn
        self.spade_blocks = spade_blocks
        self.is_baseline = is_baseline
        self.is_supervised = is_supervised

        if self.is_supervised:
            self.im_to_noise_conv = nn.Conv2d(3, 32, 1, stride=1)

        self.image_feats_branch = img_feats_branch

        self.layout_pooling = layout_pooling

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        if self.is_baseline or self.is_supervised:
            gconv_input_dims = embedding_dim
        else:
            if self.feats_in_gcn:
                gconv_input_dims = embedding_dim + 4 + feat_dims
            else:
                gconv_input_dims = embedding_dim + 4

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(gconv_input_dims, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim_obj': gconv_input_dims,
                'input_dim_pred': embedding_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim_obj': gconv_dim,
                'input_dim_pred': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers - 1,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        self.mask_net = None
        if mask_size is not None and mask_size > 0:
            self.mask_net = self._build_mask_net(gconv_dim, mask_size)

        if self.is_baseline or not self.feats_out_gcn or self.is_supervised:
            ref_input_dim = gconv_dim + layout_noise_dim
        else:
            ref_input_dim = gconv_dim + layout_noise_dim + feat_dims

        decoder_kwargs = {
            'dims': (ref_input_dim,) + decoder_dims,
            'normalization': normalization,
            'activation': activation,
            'spade_blocks': spade_blocks,
            'source_image_dims': layout_noise_dim
        }

        self.decoder_net = DecoderNetwork(**decoder_kwargs)#.to("cuda:1")

        if self.image_feats_branch:
            self.conv_img = nn.Sequential(
                nn.Conv2d(4, layout_noise_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(layout_noise_dim),
                nn.ReLU()
            )

        if not (self.is_baseline or self.is_supervised):
            self.high_level_feats = self.build_obj_feats_net()
            # freeze vgg
            for param in self.high_level_feats.parameters():
                param.requires_grad = False

            self.high_level_feats_fc = self.build_obj_feats_fc(feat_dims)

            if self.feats_in_gcn:
                self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4 + feat_dims)
                if self.feats_out_gcn:
                    self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims)
            else:
                self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim + 4)
                self.layer_norm2 = nn.LayerNorm(normalized_shape=gconv_dim + feat_dims)

        self.p = 0.25
        self.p_box = 0.35

    def build_obj_feats_net(self):
        # get VGG16 features for each object RoI
        vgg_net = T.models.vgg16(pretrained=True)
        layers = list(vgg_net.features._modules.values())[:-1]

        img_feats = nn.Sequential(*layers)

        return img_feats

    def build_obj_feats_fc(self, feat_dims):
        # fc layer following the VGG16 backbone
        return nn.Linear(512 * int(self.image_size[0] / 64) * int(self.image_size[1] / 64), feat_dims)

    def _build_mask_net(self, dim, mask_size):
        # mask prediction network
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, objs, triples, obj_to_img=None, boxes_gt=None, masks_gt=None, src_image=None, imgs_src=None,
                keep_box_idx=None, keep_feat_idx=None, keep_image_idx=None, combine_gt_pred_box_idx=None,
                query_feats=None, mode='train', t=0, query_idx=0, random_feats=False, get_layout_boxes=False):
        """
        Required Inputs:
        - objs: LongTensor of shape (num_objs,) giving categories for all objects
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (num_objs,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (num_objs, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        - src_image: (num_images, 3, H, W) input image to be modified
        - query_feats: feature vector from another image, to be used optionally in object replacement
        - keep_box_idx, keep_feat_idx, keep_image_idx: Tensors of ones or zeros, indicating
        what needs to be kept/masked on evaluation time.
        - combine_gt_pred_box_idx: Tensor of ones and zeros, indicating if size of pred box and position of gt boxes
          should be combined. Used in the "replace" mode.
        - mode: string, can take the option 'train' or one of the evaluation modes
        - t: iteration index, intended for debugging
        - query_idx: scalar id of object where query_feats should be used
        - random_feats: boolean. Used during evaluation to use noise instead of zeros for masked features phi
        - get_layout_boxes: boolean. If true, the boxes used for final layout construction are returned
        """

        assert mode in ["train", "eval", "auto_withfeats", "auto_nofeats", "reposition", "remove", "replace", "addition"]

        evaluating = mode != 'train'

        in_image = src_image.clone()
        num_objs = objs.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (num_triples, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (num_triples,)
        edges = torch.stack([s, o], dim=1)  # Shape is (num_triples, 2)

        obj_vecs = self.obj_embeddings(objs)

        if obj_to_img is None:
            obj_to_img = torch.zeros(num_objs, dtype=objs.dtype, device=objs.device)

        if combine_gt_pred_box_idx is None:
            combine_gt_pred_box_idx = torch.zeros_like(objs)

        if not (self.is_baseline or self.is_supervised):

            box_ones = torch.ones([num_objs, 1], dtype=boxes_gt.dtype, device=boxes_gt.device)
            box_keep, feats_keep = self.prepare_keep_idx(evaluating, box_ones, in_image.size(0), obj_to_img,
                                                         keep_box_idx, keep_feat_idx)

            boxes_prior = boxes_gt * box_keep

            obj_crop = get_cropped_objs(in_image, boxes_gt, obj_to_img, feats_keep, box_keep, evaluating, mode)

            high_feats = self.high_level_feats(obj_crop)

            high_feats = high_feats.view(high_feats.size(0), -1)
            high_feats = self.high_level_feats_fc(high_feats)

            feats_prior = high_feats * feats_keep

            if evaluating and random_feats:
                # fill with noise the high level visual features, if the feature is masked/dropped
                normal_dist = tdist.Normal(loc=get_mean(self.spade_blocks), scale=get_std(self.spade_blocks))
                highlevel_noise = normal_dist.sample([high_feats.shape[0]])
                feats_prior += highlevel_noise.cuda() * (1 - feats_keep)

            # when a query image is used to generate an object of the same category
            if query_feats is not None:
                feats_prior[query_idx] = query_feats

            if self.feats_in_gcn:
                obj_vecs = torch.cat([obj_vecs, boxes_prior, feats_prior], dim=1)

            else:
                obj_vecs = torch.cat([obj_vecs, boxes_prior], dim=1)
            obj_vecs = self.layer_norm(obj_vecs)

        pred_vecs = self.pred_embeddings(p)

        # GCN pass
        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        # Box prediction network
        boxes_pred = self.box_net(obj_vecs)

        # Mask prediction network
        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(num_objs, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()

        if not (self.is_baseline or self.is_supervised) and self.feats_out_gcn:
            obj_vecs = torch.cat([obj_vecs, feats_prior], 1)
            obj_vecs = self.layer_norm2(obj_vecs)

        use_predboxes = False

        H, W = self.image_size

        if self.is_baseline or self.is_supervised:

            layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
            box_ones = torch.ones([num_objs, 1], dtype=boxes_gt.dtype, device=boxes_gt.device)

            box_keep = self.prepare_keep_idx(evaluating, box_ones, in_image.size(0), obj_to_img, keep_box_idx,
                                                keep_feat_idx, with_feats=False)

            # mask out objects
            if not evaluating:
                keep_image_idx = box_keep

            for box_id in range(keep_image_idx.size(0)):
                if keep_image_idx[box_id] == 0:
                    in_image = mask_image_in_bbox(in_image, boxes_gt, box_id, obj_to_img)
            generated = None

        else:
            if use_predboxes:
                layout_boxes = boxes_pred
            else:
                layout_boxes = boxes_gt.clone()

            if evaluating:
                # should happen on evaluation only
                # drop region in image corresponding to predicted box
                # so that a new content/object is generated there
                for idx in range(len(keep_box_idx)):
                    if keep_box_idx[idx] == 0 and combine_gt_pred_box_idx[idx] == 0:
                        in_image = mask_image_in_bbox(in_image, boxes_pred, idx, obj_to_img)
                        layout_boxes[idx] = boxes_pred[idx]

                    if keep_box_idx[idx] == 0 and combine_gt_pred_box_idx[idx] == 1:
                        layout_boxes[idx] = combine_boxes(boxes_gt[idx], boxes_pred[idx])
                        in_image = mask_image_in_bbox(in_image, layout_boxes, idx, obj_to_img)

            generated = torch.zeros([obj_to_img.size(0)], device=obj_to_img.device,
                                                        dtype=obj_to_img.dtype)
            if not evaluating:
                keep_image_idx = box_keep * feats_keep

            for idx in range(len(keep_image_idx)):
                if keep_image_idx[idx] == 0:
                    in_image = mask_image_in_bbox(in_image, boxes_gt, idx, obj_to_img)
                    generated[idx] = 1

            generated = generated > 0

        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W,
                                     pooling=self.layout_pooling)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                     obj_to_img, H, W, pooling=self.layout_pooling)#, front_idx=1-keep_box_idx)

        noise_occluding = True

        if self.image_feats_branch:

            N, C, H, W = layout.size()
            noise_shape = (N, 3, H, W)
            if noise_occluding:
                layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                           device=layout.device)
            else:
                layout_noise = torch.zeros(noise_shape, dtype=layout.dtype,
                                           device=layout.device)

            in_image[:, :3, :, :] = layout_noise * in_image[:, 3:4, :, :] + in_image[:, :3, :, :] * (
                        1 - in_image[:, 3:4, :, :])
            img_feats = self.conv_img(in_image)

            layout = torch.cat([layout, img_feats], dim=1)

        elif self.layout_noise_dim > 0:
            N, C, H, W = layout.size()
            noise_shape = (N, self.layout_noise_dim, H, W)
            if self.is_supervised:
                layout_noise = self.im_to_noise_conv(imgs_src)
            else:
                layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
                                       device=layout.device)

            layout = torch.cat([layout, layout_noise], dim=1)

        img = self.decoder_net(layout)

        # visualize layout for debugging purposes
        #if t % 50 == 0:
        #    visualize_layout(img, in_image, visualize_layout, obj_vecs, layout_boxes, layout_masks,
        #                     obj_to_img, H, W)
        if get_layout_boxes:
            return img, boxes_pred, masks_pred, in_image, generated, layout_boxes
        else:
            return img, boxes_pred, masks_pred, in_image, generated

    def forward_visual_feats(self, img, boxes):
        """
        gets VGG visual features from an image and box
        used for image query on evaluation time (evaluate_changes_vg.py)
        - img: Tensor of size [1, 3, H, W]
        - boxes: Tensor of size [4]
        return: feature vector in the RoI
        """

        left, right, top, bottom = get_left_right_top_bottom(boxes, img.size(2), img.size(3))

        obj_crop = img[0:1, :3, top:bottom, left:right]
        obj_crop = F.upsample(obj_crop, size=(img.size(2) // 4, img.size(3) // 4), mode='bilinear', align_corners=True)

        feats = self.high_level_feats(obj_crop)

        feats = feats.view(feats.size(0), -1)
        feats = self.high_level_feats_fc(feats)

        return feats

    def prepare_keep_idx(self, evaluating, box_ones, num_images, obj_to_img, keep_box_idx,
                         keep_feat_idx, with_feats=True):
        # random drop of boxes and visual feats on training time
        # use objs idx passed as argument on eval time
        imgbox_idx = torch.zeros(num_images, dtype=torch.int64)
        for i in range(num_images):
            imgbox_idx[i] = (obj_to_img == i).nonzero()[-1]

        if evaluating:
            if keep_box_idx is not None:
                box_keep = keep_box_idx
            else:
                box_keep = box_ones

            if with_feats:
                if keep_feat_idx is not None:
                    feats_keep = keep_feat_idx
                else:
                    feats_keep = box_ones
        else:
            # drop random box(es) and feature(s)
            box_keep = F.dropout(box_ones, self.p_box, True, False) * (1 - self.p_box)
            if with_feats:
                feats_keep = F.dropout(box_ones, self.p, True, False) * (1 - self.p)

        # image obj cannot be dropped
        box_keep[imgbox_idx, :] = 1

        if with_feats:
            # image obj feats should not be dropped
            feats_keep[imgbox_idx, :] = 1
            return box_keep, feats_keep

        else:
            return box_keep


def get_left_right_top_bottom(box, height, width):
    """
    - box: Tensor of size [4]
    - height: scalar, image hight
    - width: scalar, image width
    return: left, right, top, bottom in image coordinates
    """
    left = (box[0] * width).type(torch.int32)
    right = (box[2] * width).type(torch.int32)
    top = (box[1] * height).type(torch.int32)
    bottom = (box[3] * height).type(torch.int32)

    return left, right, top, bottom


def mask_image_in_bbox(image, boxes, idx, obj_to_img, mode="normal"):
    """
    - image: Tensor of size [num_images, 4, H, W]
    - boxes: Tensor of size [num_objs, 4]
    - idx: scalar, object id
    - obj_to_img: Tensor of size [num_objs]
    - mode: string, "removal" if evaluating on removal mode, "normal" otherwise
    return: image, mask channel is set to ones in the bbox area of the object with id=idx
    """
    left, right, top, bottom = \
        get_left_right_top_bottom(boxes[idx], image.size(2), image.size(3))

    image[obj_to_img[idx], 3, top:bottom, left:right] = 1

    # on removal mode, make image area gray, so that features of removed object are not extracted
    if mode == "removal":
        image[obj_to_img[idx], :3, :, :] = put_gray_mask(image[obj_to_img[idx], :3, :, :],
                                                         image[obj_to_img[idx], 3:, :, :])

    return image


def put_gray_mask(imgs, mask):
    """
    fill image with gray pixels wherever mask is one
    - imgs: Tensor of size [num_images, 3, H, W] or [3, H, W]
    - mask: Tensor of size [num_images, 1, H, W] or [3, H, W]
    return: masked image of the same size as input image
    """

    reset = False
    if imgs.dim() == 3:
        imgs = torch.unsqueeze(imgs, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        reset = True

    imgs_masked = imgs * (1 - mask) + 0.5 * mask

    if reset:
        return imgs_masked[0]
    else:
        return imgs_masked


def get_cropped_objs(imgs, boxes, obj_to_img, feats_keeps, boxes_keeps, evaluating, mode, masked_feats=True):
    """
    prepare object RoIs for feature extraction
    - imgs: Tensor of size [num_images, 4, H, W]
    - boxes: Tensor of size [num_objs, 4]
    - obj_to_img: Tensor of size [num_objs]
    - feats_keeps: Tensor of size [num_objs]
    - boxes_keeps: Tensor of size [num_objs]
    - evaluating: boolean
    - mode: string, evaluation mode
    - masked_feats: boolean, if true image areas corresponding to dropped features are filled with gray pixels
    return: object RoI images, ready for feature extraction [num_objects, 3, H/4, W/4]
    """
    cropped_objs = []

    if masked_feats:
        features_mask = imgs[:, 3:4, :, :].clone()  # zeros

        if not evaluating:
            assert (imgs[:, 3, :, :].sum() == 0)

        for i in range(obj_to_img.size(0)):

            # during training: half the times, if the features are dropped then remove a patch from the image
            # during eval modes: drop objects that are going to change position
            if (feats_keeps[i, 0] == 0 and (i % 2 == 0 or evaluating)) or \
                    (boxes_keeps[i, 0] == 0 and evaluating):

                left, right, top, bottom = \
                    get_left_right_top_bottom(boxes[i], imgs.size(2), imgs.size(3))
                features_mask[obj_to_img[i], :, top:bottom, left:right] = 1

        # put gray as a mask over the image when things are dropped
        imgs_masked = put_gray_mask(imgs[:, :3, :, :], features_mask)

    for i in range(obj_to_img.size(0)):
        left, right, top, bottom = \
            get_left_right_top_bottom(boxes[i], imgs.size(2), imgs.size(3))

        # crop all objects
        try:
            if masked_feats and not (feats_keeps[i, 0] == 1 and boxes_keeps[i, 0] == 0 and evaluating):
                obj = imgs_masked[obj_to_img[i]:obj_to_img[i] + 1, :3, top:bottom, left:right]
            else:
                obj = imgs[obj_to_img[i]:obj_to_img[i] + 1, :3, top:bottom, left:right]
            obj = F.upsample(obj, size=(imgs.size(2) // 4, imgs.size(3) // 4), mode='bilinear', align_corners=True)
        except:
            # cropped object has H or W zero
            obj = torch.zeros([1, imgs.size(1) - 1, imgs.size(2) // 4, imgs.size(3) // 4],
                              dtype=imgs.dtype, device=imgs.device)

        cropped_objs.append(obj)

    cropped_objs = torch.cat(cropped_objs, 0)

    return cropped_objs


def visualize_layout(img, in_image, layout, feats, layout_boxes, layout_masks,
                     obj_to_img, H, W, with_dimreduction=True):
    # visualize layout for debug purposes during training
    # if with_dimreduction=True applies PCA on the layout
    # else, gets the first 3 channels of the original layout

    if with_dimreduction:
        pca = PCA(n_components=3)
        feats_reduced  = pca.fit_transform(feats.detach().cpu().numpy())
        feats_reduced = torch.Tensor(feats_reduced).cuda()

        layout = masks_to_layout(feats_reduced, layout_boxes, layout_masks,
                                     obj_to_img, H, W)

    vis_image = torch.cat([2*layout[:,:3,:,:], in_image[:,:-1,:,:], img], 3)
    vis_image = imagenet_deprocess_batch(vis_image)
    vis_image = torch.transpose(vis_image, 1, 2)
    vis_image = torch.transpose(vis_image, 2, 3)
    vis_image = torch.cat([vis_image[0], vis_image[1]], 1)
    vis_image = cv2.resize(vis_image.cpu().numpy(), (256*6, 256), interpolation = cv2.INTER_AREA)
    cv2.imshow("vis", vis_image)
    cv2.waitKey(25000)


def combine_boxes(gt, pred):
    """
    take position of gt bbox given as [left, top, right, bottom] in normalized coords
    and size of predicted bbox given as [left, top, right, bottom] in normalized coords
    used in object replacement, to adapt to the new class while keeping the original position
    """

    # center of gt bbox
    c_x = (gt[2] + gt[0]) / 2
    c_y = (gt[3] + gt[1]) / 2

    # pred H and W
    h = pred[3] - pred[1]
    w = pred[2] - pred[0]

    new_box = torch.zeros_like(gt)

    # update the comnined box, with a bit of care for corner cases
    new_box[0] = max(0.0, c_x - w / 2) # left
    new_box[2] = min(1.0, c_x + w / 2) # right
    new_box[1] = max(0.0, c_y - h / 2) # top
    new_box[3] = min(1.0, c_y + h / 2) # bottom

    return new_box
