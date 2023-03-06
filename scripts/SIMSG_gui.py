#!/usr/bin/python
#
# Copyright 2020 Azade Farshad, Helisa Dhamo
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
===============
Embedding in Qt
===============

Simple Qt application embedding Matplotlib canvases.  This program will work
equally well using Qt4 and Qt5.  Either version of Qt can be selected (for
example) by setting the ``MPLBACKEND`` environment variable to "Qt4Agg" or
"Qt5Agg", or by first importing the desired version of PyQt.
"""

import sys
from builtins import enumerate
import networkx as nx
from grave import plot_network
from grave.style import use_attributes
import matplotlib.pyplot as plt
import os, json, argparse
import numpy as np
from simsg.model import SIMSGModel
import torch

from simsg.data import imagenet_deprocess_batch
from simsg.loader_utils import build_eval_loader
from simsg.utils import int_tuple, bool_flag

import scripts.eval_utils as eval_utils

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5, QtGui

if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas)

from matplotlib.figure import Figure
import matplotlib as mpl

mpl.rcParams['savefig.pad_inches'] = 0
plt.margins(0.0)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='E:/Fcis/4th Year Fcis/simsg/simsg/checkpoints/spade_64_vg_model.pt')
parser.add_argument('--dataset', default='vg', choices=['clevr', 'vg'])
parser.add_argument('--data_h5', default=None)
parser.add_argument('--predgraphs', default=True, type=bool_flag)
parser.add_argument('--image_size', default=(64, 64), type=int_tuple)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--update_input', default=True, type=bool_flag)
parser.add_argument('--shuffle', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=1, type=int)
# deterministic vs diverse results
# instead of having zeros as visual feature, choose a random one from our feature distribution
parser.add_argument('--random_feats', default=False, type=bool_flag)

args = parser.parse_args()
args.mode = "eval"
if args.dataset == "clevr":
    assert args.random_feats == False
    DATA_DIR = "./datasets/clevr/target/"
    args.data_image_dir = DATA_DIR
else:
    DATA_DIR = "E:/Fcis/4th Year Fcis/simsg/simsg/datasets/vg/"
    args.data_image_dir = os.path.join(DATA_DIR, 'images')

if args.data_h5 is None:
    if args.predgraphs:
        args.data_h5 = os.path.join(DATA_DIR, 'test_predgraphs.h5')
    else:
        args.data_h5 = os.path.join(DATA_DIR, 'test.h5')


vocab_json = os.path.join(DATA_DIR, "vocab.json")
with open(vocab_json, 'r') as f:
    vocab = json.load(f)

preds = sorted(vocab['pred_idx_to_name'])
objs = sorted(vocab['object_idx_to_name'])

checkpoint = None

def build_model():
    global checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    model = SIMSGModel(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.image_size = args.image_size
#   model.cuda()
    return model


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        """
        Define all UI objects (buttons, comboboxes) and events
        """
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.resize(1600,700)
        self.pixmap = None
        self.imCounter = 0
        self.imLoadCounter = 0
        self.graphCounter = 0
        self.model = build_model()
        self.data_loader = iter(build_eval_loader(args, checkpoint, no_gt=True))
        self.mode = "auto_withfeats"
        self.new_objs = None
        self.new_triples = None

        self.in_edge_width = 2
        self.out_edge_width = 1

        self.graph = None

        self.selected_node = None
        layout = QtWidgets.QGridLayout(self._main)

        self.btnLoad = QtWidgets.QPushButton("Load image")
        self.btnLoad.resize(self.btnLoad.minimumSizeHint())
        self.btnLoad.clicked.connect(self.getfile)

        layout.addWidget(self.btnLoad, 6, 1, 1, 1)

        self.btnSave = QtWidgets.QPushButton("Save image")
        self.btnSave.resize(self.btnSave.minimumSizeHint())
        self.btnSave.clicked.connect(self.savefile)

        layout.addWidget(self.btnSave, 6, 6, 1, 1)

        self.imb = QtWidgets.QLabel("Source Image")
        self.imb.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.imb,0,1,4,2)

        self.ima = QtWidgets.QLabel("Target Image")
        self.ima.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.ima, 0, 5, 4, 2)

        self.g_layout = self.static_layout
        self.static_canvas = FigureCanvas(Figure(figsize=(4, 8)))
        layout.addWidget(self.static_canvas,1,3,2,2)

        self._static_ax = self.static_canvas.figure.subplots()
        self._static_ax.set_xlim([0,8])
        self._static_ax.set_ylim([0,8])
        self.static_canvas.figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        self.static_canvas.mpl_connect('pick_event', self.hilighter)

        self.comboBox = QtWidgets.QComboBox()
        for pred in preds[1:]:
            self.comboBox.addItem(pred)

        self.comboBoxLabel = QtWidgets.QLabel("Change relationship")
        layout.addWidget(self.comboBoxLabel, 3, 1, 1, 1)
        layout.addWidget(self.comboBox, 3, 2, 1, 1)
        self.comboBox.activated[str].connect(self.set_predicate)

        self.comboBox_obj = QtWidgets.QComboBox()
        for obj in objs[1:]:
            self.comboBox_obj.addItem(obj)

        layout.addWidget(self.comboBox_obj, 2, 2, 1, 1)
        self.comboBox_objLabel = QtWidgets.QLabel("Replace object")
        layout.addWidget(self.comboBox_objLabel, 2, 1, 1, 1)
        self.comboBox_obj.activated[str].connect(self.set_obj)

        self.comboBox_p2 = QtWidgets.QComboBox()
        for pred in preds[1:]:
            self.comboBox_p2.addItem(pred)
        layout.addWidget(self.comboBox_p2, 4, 4, 1, 1)
        self.comboBox_p2Label = QtWidgets.QLabel("Add relationship")
        layout.addWidget(self.comboBox_p2Label, 4, 3, 1, 1)

        self.comboBox_obj2 = QtWidgets.QComboBox()
        layout.addWidget(self.comboBox_obj2, 5, 4, 1, 1)
        self.comboBox_obj2Label = QtWidgets.QLabel("Connect to object")
        layout.addWidget(self.comboBox_obj2Label, 5, 3, 1, 1)

        self.comboBox_sub2 = QtWidgets.QComboBox()
        for obj in objs[1:]:
            self.comboBox_sub2.addItem(obj)
        layout.addWidget(self.comboBox_sub2, 6, 4, 1, 1)
        self.comboBox_sub2Label = QtWidgets.QLabel("Add new node")
        layout.addWidget(self.comboBox_sub2Label, 6, 3, 1, 1)

        self.btnAddSub = QtWidgets.QPushButton("Add as subject")
        self.btnAddSub.resize(self.btnAddSub.minimumSizeHint())
        self.btnAddSub.clicked.connect(self.add_as_subject)
        layout.addWidget(self.btnAddSub, 3, 3, 1, 1)

        self.btnAddObj = QtWidgets.QPushButton("Add as object")
        self.btnAddObj.resize(self.btnAddObj.minimumSizeHint())
        self.btnAddObj.clicked.connect(self.add_as_object)
        layout.addWidget(self.btnAddObj, 3, 4, 1, 1)

        self.btnPred = QtWidgets.QPushButton("Get Graph")
        self.btnPred.clicked.connect(self.reset_graph)
        self.btnPred.resize(self.btnPred.minimumSizeHint())
        layout.addWidget(self.btnPred, 6, 2, 1, 1)

        self.btn5 = QtWidgets.QPushButton("Remove node")
        self.btn5.clicked.connect(self.remove_node)
        self.btn5.resize(self.btn5.minimumSizeHint())
        layout.addWidget(self.btn5, 4, 1, 1, 2) # 6, 3, 1, 2

        self.btnRem = QtWidgets.QPushButton("Generate Image")
        self.btnRem.clicked.connect(self.gen_image)
        self.btnRem.resize(self.btnRem.minimumSizeHint())
        layout.addWidget(self.btnRem, 6, 5, 1, 1)

    def hilighter(self, event):
        # if we did not hit a node, bail
        if not hasattr(event, 'nodes') or not event.nodes:
                return

        # pull out the graph,
        graph = event.artist.graph

        # clear any non-default color on nodes
        for node, attributes in graph.nodes.data():
            attributes.pop('color', None)
            graph.nodes[node]['color'] = 'w'
            graph.nodes[node]['edgecolor'] = 'g'

        for node in event.nodes:
            self.selected_node = node
            self.graph.nodes[node]['edgecolor'] = 'C1'

            for combo_idx in range(self.comboBox_obj2.count()):
                if self.comboBox_obj2.itemText(combo_idx) == self.selected_node:
                    break
            self.comboBox_obj2.setCurrentIndex(combo_idx)
            graph.nodes[node]['size'] = 2500

            for edge_attribute in graph[node].values():
                edge_attribute['arrowsize'] = 200
                edge_attribute['arrowstyle'] = "fancy"

            # draw object box whenever an object node is clicked
            if self.selected_node.split(".")[0] in vocab["object_idx_to_name"]:
                idx = int(self.selected_node.split(".")[1])
                self.draw_input_image(idx1=idx)

            # draw object boxes + edge whenever a predicate node is clicked
            for [s, p, o] in self.curr_triples:
                if self.selected_node == p:
                    idx1 = int(s.split(".")[1])
                    idx2 = int(o.split(".")[1])
                    self.draw_input_image(idx1=idx1, idx2=idx2)
                    break

        # update the screen
        event.artist.stale = True
        event.artist.figure.canvas.draw_idle()

    def reset_graph(self):
        """
        Initializes new networkx graph from the current state of the objects and triples
        and draws the graph on canvas
        """
        self.graph = nx.DiGraph()
        if self.new_triples is not None:
            curr_triples = self.new_triples.cpu().numpy()
        else:
            curr_triples = self.triples.cpu().numpy()
        self.curr_triples, self.pos = self.preprocess_graph(curr_triples)

        i = 0
        import matplotlib.patches
        astyle = matplotlib.patches.ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)
        for s, p, o in self.curr_triples:
            self.graph.add_node(s)
            if "__image__" not in s and "__in_image__" not in p and "__image__" not in o:
                # make s->p edge thicker than p->o, to indicate direction
                self.graph.add_edge(s, p, width=self.in_edge_width, arrows=True, arrowstyle=astyle)
                self.graph.add_edge(p, o, width=self.out_edge_width)
                i += 1

        for node in self.graph.nodes:
            self.graph.nodes[node]['color'] = 'w'
            self.graph.nodes[node]['edgecolor'] = 'g'
            self.graph.nodes[node]['size'] = 2500

            for edge_attribute in self.graph[node].values():
                edge_attribute['arrows'] = True

        self.set_graph()
        self.graphCounter += 1

    def set_graph(self):
        """
        Draws current graph on canvas
        """
        # add dummy edge if no edges
        # circumvent the fact that drawing graph with no edges is not supported
        if self.graph.number_of_edges() == 0:
            for n in self.graph.nodes:
                self.graph.add_edge(n, n)
        if args.dataset == "clevr":
            layout = "circular"
        else:
            layout = self.g_layout
        self._static_ax.clear()
        self.art = plot_network(self.graph, layout=layout, ax=self._static_ax, #self.g_layout, "spring"
                                node_style=use_attributes(),  # use_attributes(), #node_options, #dict(node_size=50),
                                edge_style=use_attributes(),  # ) #edge_options) # #,
                                node_label_style={'font_size': '10', 'font_weight': 'bold'}) # layout=self.g_layout

        self.art.set_picker(10)
        self._static_ax.figure.canvas.draw_idle()

        # reset combobox for node choices and update with new object list
        for i in range(self.comboBox_obj2.count()):
            self.comboBox_obj2.removeItem(0)

        for obj in self.graph.nodes:
            if obj.split(".")[0] in objs:
                self.comboBox_obj2.addItem(obj)

    def set_predicate(self):
        """
        Sets a new predicate category in a predicate node
        Used in the relationship change mode
        """
        if self.selected_node is not None:
            # extract user input
            new_label = self.comboBox.currentText()
            idx_t = self.selected_node.split(".")[1]
            mapping = {self.selected_node: new_label + "." + idx_t}

            # update list of relationship triples with the change
            self.triples[int(idx_t)][1] = vocab["pred_name_to_idx"][new_label]
            s = self.triples[int(idx_t)][0]
            self.keep_box_idx[s] = 0
            self.keep_image_idx[s] = 0
            self.new_triples = self.triples

            # objects remain the same
            self.new_objs = self.objs

            # update the networkx graph and the list of triples for visualization
            for idx, [s, p, o] in enumerate(self.curr_triples):
                if p == self.selected_node:
                    self.curr_triples[idx][1] = new_label+ "." + idx_t

            self.pos[new_label+ "." + idx_t] = self.pos[self.selected_node]
            del self.pos[self.selected_node]

            self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)

            self.selected_node = self.comboBox.currentText()
            self.mode = "reposition"

        self.set_graph()

    def set_obj(self):
        """
        Sets a new object category in an object node
        Used in the object replacement mode
        """
        if self.selected_node is not None:
            # extract user input
            new_label = self.comboBox_obj.currentText()
            idx_t = self.selected_node.split(".")[1]
            mapping = {self.selected_node: new_label + "." + idx_t}

            # update keep vectors
            self.keep_feat_idx[int(idx_t)] = 0
            self.keep_image_idx[int(idx_t)] = 0

            # for clevr keep object size as it is
            # for vg let it adapt to the new object category
            # position remains the same in both cases
            if args.dataset == "vg" and not eval_utils.is_background(vocab["object_name_to_idx"][new_label]):
                self.keep_box_idx[int(idx_t)] = 0
                self.combine_gt_pred_box_idx[int(idx_t)] = 1

            # update the list of objects with the new object category
            self.objs[int(idx_t)] = vocab["object_name_to_idx"][new_label]
            self.new_objs = self.objs

            # update the networkx graph and the list fo triples with the new object category for visualization
            for idx, [s, p, o] in enumerate(self.curr_triples):
                if s == self.selected_node:
                    self.curr_triples[idx][0] = new_label+ "." + idx_t
                if o == self.selected_node:
                    self.curr_triples[idx][2] = new_label+ "." + idx_t
            self.pos[new_label + "." + idx_t] = self.pos[self.selected_node]
            self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)

            self.selected_node = new_label + "." + idx_t
            self.mode = "replace"

        self.set_graph()

    def remove_node(self):
        """
        Removes an object node and all its connections
        Used in the object removal mode
        """

        if self.selected_node is not None:

            idx = int(self.selected_node.split(".")[1])
            # remove node and all connecting edges
            self.new_objs, self.new_triples, self.boxes, self.imgs_in, self.obj_to_img, _ = \
                        eval_utils.remove_node(self.objs, self.triples,
                                               self.boxes, self.imgs_in, [idx],
                                               torch.zeros_like(self.objs),
                                               torch.zeros_like(self.triples))

            # update keep arrays
            idlist = list(range(self.objs.shape[0]))
            keep_idx = [i for i in idlist if i != idx]
            self.keep_box_idx = self.keep_box_idx[keep_idx]
            self.keep_feat_idx = self.keep_feat_idx[keep_idx]
            self.keep_image_idx = self.keep_image_idx[keep_idx]
            self.added_objs_idx = self.added_objs_idx[keep_idx]
            self.combine_gt_pred_box_idx = self.combine_gt_pred_box_idx[keep_idx]

            self.objs = self.new_objs
            self.triples = self.new_triples

            # update the networkx graph for visualization
            self.reset_graph()

            self.mode = "remove"

        #self.set_graph()

    def add_as_subject(self):
        self.add_triple(is_subject=True)

    def add_as_object(self):
        self.add_triple(is_subject=False)

    def add_triple(self, is_subject=True):
        """
        Adds a triple between an existing object node and a new object
        Used in object addition mode
        """

        # extract user input
        anchor_label, anchor_idx = self.comboBox_obj2.currentText().split(".")
        new_pred = self.comboBox_p2.currentText()

        if torch.cuda.is_available():
            pred_id = torch.tensor(vocab["pred_name_to_idx"][new_pred]).cuda()
        else:
            pred_id = torch.tensor(vocab["pred_name_to_idx"][new_pred])

        new_node = self.comboBox_sub2.currentText()

        if torch.cuda.is_available():
            new_node_idx = torch.tensor(vocab["object_name_to_idx"][new_node]).cuda()
        else:
            new_node_idx = torch.tensor(vocab["object_name_to_idx"][new_node])

        anchor_idx = int(anchor_idx)
        imgbox_idx = self.objs.shape[0] - 1
        newimgbox_idx = imgbox_idx + 1

        new_objs = []
        # add new node in list of objects
        for obj in self.objs[:-1]: # last element is the image node
            new_objs.append(obj)
        new_objs.append(new_node_idx)
        new_objs.append(self.objs[-1])

        new_boxes = []
        # add new box to list of boxes
        for box in self.boxes[:-1]:
            new_boxes.append(box)

        if torch.cuda.is_available():
            new_boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32).cuda())
        else:
            new_boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))

        new_boxes.append(self.boxes[-1])

        # expand and update the keep arrays. Set box and feat to 0 for the new node
        self.keep_feat_idx = torch.cat((self.keep_feat_idx, torch.tensor([[1]], dtype=torch.float32).cuda()), 0)
        self.keep_box_idx = torch.cat((self.keep_box_idx, torch.tensor([[1]], dtype=torch.float32).cuda()), 0)
        self.keep_image_idx = torch.cat((self.keep_image_idx, torch.tensor([[1]], dtype=torch.float32).cuda()), 0)
        self.added_objs_idx = torch.cat((self.added_objs_idx, torch.tensor([[0]], dtype=torch.float32).cuda()), 0)
        self.combine_gt_pred_box_idx = torch.cat((self.combine_gt_pred_box_idx,
                                                  torch.tensor([0], dtype=torch.int64).cuda()), 0)
        self.keep_feat_idx[-2] = 0
        self.keep_box_idx[-2] = 0
        self.added_objs_idx[-2] = 1

        new_triples = []
        # copy already existing triples, with a bit of care for the image modes that have incremented idx
        for [s,p,o] in self.triples:
            if s == imgbox_idx:
                s = newimgbox_idx
            if o == imgbox_idx:
                o = newimgbox_idx
            new_triples.append(torch.LongTensor([s,p,o]).cuda())

        new_pred_pos = len(new_triples)

        predicate_tag = new_pred + "." + str(new_pred_pos)
        new_node_tag = new_node + "." + str(imgbox_idx)
        anchor_tag = anchor_label + "." + str(anchor_idx)
        if is_subject:
            subject_tag = new_node_tag
            object_tag = anchor_tag
            new_triples.append(torch.LongTensor([imgbox_idx, pred_id, anchor_idx]).cuda())
        else:
            object_tag = new_node_tag
            subject_tag = anchor_tag

            new_triples.append(torch.LongTensor([anchor_idx, pred_id, imgbox_idx]).cuda())

        new_triples.append(torch.LongTensor([imgbox_idx,0,newimgbox_idx]).cuda())

        self.new_triples = torch.stack(new_triples)
        self.triples = self.new_triples
        self.boxes = torch.stack(new_boxes)
        self.objs = torch.stack(new_objs)
        self.new_objs = torch.stack(new_objs)

        # update networkx graph for visualization
        self.graph.add_edge(subject_tag, predicate_tag, width=self.in_edge_width)
        self.graph.add_edge(predicate_tag, object_tag, width=self.out_edge_width)
        # place new object in the middle of the canvas (if static_layout2 is used)
        self.pos[new_node_tag] = [0.5, 0.5]
        self.pos[predicate_tag] = [((self.pos[new_node_tag][0] + self.pos[anchor_tag][0]) / 2),
                                   ((self.pos[new_node_tag][1] + self.pos[anchor_tag][1]) / 2)]

        # update style so that the new nodes don't get the default ones
        for node in self.graph.nodes:
            self.graph.nodes[node]['color'] = 'w'
            self.graph.nodes[node]['edgecolor'] = 'g'
            self.graph.nodes[node]['size'] = 2500
            for edge_attribute in self.graph[node].values():
                edge_attribute['arrows'] = True
        self.curr_triples.append([subject_tag, predicate_tag, object_tag])

        self.mode = "addition"
        # update the graph canvas
        self.set_graph()

    def static_layout2(self, graph):
        """
        Let's build my own layout. It places the nodes according to their position in the image
        """
        next_pos = {}

        for k in graph.nodes:
            next_pos[k] = [x * 10 for x in self.pos[k]] # * 10

        return next_pos

    def update_node_pos(self, next_pos, already_there, curr_x, curr_y, t, step=1):
        s, p, o = t

        if s not in next_pos.keys():
            next_pos[s] = [curr_x, curr_y]
            already_there.append(self.hash_func(next_pos[s]))

        if o not in next_pos.keys():
            next_pos[o] = [curr_x+4, curr_y]
            already_there.append(self.hash_func(next_pos[o]))

        if self.hash_func([(next_pos[s][0]+next_pos[o][0])/2,
                  (next_pos[s][1]+next_pos[o][1])/2]) in already_there:

            next_pos[p] = [(next_pos[s][0]+next_pos[o][0])/2+0.5,
                       (next_pos[s][1]+next_pos[o][1])/2+0.5]

        else:
            next_pos[p] = [(next_pos[s][0]+next_pos[o][0])/2,
                       (next_pos[s][1]+next_pos[o][1])/2]

        already_there.append(self.hash_func(next_pos[p]))

        for new_triplet in self.curr_triples:
            if curr_x >= 4:
                break
            if new_triplet[0] == o and new_triplet[1] not in next_pos.keys():
                if "image" not in new_triplet[2]:
                    curr_x = next_pos[o][0]
                    curr_x, curr_y, step = self.update_node_pos(next_pos, already_there, curr_x, curr_y,
                                                                new_triplet, step=2)

        for new_triplet in self.curr_triples:
            if new_triplet[0] == s and new_triplet[1] not in next_pos.keys() and new_triplet[2] == o:
                if "image" not in new_triplet[2]:
                    curr_x = next_pos[s][0]
                    #curr_y += 1
                    curr_x, curr_y, step = self.update_node_pos(next_pos, already_there, curr_x, curr_y, new_triplet,
                                                                max(step, 1))

        for new_triplet in self.curr_triples:
            if new_triplet[0] == s and new_triplet[1] not in next_pos.keys():
                if "image" not in new_triplet[2]:
                    curr_x = next_pos[s][0]
                    if new_triplet[2] not in next_pos.keys():
                        curr_y += 1
                    curr_x, curr_y, step = self.update_node_pos(next_pos, already_there, curr_x, curr_y, new_triplet,
                                                                max(step, 1))

        for new_triplet in self.curr_triples:
            if new_triplet[2] == o and new_triplet[1] not in next_pos.keys():
                if "image" not in new_triplet[2]:
                    curr_x = next_pos[s][0]
                    if new_triplet[0] not in next_pos.keys():
                        curr_y += 1
                    curr_x, curr_y, step = self.update_node_pos(next_pos, already_there, curr_x, curr_y, new_triplet,
                                                                max(step, 1))

        return curr_x, curr_y, step

    def static_layout(self, graph):
        """
        Let's build my own layout to reduce node occlusion. Hopefully better than random
        """
        next_pos = {}
        already_there = []
        limit = 5
        curr_x = 0
        curr_y = 0
        step = 1

        # add triplet positions
        for [s, p, o] in self.curr_triples:
            if "image" not in s and "image" not in p and "image" not in o and p not in next_pos.keys():
                curr_x, curr_y, step = self.update_node_pos(next_pos, already_there, curr_x, curr_y, [s, p, o], step)
                curr_y += 1

        # add unconnected node positions
        curr_x = 0
        for k in graph.nodes:
            if k not in next_pos.keys():
                next_pos[k] = [step*curr_x, curr_y] # * 10
                curr_x += 1
                if curr_x == limit:
                    curr_x = 0
                    curr_y += 0.5

        # reverse y axis
        for k in next_pos.keys():
            next_pos[k][1] = 20 - next_pos[k][1]

        return next_pos

    def hash_func(self, x):
        return x[0] * 20 + x[1]

    def draw_input_image(self, idx1=None, idx2=None, new_image=False):
        """
        Draws input image, with additional box visualizations if a node is selected
        - idx1: idx of object1 (subject)
        - idx2: idx of object2 (object)
        """
        image = np.copy(self.image)

        if idx1 is not None and idx2 is None:
            # its an object node
            image = eval_utils.draw_image_box(image, self.boxes[idx1].cpu().numpy())

        if idx1 is not None and idx2 is not None:
            # its a predicate node
            image = eval_utils.draw_image_edge(image, self.boxes[idx1].cpu().numpy(), self.boxes[idx2].cpu().numpy())
            image = eval_utils.draw_image_box(image, self.boxes[idx1].cpu().numpy())
            image = eval_utils.draw_image_box(image, self.boxes[idx2].cpu().numpy())

        image = QtGui.QImage(image, image.shape[1], \
                             image.shape[0], QtGui.QImage.Format_RGB888)

        self.pixmap = QtGui.QPixmap(image)
        self.imb.setPixmap(self.pixmap.scaled(200,200))

        if new_image:
            self.ima.setVisible(0)
            self.imLoadCounter += 1

    def getfile(self):
        """
        Loads input data
        """
        self.batch = next(self.data_loader)

        # self.imgs, self.objs, self.boxes, self.triples, self.obj_to_img, self.triple_to_img, self.imgs_in = \
        #     [x.cuda() for x in self.batch]
        self.imgs, self.objs, self.boxes, self.triples, self.obj_to_img, self.triple_to_img, self.imgs_in = \
                [x for x in self.batch]

        self.keep_box_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_feat_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.keep_image_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
        self.combine_gt_pred_box_idx = torch.zeros_like(self.objs)
        self.added_objs_idx = torch.zeros_like(self.objs.unsqueeze(1), dtype=torch.float)

        self.new_triples, self.new_objs = None, None

        image = imagenet_deprocess_batch(self.imgs)
        image = image[0].numpy().transpose(1, 2, 0).copy()

        self.image = image
        self.draw_input_image(new_image=True)


    def savefile(self):
        """
        Saves generated image
        """
        p = self.ima.pixmap()
        p.save("./edited_ims/" +str(self.imLoadCounter) + "_" + str(self.imCounter) + ".png", "PNG")

    def gen_image(self):
        """
        Generates an image, as indicated by the modified graph
        """
        if self.new_triples is not None:
            triples_ = self.new_triples
        else:
            triples_ = self.triples

        query_feats = None

        model_out = self.model(self.new_objs, triples_, None,
                               boxes_gt=self.boxes, masks_gt=None, src_image=self.imgs_in, mode=self.mode,
                               query_feats=query_feats, keep_box_idx=self.keep_box_idx,
                               keep_feat_idx=self.keep_feat_idx, combine_gt_pred_box_idx=self.combine_gt_pred_box_idx,
                               keep_image_idx=self.keep_image_idx, random_feats=args.random_feats,
                               get_layout_boxes=False)

        imgs_pred, boxes_pred, masks_pred, noised_srcs, _, layout_boxes = model_out

        image = imagenet_deprocess_batch(imgs_pred)
        image = image[0].detach().numpy().transpose(1, 2, 0).copy()
        if args.update_input:
            self.image = image.copy()

        image = QtGui.QImage(image, image.shape[1],
                             image.shape[0], QtGui.QImage.Format_RGB888)

        im_pm = QtGui.QPixmap(image)
        self.ima.setPixmap(im_pm.scaled(200,200))
        self.ima.setVisible(1)
        self.imCounter += 1

        if args.update_input:
            # reset everything so that the predicted image is now the input image for the next step
            self.imgs = imgs_pred.detach().clone()
            self.imgs_in = torch.cat([self.imgs, torch.zeros_like(self.imgs[:,0:1,:,:])], 1)
            self.draw_input_image()
            self.boxes = layout_boxes.detach().clone()
            self.keep_box_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.keep_feat_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.keep_image_idx = torch.ones_like(self.objs.unsqueeze(1), dtype=torch.float)
            self.combine_gt_pred_box_idx = torch.zeros_like(self.objs)
        else:
            # input image is still the original one - don't reset anything
            # if an object is added for the first time, the GT/input box is still a dummy (set in add_triple)
            # in this case, we update the GT/input box, using the box predicted from SGN,
            # so that it can be used in future changes that rely on the GT/input box, e.g. replacement
            self.boxes = self.added_objs_idx * layout_boxes.detach().clone() + (1 - self.added_objs_idx) * self.boxes
            self.added_objs_idx = torch.zeros_like(self.objs.unsqueeze(1), dtype=torch.float)

    def preprocess_graph(self, triples):
        """
        Prepares graphs in the right format for networkx
        """
        if self.new_objs is not None:
            objs = self.new_objs.cpu().numpy()
        else:
            objs = self.objs
        new_triples = []
        boxes = self.boxes.cpu().numpy()
        boxes_ = {}
        triple_idx = 0
        for [s, p, o] in triples:
            s2 = vocab['object_idx_to_name'][objs[s]] + "." + str(s)
            o2 = vocab['object_idx_to_name'][objs[o]] + "." + str(o)
            p2 = vocab['pred_idx_to_name'][p] + "." + str(triple_idx)
            new_triples.append([s2, p2, o2])

            x1_o, y1_o, x2_o, y2_o = boxes[o]
            x1_s, y1_s, x2_s, y2_s = boxes[s]
            xc_o = x1_o + (x2_o - x1_o) / 2
            yc_o = y1_o + (y2_o - y1_o) / 2
            xc_s = x1_s + (x2_s - x1_s) / 2
            yc_s = y1_s + (y2_s - y1_s) / 2
            x_p = (xc_o + xc_s) / 2
            y_p = (yc_o + yc_s) / 2
            if vocab['pred_idx_to_name'][p] in boxes_.keys():
                old_xc, old_yc = boxes_[vocab['pred_idx_to_name'][p]]
                boxes_[vocab['pred_idx_to_name'][p] + "." + str(triple_idx)] = [1 - ((x_p + old_xc) / 2),
                                                                                1 - ((y_p + old_yc) / 2)]
            else:
                boxes_[vocab['pred_idx_to_name'][p] + "." + str(triple_idx)] = [1 - x_p, 1 - y_p]

            triple_idx += 1

        for i, obj in enumerate(objs):
            x1, y1, x2, y2 = boxes[i]
            xc = x1 + (x2 - x1) / 2
            yc = y1 + (y2 - y1) / 2
            boxes_[vocab['object_idx_to_name'][obj] + "." + str(i)] = [1 - xc, 1 - yc]

        return new_triples, boxes_


def convert_idx_to_name(triple):
    global vocab
    triples = []
    for i in range(len(triple)):
        s = vocab["object_idx_to_name"][triple[i][0]]
        o = vocab["object_idx_to_name"][triple[i][2]]
        p = vocab["pred_idx_to_name"][triple[i][1]]
        triples.append([s, p, o])
    return triples


def convert_name_to_idx(triple):
    global vocab
    triples = []
    for i in range(len(triple)):
        s = vocab["object_name_to_idx"][triple[i][0]]
        o = vocab["object_name_to_idx"][triple[i][2]]
        p = vocab["pred_name_to_idx"][triple[i][1]]
        triples.append([s, p, o])
    return triples


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
