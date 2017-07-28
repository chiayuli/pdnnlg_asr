# Copyright 2013    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Various functions to write models from nets to files, and to read models from
# files to nets

import numpy as np
import os
import sys
import cPickle

from StringIO import StringIO
import json

import theano
import theano.tensor as T

from datetime import datetime

from io_func import smart_open

import lasagne

# print log to standard output
def log(string):
    sys.stderr.write('[' + str(datetime.now()) + '] ' + str(string) + '\n')

# convert an array to a string
def array_2_string(array):
    str_out = StringIO()
    np.savetxt(str_out, array)
    return str_out.getvalue()

# convert a string to an array
#def string_2_array(string):
#    str_in = StringIO(string)
#    return np.loadtxt(str_in)

def string_2_array(string):
    str_in = StringIO(string)
    array_tmp = np.loadtxt(str_in)
    if len(array_tmp.shape) == 0:
        return np.array([array_tmp])
    return array_tmp

def _nnet2file(model, set_layer_num = -1, filename='nnet.out', start_layer = 0, input_factor = 0.0, factor=[]):
    print "[Debug] _nnet2file"
    nnet_dict = {}
    weight = []
    b = []
    layers = model.layers
    params = lasagne.layers.get_all_params(model.network)

    for i in range(0, len(params), 2):
        weight.append(params[i].get_value())
        b.append(params[i+1].get_value())

    for i in range(len(layers)):
       dict_a = 'W' + str(i)
       dropout_factor = 0.0
       print "[Debug] w.shape: ", weight[i].shape
   
       if layers[i].type == 'fc':
           nnet_dict[dict_a] = array_2_string((1.0 - dropout_factor) * weight[i])
       elif layers[i].type == 'conv':
           filter_shape = layers[i].filter_shape
           print "[Debug] filter_shape: ", filter_shape
           for next_X in xrange(filter_shape[0]):
               for this_X in xrange(filter_shape[1]):
                   new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                   nnet_dict[new_dict_a] = array_2_string(weight[i][next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(b[i])

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

# save the config classes; since we are using pickle to serialize the whole class, it's better to set the
# data reading and learning rate interfaces to None.
def _cfg2file(cfg, filename='cfg.out'):
    cfg.lrate = None
    cfg.train_sets = None; cfg.train_xy = None; cfg.train_x = None; cfg.train_y = None
    cfg.valid_sets = None; cfg.valid_xy = None; cfg.valid_x = None; cfg.valid_y = None
    cfg.activation = None  # saving the rectifier function causes errors; thus we don't save the activation function
                           # the activation function is initialized from the activation text ("sigmoid") when the network
                           # configuration is loaded
    with smart_open(filename, "wb") as output:
        cPickle.dump(cfg, output, cPickle.HIGHEST_PROTOCOL)

def _file2nnet(model, set_layer_num = -1, filename='nnet.in',  factor=1.0):
    print "[Debug] _file2nnet"
    n_layers = lasagne.layers.get_all_layers(model.network)
    nnet_dict = {}
    old_params = lasagne.layers.get_all_params(model.network)
    new_params = []

    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(len(model.layers)):
        dict_a = 'W' + str(i)
        
        layer = model.layers[i]
        if layer.type == 'fc':
            mat_shape = old_params[2*i].get_value().shape
            W_mat = factor * np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_mat)
        elif layer.type == 'conv':
            filter_shape = layer.filter_shape
            W_array = old_params[2*i].get_value()
            for next_X in xrange(filter_shape[0]):
                for this_X in xrange(filter_shape[1]):
                    new_dict_a = dict_a + ' ' + str(next_X) + ' ' + str(this_X)
                    mat_shape = W_array[next_X, this_X, :, :].shape
                    W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[new_dict_a]), dtype=theano.config.floatX).reshape(mat_shape)
            new_params.append(W_array)

        dict_a = 'b' + str(i)
        b_vec = np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX)
        new_params.append(b_vec)

    # update params to network
    lasagne.layers.set_all_param_values(model.network, new_params)

def _cnn2file(conv_layers, filename='nnet.out', input_factor = 1.0, factor=[]):
    n_layers = len(conv_layers)
    nnet_dict = {}
    for i in xrange(n_layers):
       conv_layer = conv_layers[i]
       filter_shape = conv_layer.filter_shape

       dropout_factor = 0.0
       if i == 0:
           dropout_factor = input_factor
       if i > 0 and len(factor) > 0:
           dropout_factor = factor[i-1]

       for next_X in xrange(filter_shape[0]):
           for this_X in xrange(filter_shape[1]):
               dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
               nnet_dict[dict_a] = array_2_string(dropout_factor * (conv_layer.W.get_value())[next_X, this_X])

       dict_a = 'b' + str(i)
       nnet_dict[dict_a] = array_2_string(conv_layer.b.get_value())

    with smart_open(filename, 'wb') as fp:
        json.dump(nnet_dict, fp, indent=2, sort_keys = True)
        fp.flush()

def _file2cnn(conv_layers, filename='nnet.in', factor=1.0):
    n_layers = len(conv_layers)
    nnet_dict = {}

    with smart_open(filename, 'rb') as fp:
        nnet_dict = json.load(fp)
    for i in xrange(n_layers):
        conv_layer = conv_layers[i]
        filter_shape = conv_layer.filter_shape
        W_array = conv_layer.W.get_value()

        for next_X in xrange(filter_shape[0]):
            for this_X in xrange(filter_shape[1]):
                dict_a = 'W' + str(i) + ' ' + str(next_X) + ' ' + str(this_X)
                W_array[next_X, this_X, :, :] = factor * np.asarray(string_2_array(nnet_dict[dict_a]))

        conv_layer.W.set_value(W_array)

        dict_a = 'b' + str(i)
        conv_layer.b.set_value(np.asarray(string_2_array(nnet_dict[dict_a]), dtype=theano.config.floatX))
