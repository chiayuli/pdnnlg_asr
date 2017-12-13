# Copyright 2014    Yajie Miao    Carnegie Mellon University

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

import cPickle
import gzip
import os
import sys
import time
import collections

import numpy
import json

import theano
import theano.tensor as T

from io_func import smart_open
import lasagne
from lasagne.layers import (InputLayer, Conv2DLayer, ConcatLayer, DenseLayer,
                             DropoutLayer, Pool2DLayer, GlobalPoolLayer,
                             NonlinearityLayer)

from lasagne.nonlinearities import rectify, softmax
#try:
#    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
#except ImportError:
from lasagne.layers import BatchNormLayer

class ConvLayer_Config(object):
    """configuration for a convolutional layer """

    def __init__(self, input_shape=(3,1,28,28), filter_shape=(2, 1, 5, 5),
                 poolsize=(1, 1), activation=T.tanh, output_shape=(3,1,28,28),
                 flatten = False):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.poolsize = pool_size
        self.output_shape = output_shape
        self.flatten = flatten

class layer_info(object):
    def __init__(self, layer_type='fc', filter_shape=(256, 1024, 3,3), num_params=2):
        self.type = layer_type
        self.filter_shape = filter_shape
        self.num_params = num_params
        self.W = None
        self.b = None

    def set_filter_shape(self, shape):
        self.filter_shape = shape

class DENSENET(object):

    def __init__(self, numpy_rng, theano_rng=None, cfg = None, testing = False, input = None):

        self.layers = []
        self.params = []
        self.network = None

        self.cfg = cfg
        self.conv_layer_configs = cfg.conv_layer_configs
        self.conv_activation = cfg.conv_activation
        self.use_fast = cfg.use_fast

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        if input == None:
            self.x = T.matrix('x')
            self.input = T.tensor4('inputs')
        else:
            self.x = input
        self.y = T.ivector('y')

        self.conv_layer_num = len(self.conv_layer_configs)

        config = self.conv_layer_configs[0]
        d1 = config['input_shape'][1]
        d2 = config['input_shape'][2]
        d3 = config['input_shape'][3]

        self.input = self.x.reshape((-1,d1,d2,d3))
        #print "[Debug] input_shape: ", config['input_shape']
        print "[Debug] d1,d2,d3: ", d1, d2, d3
        num_blocks = 3
        depth = 40
        growth_rate = 12
        dropout=0
    
        self.network = lasagne.layers.InputLayer(shape=(None, d1, d2, d3), input_var=self.input)
        self.network = lasagne.layers.Conv2DLayer(self.network, num_filters=256, filter_size=3, 
                            W=lasagne.init.HeNormal(gain='relu'),
                            b=None, nonlinearity=None, name='pre_conv')
        self.layers.append(layer_info('conv', (256, 3, 3, 3), 1)) # W
        # note: The authors' implementation does *not* have a dropout after the
        #       initial convolution. This was missing in the paper, but important.
        # if dropout:
        #     network = DropoutLayer(network, dropout)
        # dense blocks with transitions in between
        n = (depth - 1) // num_blocks
        for b in range(num_blocks):
            nam = 'block' + str(b + 1)
            self.network = self.dense_block(self.network, n - 1, growth_rate, dropout, nam)
            if b < num_blocks - 1:
                nam = 'block' + str(b + 1) + '_trs'
                self.network = self.transition(self.network, dropout, nam)

            net_shape_tmp = lasagne.layers.get_output_shape(self.network)
            print "[Debug] net_shape: ", b, net_shape_tmp

        # post processing until prediction
        self.network = BatchNormLayer(self.network, name='post_bn')
        self.layers.append(layer_info('bn', num_params=2))

        self.network = NonlinearityLayer(self.network, nonlinearity=rectify,
                                name='post_relu')
        self.layers.append(layer_info('relu'))

        self.network = GlobalPoolLayer(self.network, name='post_pool')
        self.layers.append(layer_info('pool'))

        net_shape = lasagne.layers.get_output_shape(self.network)
        print "[Debug] before fc, net_shape: ", net_shape

        self.conv_output_dim = net_shape[1]
        cfg.n_ins = net_shape[1]


        self.network = DenseLayer(self.network, cfg.n_outs, nonlinearity=softmax,
                             W=lasagne.init.HeNormal(gain=1), name='output')

        self.layers.append(layer_info('fc', (cfg.n_ins, cfg.n_outs), 2))

        # define the cost and error
        prediction = lasagne.layers.get_output(self.network)
        self.finetune_cost = lasagne.objectives.categorical_crossentropy(prediction, self.y).mean()
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)

    def dense_block(self, network, num_layers, growth_rate, dropout, name_prefix):
        # concatenated 3x3 convolutions
        for n in range(num_layers):
            nam=name_prefix + '_l' + str(n + 1)
            conv = self.bn_relu_conv(network, channels=growth_rate,
                                filter_size=3, dropout=dropout,
                                name_prefix=nam)
            nam=name_prefix + '_l' + str(n + 1) + '_join'
            network = ConcatLayer([network, conv], axis=1,
                                  name=nam)
            self.layers.append(layer_info('concat'))

        return network


    def transition(self, network, dropout, name_prefix):
        # a transition 1x1 convolution followed by avg-pooling
        network = self.bn_relu_conv(network, channels=network.output_shape[1],
                               filter_size=1, dropout=dropout,
                               name_prefix=name_prefix)
        nam=name_prefix + '_pool'
        network = Pool2DLayer(network, 2, mode='average_inc_pad', name=nam)
        self.layers.append(layer_info('pool')) 

        return network

    def bn_relu_conv(self, network, channels, filter_size, dropout, name_prefix):
        nam=name_prefix + '_bn'
        network = BatchNormLayer(network, name=nam)
        self.layers.append(layer_info('bn', num_params=4))
 
        nam=name_prefix + '_relu'
        network = NonlinearityLayer(network, nonlinearity=rectify, name=nam)
        self.layers.append(layer_info('relu')) 

        nam=name_prefix + '_conv'
        network = Conv2DLayer(network, channels, filter_size, pad='same',
                              W=lasagne.init.HeNormal(gain='relu'),
                              b=None, nonlinearity=None,
                              name=nam)
        self.layers.append(layer_info('conv', (channels, channels, 3, 3), 1))

        if dropout:
            network = DropoutLayer(network, dropout)
        return network


    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        print "[Debug] build_finetune_functions..."
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')

        prediction = lasagne.layers.get_output(self.network)
        y_pred = T.argmax(prediction, axis=1)
        acc = T.mean(T.eq(y_pred, self.y), dtype=theano.config.floatX)
        err = T.mean(T.neq(y_pred, self.y), dtype=theano.config.floatX)

        l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
                self.network, lasagne.regularization.l2, {'trainable': True})

        updates = lasagne.updates.nesterov_momentum(
                self.finetune_cost + l2_loss, self.params, learning_rate, momentum=0.9)

        train_fn = theano.function(inputs=[index, learning_rate],
              outputs=[acc, err],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})
 
        valid_fn = theano.function(inputs=[index],
              outputs=[acc, err],
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn

    def build_extract_feat_function(self, output_layer):

        feat = T.matrix('feat')

        layers = lasagne.layers.get_all_layers(self.network)
        print "[Debug] build_extract_feat_function"
        print "[Debug] layers: ", len(layers)
        print "[Debug] layers[-2] = ", layers[-2]
        print "[Debug] dim: ", self.conv_output_dim
        intermediate = lasagne.layers.get_output(layers[-2])
        output = intermediate.reshape((-1, self.conv_output_dim))
        out_da = theano.function([feat], output, updates = None, givens={self.x:feat}, on_unused_input='warn')
        return out_da

    def write_model_to_kaldi(self, file_path, with_softmax = True):
        # determine whether it's BNF based on layer sizes
        print "[Debug] write_model_to_kaldi"

        fout = smart_open(file_path, 'wb')
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        
        activation_text = '<softmax>'
        W_mat = params[-2].get_value()
        b_vec = params[-1].get_value()
        input_size, output_size = W_mat.shape
        W_layer = []; b_layer = ''

        for rowX in xrange(output_size):
            W_layer.append('')
 
        for x in xrange(input_size):
            for t in xrange(output_size):
                W_layer[t] = W_layer[t] + str(W_mat[x][t]) + ' '

        for x in xrange(output_size):
            b_layer = b_layer + str(b_vec[x]) + ' '

        fout.write('<affinetransform> ' + str(output_size) + ' ' + str(input_size) + '\n')
        fout.write('[' + '\n')
        for x in xrange(output_size):
            fout.write(W_layer[x].strip() + '\n')
        fout.write(']' + '\n')
        fout.write('[ ' + b_layer.strip() + ' ]' + '\n')
        if activation_text == '<maxout>':
            fout.write(activation_text + ' ' + str(output_size/self.pool_size) + ' ' + str(output_size) + '\n')
        else:
            fout.write(activation_text + ' ' + str(output_size) + ' ' + str(output_size) + '\n')
        fout.close()
