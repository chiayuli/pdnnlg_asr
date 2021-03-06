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

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

from models.dnn import DNN

from io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from utils.utils import parse_arguments
from utils.learn_rates import _lrate2file, _file2lrate

from utils.network_config import NetworkConfig
from learning.sgd import train_sgd, validate_by_minibatch

if __name__ == '__main__':

    # check the arguments
    arg_elements = [sys.argv[i] for i in range(1, len(sys.argv))]
    arguments = parse_arguments(arg_elements)
    required_arguments = ['train_data', 'valid_data', 'nnet_spec', 'wdir']
    for arg in required_arguments:
        if arguments.has_key(arg) == False:
            print "Error: the argument %s has to be specified" % (arg); exit(1)

    # mandatory arguments
    train_data_spec = arguments['train_data']
    valid_data_spec = arguments['valid_data']
    nnet_spec = arguments['nnet_spec']
    wdir = arguments['wdir']

    # parse network configuration from arguments, and initialize data reading
    cfg = NetworkConfig()
    cfg.parse_config_dnn(arguments, nnet_spec)
    cfg.init_data_reading(train_data_spec, valid_data_spec)

    # check working dir to see whether it's resuming training
    resume_training = False
    if os.path.exists(wdir + '/nnet.tmp_DNN') and os.path.exists(wdir + '/training_state.tmp_DNN'):
        resume_training = True
        cfg.lrate = _file2lrate(wdir + '/training_state.tmp_DNN')
        log('> ... found nnet.tmp_DNN and training_state.tmp_DNN, now resume training from epoch ' + str(cfg.lrate.epoch))

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
    log('> ... building the model')

    # setup model
    if cfg.do_dropout:
        dnn = DNN_Dropout(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)
    else:
        dnn = DNN(numpy_rng=numpy_rng, theano_rng = theano_rng, cfg = cfg)

    # initialize model parameters
    if resume_training:
        _file2nnet(dnn, filename = wdir + '/nnet.tmp_DNN')

    # get the training, validation and testing function for the model
    log('> ... getting the finetuning functions')
    train_fn, valid_fn = dnn.build_finetune_functions(
                (cfg.train_x, cfg.train_y), (cfg.valid_x, cfg.valid_y),
                batch_size=cfg.batch_size)

    log('> ... finetuning the model')
    count = 0
    while (cfg.lrate.get_rate() != 0):
        # one epoch of sgd training
        train_acc, train_err = train_sgd(train_fn, cfg)
        log('> epoch %d, lrate %f, training accuracy %f' % (cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(train_acc)) + '(%)')
        # validation 
        valid_acc, valid_err = validate_by_minibatch(valid_fn, cfg)
        log('> epoch %d, lrate %f, validate accuracy %f' % (cfg.lrate.epoch, cfg.lrate.get_rate(), 100*numpy.mean(valid_acc)) + '(%)')
        cfg.lrate.get_next_rate(current_error = 100*numpy.mean(valid_err))
        # output nnet parameters and lrate, for training resume
        if cfg.lrate.epoch % cfg.model_save_step == 0:
            _nnet2file(dnn, filename=wdir + '/nnet.tmp_DNN')
            _lrate2file(cfg.lrate, wdir + '/training_state.tmp_DNN') 
    
	
    # save the model and network configuration
    if cfg.param_output_file != '':
        _nnet2file(dnn, filename=cfg.param_output_file, input_factor = cfg.input_dropout_factor, factor = cfg.dropout_factor)
        log('> ... the final PDNN model parameter is ' + cfg.param_output_file)
    if cfg.cfg_output_file != '':
        _cfg2file(dnn.cfg, filename=cfg.cfg_output_file)
        log('> ... the final PDNN model config is ' + cfg.cfg_output_file)

    # output the model into Kaldi-compatible format
    if cfg.kaldi_output_file != '':
        dnn.write_model_to_kaldi(cfg.kaldi_output_file)
        log('> ... the final Kaldi model is ' + cfg.kaldi_output_file) 

    # remove the tmp_DNN files (which have been generated from resuming training) 
    #os.remove(wdir + '/nnet.tmp_DNN')
    #os.remove(wdir + '/training_state.tmp_DNN') 


