The code was originally written by [@kaldipdnn](https://github.com/yajiemiao/kaldipdnn) and modified by [@Chia-Yu Li](https://github.com/chiayuli).

# Citations
```
@INPROCEEDINGS{8578047,
  author={Li, Chia Yu and Vu, Ngoc Thang},
  booktitle={Speech Communication; 13th ITG-Symposium}, 
  title={Densely Connected Convolutional Networks for Speech Recognition}, 
  year={2018},
  volume={},
  number={},
  pages={1-5},
  keywords={},
  doi={}}
```

# Datasets
* SWBD (LDC97S62)
* WSJ (LDC97S62 and LDC94S13A)

# Toolkits
install_pfile_utils.sh

# run scripts
* For SWBD task, the recipes of training DNN/CNN/CNN-LACE based acoustic models are in the following folders
- run_swbd_pdnnlg
- pdnnlg 
- steps_pdnnlg 

Please copy all three folders under Kaldi-trunck/egs/swbd/s5c/ , and follow these steps:
	1) Download/Install Kaldi
	2) Use $KALDI_ROOT/egs/swbd/s5c/run.sh recipe to get GMM-HMMs SAT model (tri4) 
	3) Use run_swbd/pdnnlg/run-dnn.sh recipe to train DNN-HMMs model with fMLLR features, and do decoding
	       run_swbd/pdnnlg/run-dnn-fbank.sh recipe to train DNN-HMMs model with filter-bank features, and do decoding
	       run_swbd/pdnnlg/run-cnn.sh recipe to train CNN-HMMs model with filter-bank features, and do decoding
	       run_swbd/pdnnlg/run-cnn-lacea.sh recipe to train CNN-LACE-HMMs model with filter-bank, and do decoding
	       
### 80 hours Wall Street Journal Task ###
wsj folder includes the recipes of training DNN/CNN/CNN-LACE based Acoustic Models with WSJ Data Set for ASR system
In contains
	run_wsj_pdnnlg/  
	pdnnlg/ 
	steps_pdnnlg/ 

Please copy all three folders under Kaldi-trunck/egs/wsj/s5/ , and follow these steps:
	1) Download/Install Kaldi
	2) Use $KALDI_ROOT/egs/wsj/s5/run.sh recipe to get GMM-HMMs SAT model (tri4b) 
	3) Use run_wsj/pdnnlg/run-dnn.sh recipe to train DNN-HMMs model with fMLLR features, and do decoding
	       run_wsj/pdnnlg/run-dnn-fbank.sh recipe to train DNN-HMMs model with filter-bank featuers, and do decoding
	       run_wsj/pdnnlg/run-cnn.sh recipe to train CNN-HMMs model with filter-bank, and do decoding
	       run_wsj/pdnnlg/run-cnn-lacea.sh recipe to train CNN-LACE-HMMs model with filter-bank, and do decoding
	       run_wsj/pdnnlg/run-densenet.sh recipe to train DENSENET-HMMs model with filter-bank, and do decoding
	       
