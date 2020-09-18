# GCNN-Explainability
Unofficial implementation of ["Explainability Methods for Graph Convolutional Neural Networks" from HRL Laboratories](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf). I also added a new method called unsigned Grad-CAM (UGrad-CAM) which shows both positive and negative contributions from nodes.

![2](https://user-images.githubusercontent.com/10405248/70907972-2dcc9280-1fd8-11ea-820a-f4be4521f8be.png)

To train a GCNN on the BBBP dataset and save the model weights: `python train.py`

To load the weights of a trained GCNN and generate explanations: `python explain.py`
