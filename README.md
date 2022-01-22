# MVA RecVis course final project : 

## Goal of the project : Detecting Temporal Boundaries in Sign Language videos. 

Sign language automatic indexing is an important challenge to develop better communication tools for the deaf community. However, annotated datasets for sign langage are limited, and there are few people with skills to anotate such data, which makes it hard to train performant machine learning models. An important challenge is therefore to : 

*  Increase available training datasets. 
*  Make labeling easier for professionnals to reduce risks of bad annotations. 

In this context, techniques have emerged to perform automatic sign segmentation in videos, by marking the boundaries between individual signs in sign language videos. The developpment of such tools offers the potential to alleviate the limited supply of labelled dataset currently available for sign research. 

## Previous works and personal contribution : 

This repository provides code for the Object Recognition & Computer Vision (RecVis) course Final project. For more details please refer the the project report `report.pdf`.
In this project, we reproduced the results obtained on the following paper (by using the code from this [ repository](https://github.com/RenzKa/sign-segmentation)) :  

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)

We used the pre-extracted frame-level features obtained by applying the I3D model on videos to retrain the MS-TCN architecture for frame-level binary classification and reproduce the papers results. The `test [302MB]` folder proposes a notebook for reproducing the original paper results, with a meanF1B = 68.68 on the evaluation set of the BSL Corpus. 

We further implemented new models in order to improve this result. We wanted to try attention based models as they have received recently a huge gain of interest in the vision research community. We first tried to train a Vanilla Transformer Encoder from scratch, but the results were not satisfactory. 

- [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin:  (2018). 

We then implemented the ASFormer model (Transformer for Action Segementation), using this [code](https://github.com/ChinaYi/ASFormer) : a hybrid transformer model using some interesting ideas from the MS-TCN architecture. The motivations behind the model and its architecture are detailed in the following paper : 

- [ASFormer: Transformer for Action Segmentation](https://arxiv.org/abs/2110.08568), Fangqiu Yi, Hongyu Wen, Tingting Jiang (2021).


We trained this model on the I3D extracted features and obtained an improvement over the MS-TCN architecture. The results are given in the following table : 

*TODO*

![demo](demo/results/demo.gif)


## Contents
* [Setup](#setup)
* [Data and models](#data-and-models)
* [Demo](#demo)
* [Training](#training)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Setup

``` bash
# Clone this repository
git clone https://github.com/kamilakesbi/MVARecVisProject.git
cd MVARecVisProject/
# Create signseg_env environment
conda env create -f environment.yml
conda activate signseg_env
```


## Data and models
You can download the pretrained models (I3D and MS-TCN) (`models.zip [302MB]`) and data (`data.zip [5.5GB]`) used in the experiments [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing) or by executing `download/download_*.sh`. The unzipped `data/` and `models/` folders should be located on the root directory of the repository (for using the demo downloading the `models` folder is sufficient).

You can also download our best pretrained asformer model weights here : TO DO ! 


### Data:
Please cite the original datasets when using the data: [BSL Corpus](https://bslcorpusproject.org/cava/acknowledgements-and-citation/) 
The authors of [github.com/RenzKa/sign-segmentation](https://github.com/RenzKa/sign-segmentation) provide the pre-extracted features and metadata. See [here](data/README.md) for a detailed description of the data files. 
- Features: `data/features/*/*/features.mat`
- Metadata: `data/info/*/info.pkl`

### Models:
- I3D weights, trained for sign classification: `models/i3d/*.pth.tar`
- MS-TCN weights for the demo (see tables below for links to the other models): `models/ms-tcn/*.model`
- As_former weights of our best model : `models/asformer/*.model`

The folder structure should be as below:
```
sign-segmentation/models/
  i3d/
    i3d_kinetics_bsl1k_bslcp.pth.tar
    i3d_kinetics_bslcp.pth.tar
    i3d_kinetics_phoenix_1297.pth.tar
  ms-tcn/
    mstcn_bslcp_i3d_bslcp.model
  asformer/
    best_asformer_bslcp.model
```
## Demo
The demo folder contains a sample script to estimate the segments of a given sign language video, one can run `demo.py`to get a visualization on a sample video.

```
cd demo
python demo.py
```

The demo will: 
1. use the `models/i3d/i3d_kinetics_bslcp.pth.tar` pretrained I3D model to extract features,
2. use the `models/asformer/best_asformer_model.model` pretrained ASFormer model to predict the segments out of the features.
3. save results.

## Training
To train I3D please refer to [github.com/RenzKa/sign-segmentation](https://github.com/RenzKa/sign-segmentation). To train ASFormer on the pre-extracted I3D features run `main.py`, you can change hyperparameters in the arguments inside the file. Or you can run the notebook in the folder `test_asformer`.

* Influence of I3D training (fully-supervised segmentation results on BSL Corpus) : 

|ID | Model | mF1B | mF1S | 
|   -   |   -  |   -  |   -   | 
| 1 | MS-TCN | 68.68<sub>±0.6</sub> |47.71<sub>±0.8</sub> |
| 2 | Transformer Encoder | 60.12<sub>±0.5</sub> |42.7<sub>±1.0</sub> |
| 3 | ASFormer | 69.60<sub>±0.5</sub> |48.34<sub>±0.8</sub> |

## Citation
If you use this code and data, please cite the original papers following:

```
@inproceedings{Renz2021signsegmentation_a,
    author       = "Katrin Renz and Nicolaj C. Stache and Samuel Albanie and G{\"u}l Varol",
    title        = "Sign Language Segmentation with Temporal Convolutional Networks",
    booktitle    = "ICASSP",
    year         = "2021",
}
```
@article{yi2021asformer,
  title={Asformer: Transformer for action segmentation},
  author={Yi, Fangqiu and Wen, Hongyu and Jiang, Tingting},
  journal={arXiv preprint arXiv:2110.08568},
  year={2021}
}
```
## License
The license in this repository only covers the code. For data.zip and models.zip we refer to the terms of conditions of original datasets.


## Acknowledgements
The code builds on the [github.com/RenzKa/sign-segmentation](https://github.com/RenzKa/sign-segmentation) and [github.com/ChinaYi/ASFormer](https://github.com/ChinaYi/ASFormer) repositories. 