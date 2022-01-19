# MVA RecVis course final project : 

## Goal of the project : Detecting Temporal Boundaries in Sign Language videos. 

Sign language automatic indexing is an important challenge to develop better communication tools for the deaf community. However, annotated datasets for sign langage are limited, and people with skills to anotate data are rare, which makes it hard to train performant machine learning models. An important challenge is therefore to : 

*  Increase available training datasets. 
*  Make labeling easier for professionnals to reduce risks of bad annotations. 

In this context, techniques have emerged to perform automatic sign segmentation in videos, by marking the boundaries between individual signs in sign language videos. The developpment of such tools offers the potential to alleviate the limited supply of labelled dataset currently available for sign research. 

[[Project page]](https://www.robots.ox.ac.uk/~vgg/research/signsegmentation/)

![demo](demo/results/demo.gif)

## Contents
* [Setup](#setup)
* [Previous works and personal contribution](#previous-works-personal-contribution)
* [Data and models](#data-and-models)
* [Demo](#demo)
* [Training](#training)
  * [Train ICASSP](#train-icassp)
  * [Train CVPRW](#train-cvprw)
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

## Previous works and personal contribution : 

This repository provides code for the RecVis course Final project. In this project, we reproduced the results obtained on the following paper (by using the code from [here](https://github.com/RenzKa/sign-segmentation)) :  

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)

We used the pre-extracted frame-level features obtained by applying the I3D model on videos to retrain the MS-TCN architecture for frame binary classification and reproduce the papers results. The test folder proposes a notebook for reproducing the original paper results, with a $meanF1B = 68.68$.





## Data and models
You can download our pretrained models (`models.zip [302MB]`) and data (`data.zip [5.5GB]`) used in the experiments [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing) or by executing `download/download_*.sh`. The unzipped `data/` and `models/` folders should be located on the root directory of the repository (for using the demo downloading the `models` folder is sufficient).


### Data:
Please cite the original datasets when using the data: [BSL Corpus](https://bslcorpusproject.org/cava/acknowledgements-and-citation/) | [Phoenix14](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
We provide the pre-extracted features and metadata. See [here](data/README.md) for a detailed description of the data files. 
- Features: `data/features/*/*/features.mat`
- Metadata: `data/info/*/info.pkl`

### Models:
- I3D weights, trained for sign classification: `models/i3d/*.pth.tar`
- MS-TCN weights for the demo (see tables below for links to the other models): `models/ms-tcn/*.model`

The folder structure should be as below:
```
sign-segmentation/models/
  i3d/
    i3d_kinetics_bsl1k_bslcp.pth.tar
    i3d_kinetics_bslcp.pth.tar
    i3d_kinetics_phoenix_1297.pth.tar
  ms-tcn/
    mstcn_bslcp_i3d_bslcp.model
```
## Demo
The demo folder contains a sample script to estimate the segments of a given sign language video. It is also possible to use pre-extracted I3D features as a starting point, and only apply the MS-TCN model.
`--generate_vtt` generates a `.vtt` file which can be used with [our modified version of VIA annotation tool](https://github.com/RenzKa/VIA_sign-language-annotation):
```
usage: demo.py [-h] [--starting_point {video,feature}]
               [--i3d_checkpoint_path I3D_CHECKPOINT_PATH]
               [--mstcn_checkpoint_path MSTCN_CHECKPOINT_PATH]
               [--video_path VIDEO_PATH] [--feature_path FEATURE_PATH]
               [--save_path SAVE_PATH] [--num_in_frames NUM_IN_FRAMES]
               [--stride STRIDE] [--batch_size BATCH_SIZE] [--fps FPS]
               [--num_classes NUM_CLASSES] [--slowdown_factor SLOWDOWN_FACTOR]
               [--save_features] [--save_segments] [--viz] [--generate_vtt]
```

Example usage:
``` bash
# Print arguments
python demo/demo.py -h
# Save features and predictions and create visualization of results in full speed
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 1 --save_features --save_segments --viz
# Save only predictions and create visualization of results slowed down by factor 6
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --save_segments --viz
# Create visualization of results slowed down by factor 6 and .vtt file for VIA tool
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --viz --generate_vtt
```

The demo will: 
1. use the `models/i3d/i3d_kinetics_bslcp.pth.tar` pretrained I3D model to extract features,
2. use the `models/ms-tcn/mstcn_bslcp_i3d_bslcp.model` pretrained MS-TCN model to predict the segments out of the features,
3. save results (depending on which flags are used).

## Training
### Train ICASSP
Run the corresponding run-file (`*.sh`) to train the MS-TCN with pre-extracted features on BSL Corpus.
During the training a `.log` file for tensorboard is generated. In addition the metrics get saved in `train_progress.txt`.

* Influence of I3D training (fully-supervised segmentation results on BSL Corpus)

|ID | Model | mF1B | mF1S | Links (for seed=0) |
|   -   |   -  |   -  |   -   |  -   | 
| 1 | BSL Corpus | 68.68<sub>±0.6</sub> |47.71<sub>±0.8</sub> | [run](https://drive.google.com/file/d/1na-b_WoPPajPN9WCd8kdQ0WrMKr-EBzH/view?usp=sharing), [args](https://drive.google.com/file/d/1FHC0mHt3meXBobnuPWN17vwlPaekF-qU/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1ot6VNYfzn9UlVdRt31mQ8Mfics1DxP7T/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1y-LeaNuZSAeLTc1yKXA0XUWnwINr8pVo?usp=sharing) |
| 2 | BSL1K -> BSL Corpus | 66.17<sub>±0.5</sub> |44.44<sub>±1.0</sub> | [run](https://drive.google.com/file/d/1hvRN7a3GX7YQF9jTxfmsJbS_WXebsb9Y/view?usp=sharing), [args](https://drive.google.com/file/d/1Gg_qZYYUtl3YtNQec2ku-VxBiM9r-0mR/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1ETNO6tLgmg_o-T7L0qG4eMEd8QhsZpOG/view?usp=sharing), [logs](https://drive.google.com/drive/folders/11WnAEmYY3PIC03XdBNZB0lU4BkZmVqKp?usp=sharing) |


* Fully-supervised segmentation results on PHOENIX14

|ID | I3D training data | MS-TCN training data | mF1B | mF1S | Links (for seed=0) |
| - |   -   |   -  |   -  |   -   |   -   | 
|3| BSL Corpus | PHOENIX14 | 65.06<sub>±0.5</sub> |44.42<sub>±2.0</sub> | [run](https://drive.google.com/file/d/1Vihh4MG0iWOLQalI5SqVYjQn3aELsLRP/view?usp=sharing), [args](https://drive.google.com/file/d/1PLN7wcsJBqnhIBWkdfgPvyujh3MIzea4/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1hynccDWvwKaH8uiMRAVYiWsqeK9dptm5/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1Rfklvh3-pdCe_meKOcw9rjLCR_R5j-ap?usp=sharing) |
|4| PHOENIX14 | PHOENIX14 | 71.50<sub>±0.2</sub> |52.78<sub>±1.6</sub> | [run](https://drive.google.com/file/d/1jAfJPs58ErT-UTnN3mPstOhekAgTCLEY/view?usp=sharing), [args](https://drive.google.com/file/d/1ak6VOuLvv6hrDEUsJbI1WcGoJGfVpUi5/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1q0bShF9IpuuSHrJyZIPNMUsC5guQ0B8m/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1wk5dly6jxKEivO3q5BEahTtLSsrPK1of?usp=sharing) |



## Citation
If you use this code and data, please cite the following:

```
@inproceedings{Renz2021signsegmentation_a,
    author       = "Katrin Renz and Nicolaj C. Stache and Samuel Albanie and G{\"u}l Varol",
    title        = "Sign Language Segmentation with Temporal Convolutional Networks",
    booktitle    = "ICASSP",
    year         = "2021",
}
```
```
@inproceedings{Renz2021signsegmentation_b,
    author       = "Katrin Renz and Nicolaj C. Stache and Neil Fox and G{\"u}l Varol and Samuel Albanie",
    title        = "Sign Segmentation with Changepoint-Modulated Pseudo-Labelling",
    booktitle    = "CVPRW",
    year         = "2021",
}
```

## License
The license in this repository only covers the code. For data.zip and models.zip we refer to the terms of conditions of original datasets.


## Acknowledgements
The code builds on the [github.com/yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) repository. The demo reuses parts from [github.com/gulvarol/bsl1k](https://github.com/gulvarol/bsl1k).  We like to thank C. Camgoz for the help with the BSLCORPUS data preparation.