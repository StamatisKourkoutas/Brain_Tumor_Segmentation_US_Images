# Brain Tumor Segmentation in US images

## Installation

- Clone this repository locally.
- It is better to use a python virtual environment to install all the necessary packages.
- To avoid any problems, update pip by running

    **`pip install --upgrade pip`**

- Install all the necessary packages by running

    **`pip install -r requirements.txt`**

## Usage

### RAS Network

- To train a RAS network model specify the train dataset path in RAS/train.py folder and run

    **`python3 train.py`**

- To test a RAS model specify the test dataset path in RAS/test.py folder and run

    **`python3 test.py`**

### CPD Network

- To train a CPD model specify the train dataset paths (image_root, gt_root) in CPD/train.py folder and run

    **`python3 train.py`**

- To test a CPD model specify the dataset_path in CPD/test.py folder and run

    **`python3 test.py`**

### F3Net

- To train and test a F3Net model use the notebook file in F3Net and run the cells in the order they appear. The first cells are responsible for installing during runtime the apex library.

- This architecture is presented in a notebook format as oposed to the rest, because it requires the apex library, the installation of which was failing in our own system. When the library is installed in a google colab environment via the notebook code we provide, the installation succeeds. Since this may happen in other systems as well we provide this notebook version for ease which can be run straight in a google colab environment. However, if you wish to train this architecture in your own system install the apex library, by running

    **`git clone https://github.com/NVIDIA/apex`**

    **`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex`**

### CRF

- To run the CRF post-processing method, specify the imgs_path of a folder with US images and the masks_path of a folder with the segmentations produced by a deep CNN in CRF/crf.py and run

    **`python3 crf.py`**

### Visualisation

- To run the Guided Back-propagation visualization method for the CPD network navigate to ./visualisation/guided_backpropagation and run

    **`python3 guided_backprop.py`**

- To run the Smooth Grad visualization method for the CPD network navigate to ./visualisation/smooth_grad and run

    **`python3 smooth_grad.py`**

### Evaluation

- To evaluate the saliency maps produced by each model as well as the CRF run the main.m MATLAB file in ./'saliency evaluation'

### Usefull Scripts

- The folder usefull scripts contains some scripts that were used for the processing of images and their masks.

## Acknowledgements

- The code for each model is taken from the code repositories of the authors that proposed each model respectively. [RAS](https://github.com/ShuhanChen/RAS-pytorch)[1] [CPD](https://github.com/wuzhe71/CPD)[2] [F3Net](https://github.com/weijun88/F3Net)[3].
- The code for the Guided Back-propagation and the Smoothgrad visualisation methods is taken from [here](https://github.com/utkuozbulak/pytorch-cnn-visualizations) and is altered as required.
- The code for the evaluation is taken from [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox) and is altered as required.
- The main library for the fully connected CRF that we used can be found [here](https://github.com/lucasb-eyer/pydensecrf). The code for the CRF inference loop that we used is taken from [here](https://github.com/dhawan98/Post-Processing-of-Image-Segmentation-using-CRF) and is altered as required.

We thank the authors for sharing their code.

## References

> [1] Chen S, Tan X, Wang B, Lu H, Hu X, Fu Y. Reverse Attention-Based ResidualNetwork for Salient Object Detection. IEEE Transactions on Image Processing. 2020;29:3763–3776.<br>
> [2] Wu Z, Su L, Huang Q. Cascaded partial decoder for fast and accurate salientobject detection. In: Proceedings of the IEEE Conference on Computer Vision andPattern Recognition; 2019. p. 3907–3916.<br>
> [3] Wei J, Wang S, Huang Q. F3Net: Fusion, feedback and focus for salient objectdetection; Arxiv. \[Preprint\] 2019.  Available from: ```https://arxiv.org/abs/1911.11445.```<br>
