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

To train a RAS network model specify the train dataset path in RAS/train.py folder and run

    **`python3 train.py`**

To test a RAS model set the test dataset path in RAS/test.py folder and run

    **`python3 test.py`**

### CPD Network

To train a CPD model specify the train dataset paths (image_root, gt_root) in CPD/train.py folder and run

    **`python3 train.py`**

To test a CPD model set the dataset_path in CPD/test.py folder and run

    **`python3 test.py`**

### F3Net

To train a F3Net model use the notebook file in F3Net and run the cells in the order they appear to install required packages.

### CRF

To run CRF run the notebook file  CRF/CRF.ipynb

## Acknowledgements

The code for each model is taken from the code repositories of the authors that proposed each model respectively.

[RAS](https://github.com/ShuhanChen/RAS-pytorch)[1]
[CPD](https://github.com/wuzhe71/CPD)[2]
[F3Net](https://github.com/weijun88/F3Net)[3].

We thank the authors for sharing their code.

## References

> [1] Chen S, Tan X, Wang B, Lu H, Hu X, Fu Y. Reverse Attention-Based ResidualNetwork for Salient Object Detection. IEEE Transactions on Image Processing. 2020;29:3763–3776.<br>
> [2] Wu Z, Su L, Huang Q. Cascaded partial decoder for fast and accurate salientobject detection. In:Proceedings of the IEEE Conference on Computer Vision andPattern Recognition; 2019. p. 3907–3916.<br>
> [3] Wei J, Wang S, Huang Q. F3Net: Fusion, feedback and focus for salient objectdetection. arXiv preprint arXiv:191111445. 2019;.<br>
