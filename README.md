# Brain Tumor Segmentation in US images

## Installation

- Clone this repository locally.
- Install all the necessary packages by running

    **`pip install -r requirements.txt`**

## Usage

### RAS network

To train a RAS network specify the train dataset path in RAS/train.py folder and run

    **`python3 train.py`**

To test a RAS model run the test dataset path in RAS/test.py folder and run

    **`python3 test.py`**

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
