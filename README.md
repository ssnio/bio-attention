# Modeling Attention and Binding in the Brain through Bidirectional Recurrent Gating

This is the code code repository for the paper [Modeling Attention and Binding in the Brain through Bidirectional Recurrent Gating](https://doi.org/10.1101/2024.09.09.612033)

![Demo](https://raw.githubusercontent.com/ssnio/bio-attention/refs/heads/main/demo/demo.gif)

## repository structure
`prelude.py`: Auxiliary functions for setting up the experiments and book-keeping.  
`demo_coco.py`: uses the pre-trained COCO model and the images in the demo folder to demonstrate the multi-task paradigm and it does not require installing COCO-API. It is possible to include new images in the demo directory, but their names must have suffix "_s" for single or "_g" for grid.  
`main_mnist.py`, `main_coco.py`, and `main_curve_tracing.py` are the code to perform the respecive experiment, including the training. To use the pre-trained models for testing, visualization, and analysis, please use notebooks.  

[data](.data/): It currently only contains the arrows datasets. The user must download other datasets from their corresponding repositories (see [Datasets](#datasets))  
[demo](.demo/): Contains few sample images from the MS-COCO dataset used in the demo.  
[notebooks](.notebooks/): Notebooks for each experiment to run pretrained models for test and further analysis.  
[pretrained](.pretrained/): Pre-trained models with the parameters used to train them are in the pretrained directory. For each experiment, the results could be reproduced by running the corresponding notebook.  
[src](.src/): Contains the core codes:
- `composer.py`: code used to compose new datasets and stimuli for all the experiments
- `conductor.py`: training and evaluation code.
- `curves_utils.py`: utility functions used for the analysis of curve-tracing and attention-invariant tuning experiments.
- `model.py`: the attention network model.
- `utils.py`: general utility functions, including the visualization code.
- For the COCO experiment, the COCO python tools (pycocotools) should be installed inside the src folder (see [COCO-API](https://github.com/cocodataset/cocoapi)).

## Datasets
All the datasets are publicly available in their corresponding repositories:  
**MNIST**: [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)  
**COCO**: [https://cocodataset.org](https://cocodataset.org)  
**BG-20k**: [https://github.com/JizhiziLi/GFM](https://github.com/JizhiziLi/GFM)  
**CelebA**: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
**FashionMNIST**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  

The codes for composing different input-images as well as the curve-tracing stimuli is included in the [./src/composer.py](./src/composer.py)

## Required packages
- torch (PyTorch)
- numpy
- Pillow
- torchvision
- matplotlib
