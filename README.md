# Modeling Attention and Binding in the Brain through Bidirectional Recurrent Gating

This is the code code repository for the paper [Modeling Attention and Binding in the Brain through Bidirectional Recurrent Gating](https://doi.org/10.1101/2024.09.09.612033)

### Demo
The `demo_coco.py` uses the pre-trained COCO model and the images in the demo folder to demonstrate the multi-task paradigm and it does not require installing COCO-API. It is possible to include new images in the demo directory, but their names must have suffix "_s" for single or "_g" for grid,.

### Pre-trained models
Pre-trained models with the parameters used to train them are in the pretrained directory. For each experiment, the results could be reproduced by running the corresponding notebook.

### COCO experiment
For the COCO experiment, the COCO python tools (pycocotools) should be installed inside the src folder (see [COCO-API](https://github.com/cocodataset/cocoapi)).

## Datasets
All the datasets are publicly available in their corresponding repositories:  
**MNIST**: [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)  
**COCO**: [https://cocodataset.org](https://cocodataset.org)  
**BG-20k**: [https://github.com/JizhiziLi/GFM](https://github.com/JizhiziLi/GFM)  
**CelebA**: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
**FashionMNIST**: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)  

The codes for composing different stimuli and the curve images is included in the [./src/composer.py](./src/composer.py)
