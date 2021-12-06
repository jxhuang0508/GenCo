# GenCo: Generative Co-training for Generative Adversarial Networks with Limited Data

### Updates

## Paper
![](./teaser.png)
[GenCo: Generative Co-training for Generative Adversarial Networks with Limited Data](https://xxx.pdf)  
 [Kaiwen Cui*], [Jiaxing Huang*](https://scholar.google.com/citations?user=czirNcwAAAAJ&hl=en&oi=ao), [Zhipeng Luo], [Gongjie Zhang](https://scholar.google.com/citations?user=sRBTPp4AAAAJ&hl=en), [Fangneng Zhan](https://scholar.google.com/citations?user=8zbcfzAAAAAJ&hl=en&oi=ao),[Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ&hl=en)
 
 *indicates equal contribution.
 School of Computer Science Engineering, Nanyang Technological University, Singapore  
 Thirty-Sixth AAAI Conference on Artificial Intelligence, 2022.
 


## Abstract
Training effective Generative Adversarial Networks (GANs) requires large amounts of training data, without which the trained models are usually sub-optimal with discriminator over-fitting. Several prior studies address this issue by expanding the distribution of the limited training data via massive and hand-crafted data augmentation. We handle data-limited image generation from a very different perspective. Specifically, we design GenCo, a Generative Co-training network that mitigates the discriminator over-fitting issue by introducing multiple complementary discriminators that provide diverse supervision from multiple distinctive views in training. We instantiate the idea of GenCo in two ways. The first way is Weight-Discrepancy Co-training (WeCo) which co-trains multiple distinctive discriminators by diversifying their parameters. The second way is Data-Discrepancy Co-training (DaCo) which achieves co-training by feeding discriminators with different views of the input images (e.g., different frequency components of the input images). Extensive experiments over multiple benchmarks show that GenCo achieves superior generation with limited training data. In addition, GenCo also complements the augmentation approach with consistent and clear performance gains when combined.

## Installation
1. Conda enviroment:
```bash
conda create -n hcl python=3.6
conda activate hcl
conda install -c menpo opencv
pip install torch==1.0.0 torchvision==0.2.1
```

2. Clone the [ADVENT](https://github.com/valeoai/ADVENT):
```bash
git clone https://github.com/valeoai/ADVENT.git
pip install -e ./ADVENT
```

3. Clone the repo:
```bash
https://github.com/jxhuang0508/HCL.git
pip install -e ./HCL
```

4. Install environment:
```bash
conda env create -f hcl_target.yml
```

### Prepare Dataset
* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
HCL/data/GTA5/                               % GTA dataset root
HCL/data/GTA5/images/                        % GTA images
HCL/data/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
HCL/data/Cityscapes/                         % Cityscapes dataset root
HCL/data/Cityscapes/leftImg8bit              % Cityscapes images
HCL/data/Cityscapes/leftImg8bit/val
HCL/data/Cityscapes/gtFine                   % Semantic segmentation labels
HCL/data/Cityscapes/gtFine/val
...
```

### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/jxhuang0508/HCL/releases/tag/model) and put ```GTA5_HCL_source.pth``` into ```HCL/pretrained_models/HCL_source_only_426```, ```GTA5_HCL_target.pth``` into ```HCL/pretrained_models/HCL_target_482```. 

### Training
To train GTA5_HCL_source:
```bash
conda activate hcl
cd HCL/hcl/scripts
python train.py --cfg configs/hcl_source.yml
```

To evaluate trained GTA5_HCL_source:
```bash
conda activate hcl
cd HCL/hcl/scripts
python test.py --cfg configs/hcl_source.yml
```

To train GTA5_HCL_target:
```bash
conda activate hcl_target
cd HCL/hcl_target
python generate_plabel_cityscapes_advent.py  --restore-from ../../pretrained_models/GTA5_HCL_source.pth
```
```bash
conda activate hcl_target
python train_ft_advent_hcl.py --snapshot-dir ./snapshots/HCL_target \
--restore-from ../../pretrained_models/GTA5_HCL_source.pth \
--drop 0.2 --warm-up 5000 --batch-size 9 --learning-rate 1e-4 --crop-size 512,256 --lambda-seg 0.5 --lambda-adv-target1 0 \
--lambda-adv-target2 0 --lambda-me-target 0 --lambda-kl-target 0 --norm-style gn --class-balance --only-hard-label 80 \
--max-value 7 --gpu-ids 0,1,2 --often-balance  --use-se  --input-size 1280,640  --train_bn  --autoaug False --save-pred-every 300
```

To evaluate trained GTA5_HCL_target:
```bash
conda activate hcl_target
cd HCL/hcl_target
./test.sh
```

### Evaluation over Pretrained models

To evaluate GTA5_HCL_source.pth:
```bash
conda activate hcl
cd HCL/hcl/scripts
python test.py --cfg ./configs/hcl_source_pretrained.yml
```

To evaluate GTA5_HCL_target.pth:
```bash
conda activate hcl_target
cd HCL/hcl_target
python evaluate_cityscapes_advent_best.py --restore-from ../../pretrained_models/GTA5_HCL_target.pth
```

 ## Related Works
 We also would like to thank great works as follows:
 - https://github.com/valeoai/ADVENT
 - https://github.com/layumi/Seg-Uncertainty
 - https://github.com/yzou2/CRST


## Contact
If you have any questions, please contact: jiaxing.huang@ntu.edu.sg
