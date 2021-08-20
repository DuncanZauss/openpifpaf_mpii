# openpifpaf_mpii
OpenPifPaf plugin to train and evaluate on the (MPII pose dataset)[http://human-pose.mpi-inf.mpg.de/].
Example prediction for an image of the test set:
![Image 035647817.jpg with superimposed predictions](/docs/035647817_pred.png)
## Installation
To install the openpifpaf_mpii plugin you will need to run the following command:
```sh
git clone https://github.com/DuncanZauss/openpifpaf_mpii.git
cd openpifpaf_mpii
pip install -e .
```
If OpenPifPaf is not already installed in your environment, it will be installed as well.

## Data preparation
For training the (MPII images)[http://human-pose.mpi-inf.mpg.de/] and the processed annotations will need to be downloaded. The annotations were transformed using [this toll](https://github.com/mks0601/TF-SimpleHumanPose/blob/master/tool/mpii2coco.py). You can download the images and the processed annotations with the following commands:
```sh
mkdir MPII
cd MPII
wget https://github.com/DuncanZauss/openpifpaf_mpii/releases/download/v.0.1.0-alpha/annotations.zip
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
unzip annotations.zip
tar -xzf mpii_human_pose_v1.tar.gz
rm annotations.zip mpii_human_pose_v1.tar.gz
```
The resulting folder structure should be like this:
MPII
   ├── annotations
      ├── MPII_coco_style_anns_test.json
      ├── MPII_coco_style_anns_train.json
   ├── images
      ├── 000001163.jpg
      ├── 000003072.jpg
      .....
Finally softlink the MPII folder with `ln -s` to the folder from where you will run the training and evaluation command.
## Training
To train an openpifpaf model with the MPII dataset you can run the following command:
```sh
CUDA_VISIBLE_DEVICES=0 python -m openpifpaf.train --lr=0.0001 --momentum=0.95 --b-scale=10.0 --clip-grad-value=10 --epochs=350 --lr-decay 320 340 --lr-decay-epochs=10 --batch-size=16 --weight-decay=1e-5 --checkpoint shufflenetv2k16 --dataset mpii --mpii-upsample=2 --mpii-extended-scale --mpii-orientation-invariant=0.1 --head-consolidation=create --lr-warm-up-start-epoch=250
```
To decrease the trainning time, the model in the command above is trained starting from a model that was pretrained on MS COCO. For further training options refer to the (OpenPifpaf Guide)[https://openpifpaf.github.io/train.html#shufflenet].
## Evaluation
### TBD
An overview of the MPII evaluation protocol and the matlab code can be found (here)[http://human-pose.mpi-inf.mpg.de/#evaluation].
