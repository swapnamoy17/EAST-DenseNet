
# EAST-DenseNet for English, Hindi and Bengali Scene Text Detection

This repository is based on EAST by [kurapan](https://github.com/kurapan/EAST), which is a Keras implementation of EAST based on a Tensorflow implementation made by [argman](https://github.com/argman/EAST).

The original paper by Zhou et al. is available on [arxiv](https://arxiv.org/abs/1704.03155).

+ Only RBOX geometry is implemented
+ Differences from the original paper
    + Uses ResNet-50 instead of PVANet
    + Uses dice loss function instead of balanced binary cross-entropy
    + Uses AdamW optimizer instead of the original Adam

+ Differences from the [karupan](https://github.com/kurapan/EAST) implementation:
    + Uses both ResNet-50 and DenseNet-201
    + Dice Loss with squared terms in union (denominator)
    + Saving models based on validation loss

The implementation of AdamW optimizer is borrowed from [this repository](https://github.com/shaoanlu/AdamW-and-SGDW).

The code should run under both Python 2 and Python 3.

### Requirements

Keras 2.0 or higher, and TensorFlow 1.0 or higher should be enough.

The code should run with Keras 2.1.5. If you use Keras 2.2 or higher, you have to remove `ZeroPadding2D` from the `model.py` file. Specifically, replace the line containing `ZeroPadding2D` with `x = concatenate([x, resnet.get_layer('activation_10').output], axis=3)`.

I used TensorFlow 1.15.2 and Keras 2.3.1.

### Data

You can use your own data, but the annotation files need to conform one of ICDAR 2015, ICDAR 2013 or MLT 2019 format.

+ For English Scene Text:

  ICDAR 2015 dataset can be downloaded from this [site](http://rrc.cvc.uab.es/?ch=4&com=introduction). You need the data from Task 4.1 Text Localization.\
  
  ICDAR 2013 dataset can be downloaded from [site](https://rrc.cvc.uab.es/?ch=2)

  You can also download the [MLT dataset](https://rrc.cvc.uab.es/?ch=15) (Latin set), which uses the  same annotation style as ICDAR 2015, there.

+ For Hindi & Bengal Dataset:
  
  The training dataset of [MLT dataset](https://rrc.cvc.uab.es/?ch=15) conatins segregated 1000 images for each language. 
   + Hindi: 9,001-10,000
   + Bengal: 8,001-9,000
  
  In this project, 700, 100 and 200 images were used for training, validation and evaluation of each language.
  
  The original datasets are distributed by the organizers of the [Robust Reading Competition](http://rrc.cvc.uab.es/) and are licensed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

### Training

You need to put all of your training images and their corresponding annotation files in one directory. The annotation files have to be named `gt_IMAGENAME.txt`(ICDAR15) or 'IMAGENAME.txt'(MLT19).\
You also need a directory for evaluation data, which requires the same structure as the directory with training images.
You can either create a separate directory for validation data or use a section of training for it. 

Training is started by running `train.py`. It accepts the following important arguments:
 + loss_file_name: This file contains the validation loss for the previously saved best model. Mention only the filename, it is in the checkpoint_path. 
 + validation_period: Model is saved after these many epochs depending on whether the validation loss has decresed or not. 
 + checkpoint_epochs: Model is saved regardless of validation loss. 
 + n: Epoch number from which the training is to resumed. (1 + number of epochs completed.) 
 + dataset: icdar13, icdar15 or mlt 
 + model_type: resnet or densenet
 + train_val_ratio: Ratio in which training dataset is to be split for training and validation. 
 
 For spliting training dataset into training and validation, the validation_data_path should be empty (""). 

#### Execution example
```
python train.py --loss_file_name='densenet.txt' --validation_period=1 --save_checkpoint_epochs=10 --gpu_list=0 --input_size=512 --batch_size=4 --nb_workers=1 --training_data_path ../ICDAR2015/train/ --validation_data_path=../ICDAR2015/val/ --checkpoint_path=../checkpoint-1/ --restore_model=../checkpoint-1/model-100.h5

```
Two or more directories can be used as training dataset by setting training path like this:
training_data_path /ICDAR2015/train/ /MLT19/train/

### Test

The images you want to classify have to be in one directory, whose path you have to pass as an argument. Classification is started by running `test.py` with arguments specifying path to the images to be classified, the trained model, and a directory which you want to save the output in.

#### Execution example
```
python test.py --gpu_list=0 --image_path='' --test_data_path=../ICDAR2015/test/ --model_path=../checkpoint-1/model_100.h5 --output_dir=../results/

```
If boxes are to be predicted in only one image, then use image_path=path to image. 

### Evaluation
All of Robust Reading Competitions provide scripts to calculate the evaluation metrics like F1-Score, Precision and Recall. 
A submission zip folder has to created for the same by running 'eval.py'.

#### Execution example
```
python eval.py --dataset='icdar15' --gpu_list=0 --image_path='' --test_data_path=../ICDAR2015/test/ --model_path=../checkpoint-1/model_100.h5 --output_dir=../results/

```
### Results
 + English
 + Hindi
 + Bengali

### Detection examples
![image_1](examples/img_12.jpg)
![image_2](examples/img_13.jpg)
![image_3](examples/img_14.jpg)
![image_4](examples/img_15.jpg)
![image_5](examples/img_28.jpg)
![image_6](examples/img_29.jpg)
![image_7](examples/img_33.jpg)
![image_8](examples/img_37.jpg)
![image_9](examples/img_42.jpg)
