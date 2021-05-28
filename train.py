import time
import os
import io
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
try:
    from keras.utils.training_utils import multi_gpu_model
except ImportError:
    from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
import keras.backend as K

from adamw import AdamW

from losses import dice_loss, rbox_loss


parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, default=512) # input size for training of the network
parser.add_argument('--batch_size', type=int, default=16) # batch size for training
parser.add_argument('--nb_workers', type=int, default=4) # number of processes to spin up when using process based threading, as defined in https://keras.io/models/model/#fit_generator
parser.add_argument('--n', type=int, default=1)#Epoches the model is already run for
parser.add_argument('--init_learning_rate', type=float, default=0.0001) # initial learning rate
parser.add_argument('--lr_decay_rate', type=float, default=0.94) # decay rate for the learning rate
parser.add_argument('--lr_decay_steps', type=int, default=80) # number of steps after which the learning rate is decayed by decay rate
parser.add_argument('--max_epochs', type=int, default=800) # maximum number of epochs
parser.add_argument('--gpu_list', type=str, default='0') # list of gpus to use
parser.add_argument('--checkpoint_path', type=str, default='tmp/east_resnet_50_rbox') # path to a directory to save model checkpoints during training
parser.add_argument('--save_checkpoint_epochs', type=int, default=5) # period at which checkpoints are saved (defaults to every 10 epochs)
parser.add_argument('--restore_model', type=str, default='')
parser.add_argument('--training_data_path', type=str,nargs='+',required=True) # path to training data
parser.add_argument('--validation_data_path', type=str,required=True) # path to validation data
parser.add_argument('--max_image_large_side', type=int, default=1280) # maximum size of the large side of a training image before cropping a patch for training
parser.add_argument('--model_type', type=str, default='resnet50')
parser.add_argument('--max_text_size', type=int, default=800) # maximum size of a text instance in an image; image resized if this limit is exceeded
parser.add_argument('--min_text_size', type=int, default=10) # minimum size of a text instance; if smaller, then it is ignored during training
parser.add_argument('--min_crop_side_ratio', type=float, default=0.1) # the minimum ratio of min(H, W), the smaller side of the image, when taking a random crop from thee input image
parser.add_argument('--geometry', type=str, default='RBOX') # geometry type to be used; only RBOX is implemented now, but the original paper also uses QUAD
parser.add_argument('--suppress_warnings_and_error_messages', type=bool, default=True) # whether to show error messages and warnings during training (some error messages during training are expected to appear because of the way patches for training are created)
parser.add_argument('--val_loss', type=float, default=0)
parser.add_argument('--previous_val_loss', type=float, default=0)
parser.add_argument('--validation_period', type=int, default=1)
parser.add_argument('--loss_file_name', type=str)
parser.add_argument('--train_val_ratio',type=float, default=0.875)
parser.add_argument('--dataset',type=str, required=True)

FLAGS = parser.parse_args()

gpus = list(range(len(FLAGS.gpu_list.split(','))))



class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, period, save_weights_only):
        super(CustomModelCheckpoint, self).__init__()
        self.period = period
        self.path = path
        # We set the model (non multi gpu) under an other name
        self.model_for_saving = model
        self.epochs_since_last_save = 0
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if ((FLAGS.n+epoch)%self.period)==0:
            self.epochs_since_last_save = 0
            print('previous_val_loss=',FLAGS.previous_val_loss)
            print('val_loss=',FLAGS.val_loss)
            if FLAGS.val_loss< FLAGS.previous_val_loss:
                FLAGS.previous_val_loss=FLAGS.val_loss
                loss_file_path=os.path.join(FLAGS.checkpoint_path,FLAGS.loss_file_name)
                q=open(loss_file_path,'w')
                q.write(str(FLAGS.previous_val_loss))
                q.close()
                print('The val_loss has reduced, so model saved.')
                if self.save_weights_only:
                    self.model_for_saving.save_weights(self.path.format(epoch=epoch + FLAGS.n, **logs), overwrite=True)
                else:
                    self.model_for_saving.save(self.path.format(epoch=epoch + FLAGS.n, **logs), overwrite=True)
            elif ((FLAGS.n+epoch)%FLAGS.save_checkpoint_epochs)==0:
                print('No val_loss reduction, but saving after {} epochs'.format(FLAGS.save_checkpoint_epochs))
                if self.save_weights_only:
                    self.model_for_saving.save_weights(self.path.format(epoch=epoch + FLAGS.n, **logs), overwrite=True)
                else:
                    self.model_for_saving.save(self.path.format(epoch=epoch + FLAGS.n, **logs), overwrite=True)
            else:
                print('THE val_loss has not reduced so model not saved.')

def make_image_summary(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """    
    if len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    else:
        height, width, channel = tensor.shape
        if channel == 1:
            tensor = tensor[:, :, 0]
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir, score_map_loss_weight, small_text_weight, data_generator, write_graph=False):
        self.score_map_loss_weight = score_map_loss_weight
        self.small_text_weight = small_text_weight
        self.data_generator = data_generator
        super(CustomTensorBoard, self).__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):        
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr), 'small_text_weight': K.eval(self.small_text_weight)})
        data = next(self.data_generator)
        pred_score_maps, pred_geo_maps = self.model.predict([data[0][0], data[0][1], data[0][2], data[0][3]])
        img_summaries = []
        for i in range(3):
            input_image_summary = make_image_summary(((data[0][0][i] + 1) * 127.5).astype('uint8'))
            overly_small_text_region_training_mask_summary = make_image_summary((data[0][1][i] * 255).astype('uint8'))
            text_region_boundary_training_mask_summary = make_image_summary((data[0][2][i] * 255).astype('uint8'))
            target_score_map_summary = make_image_summary((data[1][0][i] * 255).astype('uint8'))
            pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))            
            img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
            img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
            img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
            img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
            img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
            for j in range(4):
                target_geo_map_summary = make_image_summary((data[1][1][i, :, :, j] / FLAGS.input_size * 255).astype('uint8'))
                pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / FLAGS.input_size * 255).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
            target_geo_map_summary = make_image_summary(((data[1][1][i, :, :, 4] + 1) * 127.5).astype('uint8'))
            pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
            img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
            img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
        tf_summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(tf_summary, epoch + FLAGS.n)
        super(CustomTensorBoard, self).on_epoch_end(epoch + FLAGS.n, logs)

class SmallTextWeight(Callback):
    def __init__(self, weight):
        self.weight = weight

    # TO BE CHANGED
    def on_epoch_end(self, epoch, logs={}):
        #K.set_value(self.weight, np.minimum(epoch / (0.5 * FLAGS.max_epochs), 1.))
        K.set_value(self.weight, 0)

class ValidationEvaluator(Callback):
    def __init__(self, validation_data, validation_log_dir, period=1):
        super(Callback, self).__init__()

        self.period = period
        self.validation_data = validation_data
        self.validation_log_dir = validation_log_dir
        self.val_writer = tf.summary.FileWriter(self.validation_log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + FLAGS.n) % self.period == 0:
            val_loss, val_score_map_loss, val_geo_map_loss = self.model.evaluate([self.validation_data[0], self.validation_data[1], self.validation_data[2], self.validation_data[3]],
                                                                                 [self.validation_data[3], self.validation_data[4]],
                                                                                 batch_size=FLAGS.batch_size)
            FLAGS.val_loss=val_loss
            print('\nEpoch %d: val_loss: %.4f, val_score_map_loss: %.4f, val_geo_map_loss: %.4f' % (epoch + 1, val_loss, val_score_map_loss, val_geo_map_loss))
            val_loss_summary = tf.Summary()
            val_loss_summary_value = val_loss_summary.value.add()
            val_loss_summary_value.simple_value = val_loss
            val_loss_summary_value.tag = 'loss'
            self.val_writer.add_summary(val_loss_summary, epoch + FLAGS.n)
            val_score_map_loss_summary = tf.Summary()
            val_score_map_loss_summary_value = val_score_map_loss_summary.value.add()
            val_score_map_loss_summary_value.simple_value = val_score_map_loss
            val_score_map_loss_summary_value.tag = 'pred_score_map_loss'
            self.val_writer.add_summary(val_score_map_loss_summary, epoch + FLAGS.n)
            val_geo_map_loss_summary = tf.Summary()
            val_geo_map_loss_summary_value = val_geo_map_loss_summary.value.add()
            val_geo_map_loss_summary_value.simple_value = val_geo_map_loss
            val_geo_map_loss_summary_value.tag = 'pred_geo_map_loss'
            self.val_writer.add_summary(val_geo_map_loss_summary, epoch + FLAGS.n)

            pred_score_maps, pred_geo_maps = self.model.predict([self.validation_data[0][0:3], self.validation_data[1][0:3], self.validation_data[2][0:3], self.validation_data[3][0:3]])
            img_summaries = []
            for i in range(3):
                input_image_summary = make_image_summary(((self.validation_data[0][i] + 1) * 127.5).astype('uint8'))
                overly_small_text_region_training_mask_summary = make_image_summary((self.validation_data[1][i] * 255).astype('uint8'))
                text_region_boundary_training_mask_summary = make_image_summary((self.validation_data[2][i] * 255).astype('uint8'))
                target_score_map_summary = make_image_summary((self.validation_data[3][i] * 255).astype('uint8'))
                pred_score_map_summary = make_image_summary((pred_score_maps[i] * 255).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
                img_summaries.append(tf.Summary.Value(tag='overly_small_text_region_training_mask/%d' % i, image=overly_small_text_region_training_mask_summary))
                img_summaries.append(tf.Summary.Value(tag='text_region_boundary_training_mask/%d' % i, image=text_region_boundary_training_mask_summary))
                img_summaries.append(tf.Summary.Value(tag='score_map_target/%d' % i, image=target_score_map_summary))
                img_summaries.append(tf.Summary.Value(tag='score_map_pred/%d' % i, image=pred_score_map_summary))
                for j in range(4):
                    target_geo_map_summary = make_image_summary((self.validation_data[4][i, :, :, j] / FLAGS.input_size * 255).astype('uint8'))
                    pred_geo_map_summary = make_image_summary((pred_geo_maps[i, :, :, j] / FLAGS.input_size * 255).astype('uint8'))
                    img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (j, i), image=target_geo_map_summary))
                    img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (j, i), image=pred_geo_map_summary))
                target_geo_map_summary = make_image_summary(((self.validation_data[4][i, :, :, 4] + 1) * 127.5).astype('uint8'))
                pred_geo_map_summary = make_image_summary(((pred_geo_maps[i, :, :, 4] + 1) * 127.5).astype('uint8'))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_target/%d' % (4, i), image=target_geo_map_summary))
                img_summaries.append(tf.Summary.Value(tag='geo_map_%d_pred/%d' % (4, i), image=pred_geo_map_summary))
            tf_summary = tf.Summary(value=img_summaries)
            self.val_writer.add_summary(tf_summary, epoch + FLAGS.n)
            self.val_writer.flush()


def lr_decay(epoch):
    #print('epoch in lr_decay:',epoch+FLAGS.n)
    #print('epoch+1',epoch+1)
    #print('Decay Rate:',FLAGS.lr_decay_rate)
    #print('Decay steps:',FLAGS.lr_decay_steps)
    #print('ini',FLAGS.init_learning_rate)
    #print('epoch+1/steps',(epoch+1)// FLAGS.lr_decay_steps)
    #print('epoch+n/steps',(epoch+FLAGS.n)// FLAGS.lr_decay_steps)
    #print('power1',np.power(FLAGS.lr_decay_rate, (epoch+1)// FLAGS.lr_decay_steps))
    #print('power2',np.power(FLAGS.lr_decay_rate, (epoch+FLAGS.n)// FLAGS.lr_decay_steps))
    #print('final1',FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, (epoch+1)// FLAGS.lr_decay_steps))
    #print('final2',FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, (epoch+FLAGS.n)// FLAGS.lr_decay_steps))
    return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, (epoch+FLAGS.n)// FLAGS.lr_decay_steps)



def main(argv=None):
    print("change5")
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    
    # check if checkpoint path exists
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
    
    if FLAGS.dataset=='icdar15':
        import data_processor_icdar15 as data_processor
    elif FLAGS.dataset=='icdar13':
        import data_processor_icdar13 as data_processor
    elif FLAGS.dataset=='mlt':
        import data_processor_mlt as data_processor
    else:
        print('------Choose the correct dataset------------')
    
    print('training_data_path=',FLAGS.training_data_path)
    
    train_data_generator = data_processor.generator(FLAGS)
    train_samples_count = data_processor.count_samples(FLAGS)
    print('-------------------No. of samples in training: ',train_samples_count)

    val_data = data_processor.load_data(FLAGS)

    if len(gpus) <= 1:
        print('Training with 1 GPU')
        if FLAGS.model_type=='resnet':
            print('Importing ResNet50')
            from model import EAST_model
            east = EAST_model(FLAGS.input_size)
        elif FLAGS.model_type=='inceptionv3':
            print('Importing Inception V3')
            from model2 import EAST_model
            east = EAST_model(FLAGS.input_size)
        elif FLAGS.model_type=='densenet':
            print('Importing DenseNet 201')
            from model3 import EAST_model
            east = EAST_model(FLAGS.input_size)
        elif FLAGS.model_type=='vit':
            print('Importing Visual Transformer')
            from ViT import EAST_model
            east = EAST_model(FLAGS.input_size)
        else:
            print('---------Wrong model----------')
        if FLAGS.restore_model is not '':
            print('Loading weigths from {}'.format(FLAGS.restore_model))
            east.model.load_weights(FLAGS.restore_model)
        parallel_model = east.model
    else:
        print('Training with %d GPUs' % len(gpus))
        with tf.device("/cpu:0"):
            east = EAST_model(FLAGS.input_size)
        if FLAGS.restore_model is not '':
            east.model.load_weights(FLAGS.restore_model)
        parallel_model = multi_gpu_model(east.model, gpus=len(gpus))
        
    loss_file_path=os.path.join(FLAGS.checkpoint_path,FLAGS.loss_file_name)
    q=open(loss_file_path,'r')
    FLAGS.previous_val_loss=float(q.readline())
    q.close()
    
    print('previous val loss:',FLAGS.previous_val_loss)
    
    score_map_loss_weight = K.variable(0.01, name='score_map_loss_weight')

    small_text_weight = K.variable(0., name='small_text_weight')

    lr_scheduler = LearningRateScheduler(lr_decay,verbose=1)
    ckpt = CustomModelCheckpoint(model=east.model, path=FLAGS.checkpoint_path + '/model-{epoch:02d}.h5', period=FLAGS.validation_period, save_weights_only=True)
    tb = CustomTensorBoard(log_dir=FLAGS.checkpoint_path + '/train', score_map_loss_weight=score_map_loss_weight, small_text_weight=small_text_weight, data_generator=train_data_generator, write_graph=True)
    small_text_weight_callback = SmallTextWeight(small_text_weight)
    validation_evaluator = ValidationEvaluator(val_data, validation_log_dir=FLAGS.checkpoint_path + '/val',period=FLAGS.validation_period)
    
    callbacks = [lr_scheduler, tb, small_text_weight_callback, validation_evaluator, ckpt]
    #callbacks2 = ModelCheckpoint('/content/drive/My Drive/EAST-Master/checkpoint2/model-{epoch:02d}-{loss:.2f}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min',period=5)

    
    print('Initial learning rate:{}'.format(FLAGS.init_learning_rate))
    opt = AdamW(FLAGS.init_learning_rate)

    parallel_model.compile(loss=[dice_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask, score_map_loss_weight, small_text_weight),
                                 rbox_loss(east.overly_small_text_region_training_mask, east.text_region_boundary_training_mask, small_text_weight, east.target_score_map)],
                           loss_weights=[1., 1.],
                           optimizer=opt)
    east.model.summary()
    
    model_json = east.model.to_json()
    with open(FLAGS.checkpoint_path + '/model.json', 'w') as json_file:
        json_file.write(model_json)
    
#    print('Saving model diagram!')
#    tf.keras.utils.plot_model(
#    east.model,
#    to_file="model1.png",
#    show_shapes=True,
#    show_layer_names=True,
#    rankdir="TB",
#    expand_nested=False,
#    dpi=96,)
    
    history = parallel_model.fit_generator(train_data_generator, epochs=FLAGS.max_epochs, steps_per_epoch=train_samples_count/FLAGS.batch_size, workers=FLAGS.nb_workers, use_multiprocessing=False, max_queue_size=10, callbacks=callbacks, verbose=1)

if __name__ == '__main__':
    main()

