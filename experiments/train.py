#!/usr/bin/env python

import os
import re
import yaml
import argparse

from keras import backend as K
from keras.optimizers import Adam
from losses.keras_ssd_loss import SSDLoss
from utils.coco import get_coco_category_maps
from utils.ssd_input_encoder import SSDInputEncoder
from utils.object_detection_2d_geometric_ops import Resize
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from utils.object_detection_2d_data_generator import DataGenerator
from utils.object_detection_2d_photometric_ops import ConvertTo3Channels
from utils.data_augmentation_chain_original_ssd import SSDDataAugmentation
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler


class Trainer:
    def __init__(self, args):
        self.train_dir = args.train_dir
        self.val_dir = args.val_dir
        self.train_annotations = args.train_annotations
        self.val_annotations = args.val_annotations
        self.log_dir = args.log_dir
        self.weights_path = args.weights_path
        with open(args.training_config, 'r') as config_file:
            try:
                self.training_config = yaml.safe_load(config_file)
                for key, val in self.training_config.items():
                    if key == 'aspect_ratios_per_layer':
                        self.training_config[key] = [[eval(str(ar)) for ar in layer] for layer in val]
            except yaml.YAMLError as exc:
                raise Exception(exc)

    # learning rate schedule
    def lr_schedule(self, epoch):
        if epoch < 200:
            return 0.001
        elif epoch < 500:
            return 0.0001
        else:
            return 0.00001


    # set trainable layers
    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                set_trainable(
                    layer_regex, keras_model=layer)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                print("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))


    def train(self):
        # build model
        model = mobilenet_v2_ssd(self.training_config)

        # load weights
        model.load_weights(self.weights_path, by_name=True)

        # compile the model
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        # set_trainable(r"(ssd\_[cls|box].*)", model)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        print(model.summary())

        # load data
        train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

        train_dataset.parse_json(images_dirs=[self.train_dir], annotations_filenames=[self.train_annotations],
                                 ground_truth_available=True, include_classes='all', ret=False)
        val_dataset.parse_json(images_dirs=[self.val_dir], annotations_filenames=[self.val_annotations],
                               ground_truth_available=True, include_classes='all', ret=False)

        # We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
        cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(train_annotations_filename)

        # set the image transformations for pre-processing and data augmentation options.
        # For the training generator:
        ssd_data_augmentation = SSDDataAugmentation(img_height=image_size[0],
                                                    img_width=image_size[1],
                                                    background=subtract_mean)

        # For the validation generator:
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=image_size[0], width=image_size[1])

        # instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

        # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
        predictor_sizes = [model.get_layer('ssd_cls1conv2_bn').output_shape[1:3],
                           model.get_layer('ssd_cls2conv2_bn').output_shape[1:3],
                           model.get_layer('ssd_cls3conv2_bn').output_shape[1:3],
                           model.get_layer('ssd_cls4conv2_bn').output_shape[1:3],
                           model.get_layer('ssd_cls5conv2_bn').output_shape[1:3],
                           model.get_layer('ssd_cls6conv2_bn').output_shape[1:3]]

        ssd_input_encoder = SSDInputEncoder(img_height=image_size[0],
                                            img_width=image_size[1],
                                            n_classes=n_classes,
                                            predictor_sizes=predictor_sizes,
                                            scales=scales,
                                            aspect_ratios_per_layer=aspect_ratios_per_layer,
                                            two_boxes_for_ar1=two_boxes_for_ar1,
                                            steps=steps,
                                            offsets=offsets,
                                            clip_boxes=clip_boxes,
                                            variances=variances,
                                            matching_type='multi',
                                            pos_iou_threshold=0.5,
                                            neg_iou_limit=0.3,
                                            normalize_coords=normalize_coords)

        # create the generator handles that will be passed to Keras' `fit_generator()` function.

        train_generator = train_dataset.generate(batch_size=batch_size,
                                                 shuffle=True,
                                                 transformations=[ssd_data_augmentation],
                                                 label_encoder=ssd_input_encoder,
                                                 returns={'processed_images',
                                                          'encoded_labels'},
                                                 keep_images_without_gt=False)

        val_generator = val_dataset.generate(batch_size=batch_size,
                                             shuffle=False,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

        # Get the number of samples in the training and validations datasets.
        train_dataset_size = train_dataset.get_dataset_size()
        val_dataset_size = val_dataset.get_dataset_size()

        print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

        callbacks = [LearningRateScheduler(schedule=lr_schedule, verbose=1),
                     TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
                     ModelCheckpoint(
                         os.path.join(log_dir, "ssdseg_coco_{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                         monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)]

        model.fit_generator(train_generator, epochs=1000, steps_per_epoch=1000,
                            callbacks=callbacks, validation_data=val_generator,
                            validation_steps=100, initial_epoch=0)




if __name__ == "__main__":
    K.clear_session()
    parser = argparse.ArgumentParser(
        description='Training script for MobileNetV2 + SSDLite')
    parser.add_argument('train_dir', type=str, help='''the path to the directory
                        of training images''')
    parser.add_argument('train_annotations', type=str, help='''the path to the
                        annotations file of training images''')
    parser.add_argument('val_dir', type=str, help='''the path to the directory
                        of validation images''')
    parser.add_argument('val_annotations', type=str, help='''the path to the
                        annotations file of validation images''')
    parser.add_argument('--log-dir', type=str, default='logs', help='''the path
                        to the log directory (will be created if it does not
                        exist)''')
    parser.add_argument('--training-config', type=str,
                        default='training_config.yaml', help='''the path to the
                        YAML configuration file for the training session''')
    parser.add_argument('--weights-path', type=str, help='''the path to the
                        weights file for transfer learning''', required=True)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()


