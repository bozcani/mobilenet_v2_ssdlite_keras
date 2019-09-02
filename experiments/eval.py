#!/usr/bin/env python

import yaml
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from os import listdir, path
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.preprocessing import image
from losses.keras_ssd_loss import SSDLoss
from utils.ssd_input_encoder import SSDInputEncoder
from utils.object_detection_2d_data_generator import DataGenerator
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from utils.object_detection_2d_geometric_ops import Resize
from utils.object_detection_2d_photometric_ops import ConvertTo3Channels
from utils.average_precision_evaluator import Evaluator as AvgPrecisionEvaluator


class Evaluator:
    def __init__(self, args):
        self.weights_path = args.weights_path
        with open(args.training_config, 'r') as config_file:
            try:
                self.config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.path = args.path
        self.model = self._load_model()

    def _load_model(self):
        model = mobilenet_v2_ssd(self.config, mode='inference')
        print("[*] Loading weights from '{}'".format(self.weights_path))
        model.load_weights(self.weights_path, by_name=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model

    def eval(self):
        classes = ['Background', 'Target 1', 'Target 2', 'Forward gate', 'Backward gate']
        val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
        val_dataset.parse_custom_json(self.path,
                                      ground_truth_available=True)
        evaluator = AvgPrecisionEvaluator(model=self.model,
                                          n_classes=self.config['n_classes'],
                                          data_generator=val_dataset)
        results = evaluator(img_height=self.config['input_res'][0],
                            img_width=self.config['input_res'][1],
                            batch_size=self.config['batch_size'],
                            matching_iou_threshold=0.5, return_precisions=True,
                            return_recalls=True,
                            return_average_precisions=True, verbose=True)
        mean_average_precision, average_precisions, precisions, recalls = results

        for i in range(1, len(average_precisions)):
            print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
            print()
            print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script for MobileNetV2 + SSDLite')
    parser.add_argument('--training-config', type=str,
                        default='training_config.yaml', help='''the path to the
                        YAML configuration file for the training session''')
    parser.add_argument('--weights-path', type=str, help='''the path to the
                        weights file for transfer learning''', required=False)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.eval()
