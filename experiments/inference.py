#!/usr/bin/env python

import yaml
import argparse
import numpy as np

from PIL import Image
from imageio import imread
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.preprocessing import image
from losses.keras_ssd_loss import SSDLoss
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd


class Inference:
    def __init__(self, args):
        self.weights_path = args.weights_path
        with open(args.training_config, 'r') as config_file:
            try:
                self.training_config = yaml.safe_load(config_file)
                for key, val in self.training_config.items():
                    if key == 'aspect_ratios_per_layer':
                        self.training_config[key] = [[eval(str(ar)) for ar in layer] for layer in val]
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.model = self._load_model()
        self.orig_images = []
        self.input_images = self._load_images()

    def _load_model(self):
        model = mobilenet_v2_ssd(self.training_config, mode='inference')
        # 2: Load the trained weights into the model.
        model.load_weights(self.weights_path, by_name=True)
        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model

    def _load_images(self):
        input_images = [] # Store resized versions of the images here.
        # We'll only load one image in this example.
        img_path = '/home/theomorales/Code/hybrid-dataset/validation/lights_real_flight/images/000207.png'
        # img_path = '/home/theomorales/Code/hybrid-dataset/training/lights_real_flight/images/007043.png'
        # img_path = '/home/theomorales/Code/hybrid-dataset/validation/lights_real_flight/images/000354.png'
        self.orig_images.append(imread(img_path))
        img = image.load_img(img_path,
                             target_size=(self.training_config['input_res'][0],
                                          self.training_config['input_res'][1]))
        img_array = image.img_to_array(img)
        input_images.append(img_array)

        return np.array(input_images)


    def infer(self):
        y_pred = self.model.predict(self.input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])

        colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
        classes = ['Background', 'Target gate', 'Backward gate', 'Forward gate']

        plt.figure(figsize=(20, 12))
        plt.imshow(self.orig_images[0])

        current_axis = plt.gca()

        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * self.orig_images[0].shape[1] / self.training_config['input_res'][0]
            ymin = box[3] * self.orig_images[0].shape[0] / self.training_config['input_res'][1]
            xmax = box[4] * self.orig_images[0].shape[1] / self.training_config['input_res'][1]
            ymax = box[5] * self.orig_images[0].shape[0] / self.training_config['input_res'][0]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script for MobileNetV2 + SSDLite')
    parser.add_argument('--training-config', type=str,
                        default='training_config.yaml', help='''the path to the
                        YAML configuration file for the training session''')
    parser.add_argument('--weights-path', type=str, help='''the path to the
                        weights file for transfer learning''', required=False)
    args = parser.parse_args()
    inference = Inference(args)
    inference.infer()
