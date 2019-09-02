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
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd


class Inference:
    def __init__(self, args):
        self.weights_path = args.weights_path
        with open(args.training_config, 'r') as config_file:
            try:
                self.training_config = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        self.path = args.path
        self.model = self._load_model()
        self.input_images = [image.img_to_array(
            image.load_img(
                path.join(self.path, f)).resize(
                    (self.training_config['input_res'][1],
                     self.training_config['input_res'][0])))
            for f in sorted(listdir(self.path))
            if path.isfile(path.join(self.path, f))]

    def _load_model(self):
        model = mobilenet_v2_ssd(self.training_config, mode='inference_fast')
        with open('model.yaml', 'w') as f:
            yamodel = model.to_yaml()
            f.write(yamodel)
            print("Model saved!")
        model.load_weights(self.weights_path, by_name=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        return model

    def infer(self):
        n = 0
        step = 100
        colors = ['black', 'red', 'blue', 'green']
        classes = ['Background', 'Target gate', 'Backward gate', 'Forward gate']
        # create the ImageFont instance
        font_file_path = 'Hack-Regular.ttf'
        font = ImageFont.truetype(font_file_path, size=10, encoding="unic")

        with tqdm(total=len(self.input_images)) as pbar:
            for i in range(0, len(self.input_images), step):
                input_images = self.input_images[i:i+step]
                y_pred = self.model.predict(np.array(input_images))
                confidence_threshold = 0.5
                y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

                for img, predictions in zip(input_images, y_pred_thresh):
                    pilImg = image.array_to_img(img)
                    draw = ImageDraw.Draw(pilImg)
                    for box in predictions:
                        xmin, ymin, xmax, ymax = box[2], box[3], box[4], box[5]
                        if int(box[0]) is not 0:
                            color = colors[int(box[0])]
                            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
                            textSize = draw.textsize(label)
                            draw.rectangle((
                                (xmin-2, ymin-2),
                                (xmin+textSize[0]+2, ymin+textSize[1])),
                                fill=color)
                            draw.text((xmin, ymin), label, fill='white',
                                      font=font)

                    pilImg.save("inference/%06d.png" % n)
                    pilImg.close()
                    n += 1
                    pbar.update(1)


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
    inference = Inference(args)
    inference.infer()
