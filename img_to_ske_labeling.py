import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('Pose_Action_and_Scene_Understanding')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
address = os.getcwd()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-human-action-classification')
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % ('mobilenet_thin', get_graph_path('mobilenet_thin')))
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))

    for (root, dirs, files) in os.walk(args.dir):
        print('ske_' + root)
        if not os.path.exists('ske_'+root):
            os.makedirs('ske_'+root)
        for file in files:
            print(root + '\\' + file)
            image = cv2.imread(root + '\\' + file)
            logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

            # count = 0

            logger.debug('+postprocessing+')
            humans = e.inference(image, upsample_size=4.0)
            
            # Getting only the skeletal structure (with white background) of the actual image
            image = np.zeros(image.shape,dtype=np.uint8)
            image.fill(255) 
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            cv2.imwrite('ske_' + root + '\\ske_' + file, image)


# python img_to_ske_labeling.py --dir=training
'''
input:
training (original image)
│
└───normal
│   │   file011.jpg
│   │   file012.jpg
│ 
└───abnormal
    │   file021.jpg
    │   file022.jpg 

output:
ske_training (skeleton image with white background)
│
└───normal
│   │   ske_file011.jpg
│   │   ske_file012.jpg
│ 
└───abnormal
    │   ske_file021.jpg
    │   ske_file022.jpg 


'''