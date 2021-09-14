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
	parser.add_argument('--image', type=str, required=True)
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	args = parser.parse_args()

	logger.debug('initialization %s : %s' % ('mobilenet_thin', get_graph_path('mobilenet_thin')))
	e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
	image = cv2.imread(args.image)
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

	humans = e.inference(image, upsample_size=4.0)

	# Getting only the skeletal structure (with white background) of the actual image
	ske_image = np.zeros(image.shape,dtype=np.uint8)
	ske_image.fill(255) 
	ske_image = TfPoseEstimator.draw_humans(ske_image, humans, imgcopy=False)
	logger.debug('+displaying+')
	cv2.imwrite('ske_' + args.image, ske_image)
	logger.debug('+finished+')
	cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python estimatorLabeltools.py --image=1.jpg
# =============================================================================
