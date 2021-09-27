import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.action_classification as act_class
import scripts.scene_classification as sce_class

logger = logging.getLogger('Pose_Action_and_Scene_Understanding')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
address = os.getcwd()

action_graph = act_class.graph

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

	# count = 0
	
	logger.debug('+image processing+')
	logger.debug('+postprocessing+')
	start_time = time.time()
	# 원본 이미지에 있는 사람 정보를 받는다.
	humans = e.inference(image, upsample_size=4.0)
	# 사람 정보로 이미지 박스를 만든다.
	boundarys = TfPoseEstimator.get_humans_imgbox(image, humans, 0.1)
	# 최종 출력할 이미지를 복사한다.
	img = image
	# print(boundarys)
	# 이미지 박스별로 실행
	for boundary in boundarys:
		# print(boundarys[boundary])
		img_left = boundarys[boundary][0]
		img_right = boundarys[boundary][1]
		img_up = boundarys[boundary][2]
		img_down = boundarys[boundary][3]

		# 해당 이미지 박스의 사람 skeleton을 그린다. (출력용)
		# sub_img = TfPoseEstimator.draw_humans(image, [humans[boundary]], imgcopy=True)
		# 흰 바탕의 skeleton이미지만 남긴 것을 그린다. (판정용)
		sub_img_ske = np.zeros(image.shape,dtype=np.uint8)
		sub_img_ske.fill(255) 
		sub_img_ske = TfPoseEstimator.draw_humans(sub_img_ske, [humans[boundary]], imgcopy=False)
		# 흰 바탕의 skeleton이미지로 어떤 액션인지(normal / abnormal) 판정한다.
		action_class = act_class.classify(sub_img_ske[img_up:img_down, img_left:img_right], graph=action_graph)

		# 판정한 것을 박스별로 출력한다.
		if action_class[0] == 'a':
			cv2.putText(img,
                        "%s" %(action_class),
                        (img_left, img_up + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
			cv2.rectangle(img, (img_left, img_up), (img_right, img_down), (0, 0, 255))
		# normal
		elif action_class[0] == 'n':
			cv2.putText(img,
                        "%s" %(action_class),
                        (img_left, img_up + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
			cv2.rectangle(img, (img_left, img_up), (img_right, img_down), (0, 255, 0))
		# cv2.imshow("test", sub_img)
		# cv2.waitKey(0)
		

		# cv2.imshow("test", image[img_up:img_down, img_left:img_right].copy())
		# cv2.waitKey(0)

		# 스켈레톤 이미지를 출력이미지에 그린다.
		img = TfPoseEstimator.draw_humans(image, [humans[boundary]], imgcopy=False)

	# img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
	# cv2.imshow('tf-human-action-classification result', img)
	# cv2.waitKey(0)
	# logger.debug('+classification+')
	# Getting only the skeletal structure (with white background) of the actual image
	# image = np.zeros(image.shape,dtype=np.uint8)
	# image.fill(255) 
	# image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
	
	# Classification
	# action_class = act_class.classify(image, action_graph)
	# scene_class = sce_class.classify(args.image)
	end_time = time.time()
	# logger.debug('+displaying+')
	# cv2.putText(img,
	# 			"Predicted Pose: %s" %(action_class),
	# 			(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	# 			(0, 0, 255), 2)
	# cv2.putText(img,
	# 			"Predicted Scene: %s" %(scene_class),
	# 			(10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	# 			(0, 0, 255), 2)
	print('\n Overall Evaluation time (1-image): {:.3f}s\n'.format(end_time-start_time))
	cv2.imwrite('show1.png',img)
	cv2.imshow('tf-human-action-classification result', img)
	cv2.waitKey(0)
	logger.debug('+finished+')
	cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_image.py --image=test.png
# =============================================================================
