import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.action_classification as act_class
import scripts.scene_classification as scene_class

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

if __name__ == '__main__':
    # 각종 argument들을 받음.
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--video', type=str, default='testvideo.mp4')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('video read+')
    video = cv2.VideoCapture(args.video)
    ret_val, frame = video.read()


    # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
    video_width_limit = 1000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    logger.info('video frame=%dx%d' % (frame.shape[1], frame.shape[0]))

    pose_graph = act_class.graph
    # count = 0
    while video.isOpened():
        
        logger.debug('+frame processing+')
        ret_val, frame = video.read()
        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        logger.debug('+postprocessing+')
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        output_image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
        
        logger.debug('+classification+')
        # Getting only the skeletal structure (with white background) of the actual image
        skeleton_image = np.zeros(frame.shape,dtype=np.uint8)
        skeleton_image.fill(255) 
        skeleton_image = TfPoseEstimator.draw_humans(skeleton_image, humans, imgcopy=False)
        
        # Classification
        pose_class = act_class.classify(frame, graph=pose_graph)
        # scene_class = scene_class
        #.classify(frame)
        
        logger.debug('+displaying+')
        cv2.putText(output_image,
                    "Current predicted pose is : %s" %(pose_class),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        # cv2.putText(output_image,
		# 		"Predicted Scene: %s" %(scene_class),
		# 		(10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		# 		(0, 0, 255), 2)
        
        cv2.imshow('tf-pose-estimation result', output_image)
        
        fps_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        logger.debug('+finished+')
        
        # For gathering training data 
        # title = 'img'+str(count)+'.jpeg'
        # path = <enter any path you want>
        # cv2.imwrite(os.path.join(path , title), image)
        # count += 1
    
    video.release()

    cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_video.py --video=testvideo.mp4
# =============================================================================
