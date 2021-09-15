import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.action_classification as act_class
import scripts.scene_classification as sce_class

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

    # img to ske img 해주는 TfPoseEstimator model 로딩과 초기화
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    # 비디오의 첫 프레임 읽기
    logger.debug('video read+')
    video = cv2.VideoCapture(args.video)
    ret_val, frame = video.read()

    # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
    video_width_limit = 1000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    logger.info('video frame=%dx%d' % (frame.shape[1], frame.shape[0]))

    action_graph = act_class.graph

    # frame 읽기 실패한 횟수
    fail_count = 0
    while video.isOpened():
        
        logger.debug('+frame processing+')
        ret_val, frame = video.read()
        # 100번 이상 프레임읽기 실패하면 스톱
        if ret_val is False:
            fail_count += 1
            if fail_count > 100:
                break
            continue
        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        logger.debug('+postprocessing+')
        # TfPoseEstimator에서 PreTrain 된 model을 통해 사람의 skeleton point를 찾아냄
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # 찾아낸 skeleton point를 output image에 점과 연결선으로 표시
        output_image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
        
        logger.debug('+classification+')
        # Getting only the skeletal structure (with white background) of the actual image
        # 흰 바탕에 skeleton 이미지만을 그린 이미지 생성
        skeleton_image = np.zeros(frame.shape,dtype=np.uint8)
        skeleton_image.fill(255) 
        skeleton_image = TfPoseEstimator.draw_humans(skeleton_image, humans, imgcopy=False)
        
        # Classification
        # action을 분류하기 위해 skeleton image를 action 분류 모델에 넣음
        action_class = act_class.classify(skeleton_image, graph=action_graph)
        # scene_class = sce_class.classify(frame)
        
        logger.debug('+displaying+')
        # abnormal 
        if action_class[0] == 'a':
            cv2.putText(output_image,
                        "%s" %(action_class),
                        (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        elif action_class[0] == 'n':
            cv2.putText(output_image,
                        "%s" %(action_class),
                        (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
