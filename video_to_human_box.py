'''
video_to_human_box.py : 동영상 파일에서 사람의 스켈레톤 이미지를 추출하여 동작을 판단하는 python 파일
대표적인 argument
--video : 동영상 파일 (확장자까지)
'''
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

    # 이미지에서 스켈레톤 이미지를 뽑아주는 TfPoseEstimator model 로딩과 초기화
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

    # video width 가 너무 크면 속도가 느려져서 일정 제한 이상에서 width와 height를 절반으로 downscaling함
    video_width_limit = 4000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    logger.info('video frame=%dx%d' % (frame.shape[1], frame.shape[0]))

    # 모델(의 그래프)을 불러옴
    action_graph = act_class.graph

    '''
    args.video에서 입력된 (경로)동영상 이름.확장자에서
    동영상 이름만 빼 와서
    dataset\\동영상 이름\\ske
    dataset\\동영상 이름\\img
    이렇게 두 가지 폴더 경로를 만듦
    사실 코딩 개판으로 함. 다시 하라면 더 깔끔하게 할 것임.
    '''
    # 경로를 받음
    path = args.video
    # 확장자 자르는 용도로 .단위로 스플릿함 (폴더 이름에 .넣는 경우도 있는데 파일 이름에 확장자표시 말고 추가로 없으면 상관없음)
    # 해당 결과 path는 [(경로들)+동영상이름, 확장자]가 됨. ((경로들)이 절대경로(c:, d:어쩌구)라도 상관없음)
    path = path.split('.')

    # 잘린거에서 백슬래시(\)를 없애기 위해 그걸로 스플릿하여 temp에 넣음
    # 해당 결과 temp는 (경로들)이 \로 이루어졌다면 [폴더명, 폴더명, .. , 동영상이름, 확장자]가 됨
    # (경로들)이 /로 이루어졌다면 [폴더명/폴더명/.../동영상이름, 확장자]가 됨
    temp = []
    for p in path:
        for a in p.split('\\'):
            temp.append(a)
    # 위의 과정과 마찬가지로 슬래시(/) 경로를 처리
    # 해당 결과 path는 무조건 [폴더명, 폴더명, ... , 폴더명, 동영상이름, 확장자]가 됨
    path = []
    for p in temp:
        for a in p.split('/'):
            path.append(a)
    # path의 뒤에서 1번째는 확장자, 2번째는 동영상 이름이므로 2번째를 path로 저장
    path = path[-2]
    # 사람의 skeleton image 저장 경로를 dataset\\path\\ske로 만듦
    ske_path = 'dataset\\' + path + '\\ske'
    # 사람의 image 저장 경로를 dataset\\path\\ske로 만듦
    img_path = 'dataset\\' + path + '\\img'

    # 해당 폴더가 없다면 만듦
    print(ske_path, img_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(ske_path):
        os.makedirs(ske_path)


    # frame 읽기 실패한 횟수
    fail_count = 0

    # 토탈 몇 번째 프레임인지 세는 변수
    frame_num = 0

    # 동영상 읽음
    while video.isOpened():
        frame_num += 1

        logger.debug('+frame processing+')
        ret_val, frame = video.read()
        # 100번 이상 프레임읽기 실패하면 스톱
        if ret_val is False:
            fail_count += 1
            if fail_count > 100:
                break
            continue
        # 몇 프레임마다 1번 추출할지 지정 (현재 30프레임으로 지정)
        if frame_num % 30 != 0:
            continue
        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        logger.debug('+postprocessing+')
        # TfPoseEstimator에서 PreTrain 된 model을 통해 사람의 skeleton point를 찾아냄
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # skeleton point를 통해 한 프레임에서 사람들의 범위를 상하좌우 픽셀단위로 변환함.
        boundarys = TfPoseEstimator.get_humans_imgbox(frame, humans, 0.02)

        # boundarys : {사람1 : [left, right, up, down], 사람2 : [left, right, up, down], ... }
        for num_of_human in boundarys:
            # 각 변수들을 더 직관적으로 임시저장
            img_left = boundarys[num_of_human][0]
            img_right = boundarys[num_of_human][1]
            img_up = boundarys[num_of_human][2]
            img_down = boundarys[num_of_human][3]

            # 이미지에서 사람부분만 따서 프레임넘버_사람 넘버.jpg로 이미지 폴더에 저장한다.
            cv2.imwrite(img_path + "\\%05d_%02d.jpg" % (frame_num, num_of_human), frame[img_up:img_down, img_left:img_right])
            
            # 프레임 크기와 동일한 흰 바탕의 이미지를 그린다.
            ske_img = np.zeros(frame.shape,dtype=np.uint8)
            ske_img.fill(255) 

            # 해당 이미지 위에다가 num_of_human 번째 사람의 skeleton 이미지를 그린다.
            ske_img = TfPoseEstimator.draw_humans(ske_img, [humans[num_of_human]], imgcopy=True)

            # 흰 바탕의 스켈레톤 이미지 그려진 부분만 따서 프레임넘버_사람 넘버.jpg로 스켈레톤 이미지 폴더에 저장한다.
            cv2.imwrite(ske_path + "\\%05d_%02d.jpg" % (frame_num, num_of_human), ske_img[img_up:img_down, img_left:img_right])

        
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
# python video_to_human_box.py --video=testvideo.mp4
# python video_to_human_box.py --video=test_video/488-1_cam01_vandalism01_place09_day_spring_1.mp4
# =============================================================================
