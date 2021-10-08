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
    video_width_limit = 4000
    while frame.shape[1] > video_width_limit:
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    logger.info('video frame=%dx%d' % (frame.shape[1], frame.shape[0]))

    action_graph = act_class.graph

    # frame 읽기 실패한 횟수
    fail_count = 0
    video_fps = 30
    frames = []

    processing_frame = -1
    fps_time = time.time()
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    delay = round(1000/video_fps)
    out = cv2.VideoWriter('output.avi', fourcc, 10, (frame.shape[1], frame.shape[0]), isColor=True)

    while video.isOpened():
        processing_frame += 1
        frame_count += 1
        
        # logger.debug('+frame processing+')
        ret_val, frame = video.read()
        # 100번 이상 프레임읽기 실패하면 스톱
        if ret_val is False:
            fail_count += 1
            if fail_count > 100:
                break
            continue
        # if time.time() - fps_time > frame_count / video_fps:
        #     continue

        if processing_frame % 5 != 0:
            continue

        if processing_frame > 250:
            break

        fps_time = time.time()
        frame_count = 0
        # video width 가 너무 크면 속도가 느려져서 width와 height를 절반으로 downscaling함
        while frame.shape[1] > video_width_limit:
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        output_image = frame
        cv2.putText(output_image,
                            "%0d : %0d" %(int(processing_frame / video_fps / 60), int(processing_frame / video_fps % 60)),
                            (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
        
        # logger.debug('+postprocessing+')
        # TfPoseEstimator에서 PreTrain 된 model을 통해 사람의 skeleton point를 찾아냄
        humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        boundarys = TfPoseEstimator.get_humans_imgbox(frame, humans, 0.02)
        for boundary in boundarys:
            # print(boundarys[boundary])
            img_left = boundarys[boundary][0]
            img_right = boundarys[boundary][1]
            img_up = boundarys[boundary][2]
            img_down = boundarys[boundary][3]
            
            # 해당 이미지 박스의 사람 skeleton을 그린다. (출력용)
            # sub_img = TfPoseEstimator.draw_humans(frame, [humans[boundary]], imgcopy=True)
            # 흰 바탕의 skeleton이미지만 남긴 것을 그린다. (판정용)
            sub_img_ske = np.zeros(frame.shape,dtype=np.uint8)
            sub_img_ske.fill(255) 
            sub_img_ske = TfPoseEstimator.draw_humans(sub_img_ske, [humans[boundary]], imgcopy=False)
            # 흰 바탕의 skeleton이미지로 어떤 액션인지(normal / abnormal) 판정한다.
            action_class = act_class.classify(sub_img_ske[img_up:img_down, img_left:img_right], graph=action_graph)

            # 판정한 것을 박스별로 출력한다.
            if action_class[0] == 'a':
                cv2.putText(output_image,
                            "%s" %(action_class),
                            (img_left, img_up + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                cv2.rectangle(output_image, (img_left, img_up), (img_right, img_down), (0, 0, 255))
            # normal
            elif action_class[0] == 'n':
                cv2.putText(output_image,
                            "%s" %(action_class),
                            (img_left, img_up + 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.rectangle(output_image, (img_left, img_up), (img_right, img_down), (0, 255, 0))
            
            # 찾아낸 skeleton point를 output image에 점과 연결선으로 표시
            # output_image = TfPoseEstimator.draw_humans(frame, [humans[boundary]], imgcopy=False)
        
        cv2.imshow('tf-pose-estimation result', output_image)

        out.write(output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # logger.debug('+finished+')
        
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
# python run_video.py --video=test_video/488-1_cam01_vandalism01_place09_day_spring_1.mp4
# =============================================================================
