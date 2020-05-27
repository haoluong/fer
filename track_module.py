import cv2
import os
import numpy as np
import time
import settings
import sys
from _thread import start_new_thread
from modules.retinaface import RetinaFace
from modules.fer import FER
from modules.db_storage import ClassEmotionStatus, DBStorage
from modules.attention import Attentor
def main(argv):
    storager = DBStorage()
    print("DB connected")
    # init
    detect_model = RetinaFace(settings.CFG_RETINA)
    emotion_model = FER(settings.CHECKPOINT_PATH)
    attentor = Attentor()
    print("*All model loaded")
    if len(argv) <= 1:
        input_stream = 0
    elif argv[1] == 'rtsp':
        input_stream = settings.RTSP_ADDR
    else:
        input_stream = argv[1]
    cam = cv2.VideoCapture(input_stream)
    start_time = time.time()
    while cam.isOpened():
        _, frame = cam.read()
        if frame is None:
            print("no cam input")
            break
        frame_height, frame_width, _ = frame.shape
        b_boxes, keypoints, faces = detect_model.extract_faces(frame)
        results = emotion_model.predict(faces) if len(faces) > 0 else []
        labels, atten_metric = attentor.get_attention(results, keypoints, b_boxes, frame_height, frame_width) if len(b_boxes) > 0 else ([],0)
        room_stt = ClassEmotionStatus(results, atten_metric, time.strftime('%Y-%m-%d %H:%M:%S'))
        start_new_thread(storager.save, (room_stt,))
        for prior_index in range(len(b_boxes)):
            ann = b_boxes[prior_index]
            keypoint = keypoints[prior_index]
            b_box = int(ann[0] * frame_width), int(ann[1] * frame_height), \
                        int(ann[2] * frame_width), int(ann[3] * frame_height)
            label = results[prior_index]
            # cv2.circle(frame, (int((b_box[0]+b_box[2])/2),int((b_box[1]+b_box[3])/2)), 1, (255, 255, 255),2)
            # cv2.circle(frame, keypoint["left_eye"], 1, (0, 255, 0),2)
            # cv2.circle(frame, keypoint["right_eye"], 1, (0, 255, 0),2)
            # cv2.circle(frame, keypoint["nose"], 1, (0, 255, 0),2)
            # cv2.circle(frame, keypoint["left_mouth"], 1, (0, 255, 0),2)
            # cv2.circle(frame, keypoint["right_mouth"], 1, (0, 255, 0),2)
            # cv2.line(frame, keypoint["center_eye"], keypoint["center_mouth"], (0, 255, 0),2)
            # cv2.line(frame, keypoint["center_mouth"], keypoint["nose"], (0, 255, 0),2)
            # cv2.line(frame, keypoint["nose"], keypoint["center_eye"], (0, 255, 0),2)
            cv2.rectangle(frame, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (0, 255, 0), 2)
            cv2.putText(frame, labels[prior_index], (b_box[0], b_box[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.putText(frame, "{:.2f}".format(atten_metric), (25, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        cv2.putText(frame, fps_str, (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
        start_time = time.time()
        # show frame
        # cv2.imwrite('UNKNOWN/3/'+str(i)+'.jpeg', frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            exit()
        # time.sleep(1)
if __name__ == '__main__':
    main(sys.argv)
