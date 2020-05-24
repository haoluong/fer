
import os
import cv2
import numpy as np
import tensorflow as tf 
from modules.network import RetinaFaceModel
from utils.align_face import FaceAligner
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

class RetinaFace():
    def __init__(self, cfg_path):
        self.model = self.__create_model(cfg_path)
        self.aligner = FaceAligner(64)

    def __create_model(self, cfg_path):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        logger = tf.get_logger()
        logger.disabled = True
        # logger.setLevel(logging.FATAL)
        set_memory_growth()
        cfg = load_yaml(cfg_path)
        model = RetinaFaceModel(cfg, training=False, iou_th=0.4,
                                score_th=0.5)
        # load checkpoint
        checkpoint_dir = './checkpoints/' + cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=model)
        if tf.train.latest_checkpoint(checkpoint_dir):
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("[*] load ckpt from {}.".format(
                tf.train.latest_checkpoint(checkpoint_dir)))
        else:
            print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
            exit()
        return model

    def __detect_faces(self, frame):
        img = np.float32(frame.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width,_ = img.shape
        # if img.shape[1] != 640:
        #     img = cv2.resize(img, (int(img_height/img_width*640),640), interpolation=cv2.INTER_CUBIC)
        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=32)
        
        # run model
        outputs = self.model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        return outputs

    def extract_faces(self, frame):
        frame_height, frame_width,_ = frame.shape
        outputs = self.__detect_faces(frame)
        b_boxes, keypoints = [], []
        results = np.empty((0,64,64,1))
        for ann in outputs:
            if ann[-1] < 0.8: 
                continue
            b_box = max(int(ann[0] * frame_width),0), max(int(ann[1] * frame_height),0), \
                    min(int(ann[2] * frame_width), frame_width), min(int(ann[3] * frame_height), frame_height)
            # if (b_box[0]<0) or (b_box[1]<0) or (b_box[2]>=frame_width) or (b_box[3]>=frame_height):
            #     continue
            keypoint = {
                'left_eye': (int(ann[4] * frame_width),int(ann[5] * frame_height)),
                'right_eye': (int(ann[6] * frame_width),int(ann[7] * frame_height)),
                'center_eye': (int((ann[4]+ann[6]) * frame_width/2),int((ann[5]+ann[7]) * frame_height/2)),
                'nose': (int(ann[8]* frame_width), int(ann[9]* frame_height)),
                'left_mouth': (int(ann[10] * frame_width), int(ann[11] * frame_height)),
                'right_mouth': (int(ann[12] * frame_width), int(ann[13] * frame_height)),
                'center_mouth': (int((ann[10]+ann[12]) * frame_width/2),int((ann[11]+ann[13]) * frame_height/2)),
            }
            out_frame = self.aligner.align(frame, keypoint, b_box)
            out_frame =cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
            b_boxes.append(ann[:4])
            keypoints.append(keypoint)
            results = np.vstack([results, out_frame.reshape(1,64,64,1)])
        return b_boxes, keypoints, results
        