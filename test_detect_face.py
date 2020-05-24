from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from modules.network import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)
from fer.util import preprocess_input, decode_emotion

flags.DEFINE_string('cfg_path', './configs/retinaface_mbv2.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_boolean('webcam', True, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')
flags.DEFINE_string('output_path', '', 'path of output for saving')
flags.DEFINE_string('input_path', '', 'path of input for saving')
def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)
    emotion_model = tf.keras.models.load_model("fer/data/emotion_model.hdf5")
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

    if not FLAGS.webcam:
        if not os.path.exists(FLAGS.img_path):
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

        print("[*] Processing on single image {}".format(FLAGS.img_path))

        img_raw = cv2.imread(FLAGS.img_path)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                             fy=FLAGS.down_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()
        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw and save results
        save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
        for prior_index in range(len(outputs)):
            draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw,
                            img_width_raw)
            cv2.imwrite(save_img_path, img_raw)
        print(f"[*] save result at {save_img_path}")

    else:
        print(FLAGS.input_path)
        if FLAGS.input_path == '':
            cam = cv2.VideoCapture(0)
        else:
            cam = cv2.VideoCapture(FLAGS.input_path)
        if FLAGS.output_path != '':
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            fps = cam.get(cv2.CAP_PROP_FPS)
            h, w = (int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            out = cv2.VideoWriter(FLAGS.output_path,fourcc, fps, (w,h), True)
            print(f"[*] save result at {FLAGS.output_path} with FPS = {fps} height = {h} width = {w}")
        start_time = time.time()
        while cam.isOpened():
            ret, frame = cam.read()
            if ret is False:
                print("no cam input")
                break

            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                 fy=FLAGS.down_scale_factor,
                                 interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()
            # print(outputs)

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)
            # draw results
            for prior_index in range(len(outputs)):
                ann = outputs[prior_index]
                #get bounding box
                b_box = max(int(ann[0] * frame_width),0), max(int(ann[1] * frame_height),0), \
                        min(int(ann[2] * frame_width),frame_width), min(int(ann[3] * frame_height),frame_height)
                left_eye = (int(ann[4]*frame_width), int(ann[5]*frame_height))
                right_eye = (int(ann[6]*frame_width), int(ann[7]*frame_height))
                center_eye = (int((ann[4]+ann[6])*frame_width/2), int((ann[5]+ann[7])*frame_height/2))
                nose = (int(ann[8]*frame_width), int(ann[9]*frame_height))
                left_mouth = (int(ann[10]*frame_width), int(ann[11]*frame_height))
                right_mouth = (int(ann[12]*frame_width), int(ann[13]*frame_height))
                center_mouth = (int((ann[10]+ann[12])*frame_width/2), int((ann[11]+ann[13])*frame_height/2))
                cv2.circle(frame, left_eye, 1, (0, 255, 0),2)
                cv2.circle(frame, right_eye, 1, (0, 255, 0),2)
                cv2.circle(frame, nose, 1, (0, 255, 0),2)
                cv2.circle(frame, left_mouth, 1, (0, 255, 0),2)
                cv2.circle(frame, right_mouth, 1, (0, 255, 0),2)
                cv2.line(frame, center_eye, center_mouth, (0, 255, 0),2)
                cv2.line(frame, center_mouth, nose, (0, 255, 0),2)
                cv2.line(frame, nose, center_eye, (0, 255, 0),2)
                face_frame = frame[b_box[1]:b_box[3], b_box[0]:b_box[2], :]
                gray_img = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                scaled = cv2.resize(gray_img, (64,64), interpolation=cv2.INTER_CUBIC)
                scaled = preprocess_input(scaled, True)
                reshaped = scaled.reshape((1,64,64,1))
                res = emotion_model.predict(reshaped)[0]
                #decode predict
                if np.amax(res) > 0.6:
                    text = "{:.2f}".format(np.amax(res))
                    #cv2.putText(frame, text, (b_box[0], b_box[1]+15),
                    #        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    label = decode_emotion(np.argmax(res))
                else:
                    label = "neutral"
                #draw
                
                cv2.rectangle(frame, (b_box[0], b_box[1]), (b_box[2], b_box[3]), (0, 255, 0), 1)
                cv2.putText(frame, label, (b_box[0], b_box[1]),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(frame, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
            if FLAGS.output_path != '':
                out.write(frame)
            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()
        cam.release()
        if FLAGS.output_path != '':
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
