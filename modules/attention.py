import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class Attentor():
    def __init__(self):
        self.cluster = DBSCAN(eps=0.5, min_samples=5)
    
    def get_attention(self, emotions, keypoints, b_boxes, frame_height, frame_width):
        results = [self.get_eye_direction(emo, keypoint, b_box,frame_height, frame_width) for emo, keypoint, b_box in zip(emotions,keypoints, b_boxes)]
        eye_direction = [[res[1]] for res in results]
        emos = [res[0] for res in results]
        direction_cluster = self.cluster.fit(eye_direction)
        return emos, Attentor.__cal_normal_percentage(direction_cluster.labels_)

    @staticmethod
    def __cal_normal_percentage(array):
        abnormals = np.where(array == -1)[0]
        return 1-abnormals.shape[0]/array.shape[0]

    def get_eye_direction(self, emo, keypoint, b_box, frame_height, frame_width):
        b_box = int(b_box[0] * frame_width), int(b_box[1] * frame_height), \
                        int(b_box[2] * frame_width), int(b_box[3] * frame_height)
        center_face = np.asarray([(b_box[0]+b_box[2])/2, (b_box[1]+b_box[3])/2])
        tail = np.asarray([center_face[0]+keypoint["center_eye"][0]-keypoint["center_mouth"][0],
                            center_face[1]+keypoint["center_eye"][1]-keypoint["center_mouth"][1]])
        nose = np.asarray(keypoint["nose"])
        face_dvec = tail-center_face
        nose_projected = center_face+np.dot(nose - center_face, face_dvec)/np.dot(face_dvec, face_dvec)*(face_dvec)
        nose_orthor_vector = nose_projected-nose
        nose_dis = np.sqrt(np.dot(nose_orthor_vector,nose_orthor_vector))*nose_orthor_vector[0]/np.abs(nose_orthor_vector[0])
        cosin = 2*nose_dis/(b_box[2]-b_box[0])
        cosin = -1 if cosin < -1 else 1 if cosin > 1 else cosin
        nose_angle = np.arccos(cosin)
        mouth_nose_dis = np.sqrt(np.sum(np.square(nose_projected-np.asarray(keypoint["center_mouth"]))))
        eye_nose_dis = np.sqrt(np.sum(np.square(nose_projected-np.asarray(keypoint["center_eye"]))))
        if eye_nose_dis > 1.25*mouth_nose_dis:
            emo = "neutral"
        top_middle = np.array([frame_width/2,frame_height])
        bot_middle = np.array([frame_width/2,0])
        face_dis = np.cross(top_middle-bot_middle, top_middle-center_face)/np.linalg.norm(top_middle-bot_middle)*\
            (center_face[0]-frame_width/2)/np.abs(center_face[0]-frame_width/2)
        face_angle = np.arccos(2*face_dis/frame_width)
        face_angle = face_angle - np.pi/2 if np.abs(face_angle - np.pi/2) > np.pi/4 else 0
        return emo, nose_angle-face_angle