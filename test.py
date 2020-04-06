from fer import Video
from fer import FER

video_filename = "/home/hao/Downloads/faptv2.mp4"
video = Video(video_filename)

# Analyze video, displaying the output
detector = FER(mtcnn=True)
raw_data = video.analyze(detector, display=True)
df = video.to_pandas(raw_data)
# def preprocess_input(x, v2=False):
#     x = x.astype("float32")
#     x = x / 255.0
#     if v2:
#         x = x - 0.5
#         x = x * 2.0
#     return x
# import tensorflow as tf
# import cv2
# model = tf.keras.models.load_model("fer/data/emotion_model.hdf5")
# img = cv2.imread("justin.jpg",cv2.IMREAD_GRAYSCALE)
# scaled = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
# scaled = preprocess_input(scaled, True)
# reshaped = scaled.reshape((1,64,64,1))
# cv2.imshow('img', reshaped[0])
# res = model.predict(reshaped)
# print(res)

# if cv2.waitKey(0) == ord('q'):
#     exit()