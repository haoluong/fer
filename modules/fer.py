import tensorflow as tf
import numpy as np
import time
# from modules.utils import preprocess_input, l2_norm
print(tf.__version__)
THRESHOLD = 0.6
class FER():
    emotion_dict = {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral",
    }
    def __init__(self, checkpoint_path):
        self.model = self.__load_model(checkpoint_path)

    def __load_model(self, checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        print("FER model loaded at {}".format(checkpoint_path))
        return model

    def predict(self, batch):
        batch = self.__preprocess_input(batch, True)
        embeds = self.model.predict(batch)
        return self.classify(embeds)

    def classify(self, results):
        idx_maxes = np.argmax(results, axis=1)
        prob_maxes = np.amax(results, axis=1)
        return [FER.emotion_dict[idx_maxes[i]] if prob_maxes[i] > THRESHOLD else "neutral" for i in range(len(idx_maxes))]

    @staticmethod
    def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x
        