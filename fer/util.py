def preprocess_input(x, v2=False):
    x = x.astype("float32")
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
emotion_dict = {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral",
            }
def decode_emotion(index):
    return emotion_dict[index]
