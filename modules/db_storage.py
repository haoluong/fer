from pymongo import MongoClient
import time

class ClassEmotionStatus():
    def __init__(self, emotions, concentration, detected_at):
        self.status = self.count_emotions(emotions)
        self.status["concentration"] = concentration
        self.detected_at = detected_at

    @staticmethod
    def count_emotions(emotions):
        emotion_dict = {
                "angry": 0,
                "disgust": 0,
                "fear": 0,
                "happy": 0,
                "sad": 0,
                "surprise": 0,
                "neutral": 0,
        }
        for emo in emotions:
            emotion_dict[emo] += 1
        return emotion_dict

    def __str__(self):
      return "Status: {}: {}, detected_at: {}".format(self.status.keys(), self.status.values(), self.detected_at)

class DBStorage():
    def __init__(self):
        self.client = MongoClient("mongodb+srv://admin:admin@users-y49w0.gcp.mongodb.net/test?retryWrites=true&w=majority")
        self.db = self.client.class_emo #lay database
    
    def save(self, room_stt):
        room = self.db.class_twos  #lay bang users
        #update trang thai cua student
        new_status = room_stt.status
        new_status["detected_at"] = room_stt.detected_at
        results = room.insert_one(new_status)
        self.__write_logs("INSERT STATUS", str(room_stt))
    
    @staticmethod
    def __write_logs(action,msg):
        with open("logs/db_log.txt", "a+") as f:
            f.write("{t} : {action} - {msg}\n".format(
                t=time.strftime('%Y-%m-%d %H:%M:%S'), 
                action=action,
                msg=msg)
                )