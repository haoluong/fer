import flask
from flask import request,jsonify, send_file
from flask_cors import CORS, cross_origin
from pymongo import MongoClient
import requests
import os
from _thread import start_new_thread
from passlib.hash import sha256_crypt
# from helpers import decode_image
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
client = MongoClient("mongodb+srv://admin:admin@users-y49w0.gcp.mongodb.net/test?retryWrites=true&w=majority")
emo_db = client.class_emo
user_db = client.user_infos
users_col = user_db.users
user_classes = user_db.user_classes
@app.route("/signup", methods=["POST"])
@cross_origin()
def signup():
    username = request.form['username'] 
    password = request.form['password']
    name = request.form['name']
    username_query = {'username': username}
    login_results = users_col.find_one(username_query) 
    data = {"success": False}
    if login_results:
        data["message"] = "Username is already taken"
        return jsonify(data)
    else:
        new_user = {
            "username": username,
            "password": sha256_crypt.encrypt(password),
            "name": name
            }
        res = users_col.insert_one(new_user)
        if res.inserted_id is not None:
            data["success"] = True
        else:
            data["message"] = "Insert fail"
    return jsonify(data)

@app.route("/login", methods=["POST"])
@cross_origin()
def login():
    username = request.form['username'] 
    password = request.form['password']
    login_query = {'username': username}
    login_results = users_col.find_one(login_query) 
    data = {"success": False}
    if login_results:
        if sha256_crypt.verify(password, login_results["password"]):
            data["success"] = True
            return jsonify(data)
    return jsonify(data)

@app.route("/add_class", methods=["POST"])
@cross_origin()
def add_class():
    username = request.form['username'] 
    class_name = request.form['class_name']
    query = {'username': username, "class_name":class_name}
    query_results = user_classes.find_one(query) 
    data = {"success": False}
    if query_results:
        data["message"] = "Classname is duplicated"
        return jsonify(data)
    else:
        new_class = {
            "username": username,
            "class_name": class_name,
            "class_id": username + "_" + class_name,
            "status": "OFF"
        }
        res = user_classes.insert_one(new_class)
        if res.inserted_id is not None:
            data["success"] = True
        else:
            data["message"] = "Insert fail"
    return jsonify(data)

@app.route("/del_class", methods=["DELETE"])
@cross_origin()
def del_class():
    username = request.form['username']
    class_name = request.form['class_name']
    query = {'username': username, "class_name":class_name}
    res = user_classes.delete_one(query) 
    data = {"success": False}
    if res.deleted_count > 0:
        data["success"] = True
    else:
        data["message"] = "Delete fail"
    return jsonify(data)

@app.route("/classes", methods=["GET"])
@cross_origin()
def get_classes():
    username = request.args.get('username')
    query = {'username': username}
    res = user_classes.find(query, {"_id":0}) 
    data = {"success": False}
    data["data"] = list(res)
    return jsonify(data)

@app.route("/update_status", methods=["POST"])
@cross_origin()
def turnon_off():
    username = request.form['username']
    class_name = request.form['class_name']
    status = request.form["status"]
    status = "ON" if status == "ON" else "OFF"
    query = {'username': username, "class_name":class_name}
    res = user_classes.update_one(query, {"$set":{"status": status}}) 
    data = {"success": False}
    if res.matched_count > 0:
        data["success"] = True
    else:
        data["message"] = "Class name doesn't match"
    return jsonify(data)

@app.route("/emotions", methods=["GET"])
@cross_origin()
def get_emotion_history():
    class_name = "class_twos"
    emo_class = emo_db[class_name]
    res = emo_class.find({}, {"_id": 0})
    return jsonify(list(res))

if __name__ == "__main__":
   print("* Starting web service...")
   app.run(port=9999,debug=True)
