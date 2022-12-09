from flask import Flask, render_template, request,redirect, url_for, session
from flask_session import Session
import urllib, hashlib
from pymongo import MongoClient

client= MongoClient("mongodb+srv://naveen:"+urllib.parse.quote("tns@9900")+"@capstone-project.1aslv4m.mongodb.net/test")


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("static\data.csv")
label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["Hazardous"]) 
categories = list(label_encoder.inverse_transform([0, 1]))
classes = list(set(data["Hazardous"]))
data.drop(["Miss Dist.(Astronomical)","Miss Dist.(lunar)","Miss Dist.(miles)","Relative Velocity km per sec","Est Dia in M(max)","Relative Velocity km per hr","Est Dia in Feet(max)", "Est Dia in Feet(min)", "Est Dia in Miles(max)", "Est Dia in Miles(min)","Est Dia in KM(max)","Est Dia in KM(min)","Neo Reference ID","Orbit ID","Name","Close Approach Date","Equinox","Epoch Date Close Approach","Orbiting Body","Orbit Determination Date","Hazardous"], axis=1, inplace=True)

X, y = data.iloc[: , :-1], data.iloc[: , -1]
X_T, X_t, y_train, y_test = train_test_split(X, y, test_size=0.3)
scaler = StandardScaler().fit(X_T)
X_train = scaler.transform(X_T)
X_test = scaler.transform(X_t)

log_reg_model = LogisticRegression().fit(X_train, y_train)
rf_model = RandomForestClassifier().fit(X_train, y_train)
dec_tree = DecisionTreeClassifier().fit(X_train,y_train)


app=Flask(__name__)
app.db=client.capstone
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register",methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email=request.form.get("email")
        phone=request.form.get("phone")
        pwd = request.form.get("password")

        details=[username,email,phone,pwd]
        for i in details:
            if i=='':
                return render_template("register.html", str="One/More Fields are Empty.")

        if app.db.users.count_documents({"username" : username}) != 0 or app.db.users.count_documents({"email" : email}):
            return render_template("register.html",str="Please choose a different Username/Email-ID")

        password=hashlib.sha256(pwd.encode())
        app.db.users.insert_one({"username":username, "password":password.hexdigest(),"email":email,"phone":phone})
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        id = request.form.get("username")
        pwd = request.form.get("password")
        password=hashlib.sha256(pwd.encode())
        if app.db.users.count_documents({"username" : id,"password":password.hexdigest()}) == 1 or app.db.users.count_documents({"email" : id,"password":password.hexdigest()}):
            session['user']=id
            return redirect(url_for('predict'))
        else:
            return render_template("login.html",str="Incorrect Username/Email-ID or Password")
    return render_template("login.html",str="")

@app.route("/details")
def details():
    if app.db.users.count_documents({"username" : session["user"]}) != 0:
        res=app.db.users.find_one({"username":session["user"]})
        return render_template("details.html",res=res)
    
    res=app.db.users.find_one({"email":session["user"]})
    return render_template("details.html",res=res)

@app.route("/logout")
def logout():
    session["user"]=""
    return redirect(url_for('login'))

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method=="GET":
        if not session["user"]:
            return redirect(url_for('login'))
    if request.method == "POST":
        a = request.form.get("a")
        b = request.form.get("b")
        c = request.form.get("c")
        d = request.form.get("d")
        e = request.form.get("e")
        f = request.form.get("f")
        g = request.form.get("g")
        h = request.form.get("h")
        i = request.form.get("i")
        j = request.form.get("j")
        k = request.form.get("k")
        l = request.form.get("l")
        m = request.form.get("m")
        n = request.form.get("n")
        o = request.form.get("o")
        p = request.form.get("p")
        q = request.form.get("q")
        r = request.form.get("r")
        s = request.form.get("s")

        arr=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s]
        for i in arr:
            if i=='':
                return render_template("predict.html", str="One/More Fields are Empty.")

        for i in arr:
            i=float(i)

        X_t.loc[len(X_t.index)]=arr
        sc = scaler.transform(X_t)
        X_t.drop(X_t.tail(1).index,inplace=True)
        d=sc[len(sc)-1]
        results=[]
        results.append(log_reg_model.predict(np.array( [d,] ))[0])
        results.append(rf_model.predict(np.array( [d,] ))[0])
        results.append(dec_tree.predict(np.array( [d,] ))[0])
        mode=stats.mode(results)
        out=mode[0]
        if(out==0):
            return render_template("predict.html", str="Asteroid is not Hazardous")
        else:
            return render_template("predict.html", str="Asteroid is Hazardous")
    return render_template("predict.html",str="")

if __name__=="__main__":
    app.run(debug=True)
