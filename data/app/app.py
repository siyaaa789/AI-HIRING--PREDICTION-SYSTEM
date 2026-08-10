
from pathlib import Path
import json, joblib, pandas as pd
from flask import Flask,render_template,request
BASE=Path(__file__).resolve().parents[1]
MODEL=joblib.load(BASE/"models/hiring_model.pkl")
METRICS=json.loads((BASE/"models/metrics.json").read_text())
app=Flask(__name__)
SKILLS=["Java, HTML, CSS","Python, SQL, Pandas","JavaScript, React, HTML, CSS","Python, Scikit-learn, SQL","Python, Networking, Linux"]
EDUS=["Bachelor's","Master's","PhD"]
ROLES=["Software Developer","Data Analyst","Web Developer","ML Engineer","Cybersecurity Analyst"]
@app.route("/",methods=["GET","POST"])
def index():
    result=probability=error=None; form={}
    if request.method=="POST":
        form=request.form.to_dict()
        try:
            x=pd.DataFrame([{"Skills":form["Skills"],"Experience (Years)":float(form["Experience (Years)"]),"Education":form["Education"],"Certifications":int(form["Certifications"]),"Job Role":form["Job Role"],"Salary Expectation ($)":float(form["Salary Expectation ($)"]),"Projects Count":int(form["Projects Count"])}])
            p=int(MODEL.predict(x)[0]); probability=round(float(MODEL.predict_proba(x)[0][1])*100,1); result="Hire" if p else "Reject"
        except Exception as e: error=str(e)
    return render_template("index.html",result=result,probability=probability,error=error,form=form,skills=SKILLS,educations=EDUS,roles=ROLES,metrics=METRICS)
if __name__=="__main__": app.run(debug=True)
