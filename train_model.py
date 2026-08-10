
from pathlib import Path
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

BASE=Path(__file__).resolve().parents[1]
df=pd.read_csv(BASE/"data/raw/hiring_dataset.csv")
FEATURES=["Skills","Experience (Years)","Education","Certifications","Job Role","Salary Expectation ($)","Projects Count"]
X=df[FEATURES]; y=df["Recruiter Decision"].map({"Reject":0,"Hire":1})
num=["Experience (Years)","Certifications","Salary Expectation ($)","Projects Count"]
cat=["Skills","Education","Job Role"]
prep=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("preprocessor",prep),("classifier",RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr); pred=model.predict(Xte)
metrics={"accuracy":round(float(accuracy_score(yte,pred)),4),"precision":round(float(precision_score(yte,pred,zero_division=0)),4),"recall":round(float(recall_score(yte,pred,zero_division=0)),4),"f1":round(float(f1_score(yte,pred,zero_division=0)),4),"dataset_rows":len(df)}
(BASE/"models").mkdir(exist_ok=True)
joblib.dump(model,BASE/"models/hiring_model.pkl")
(BASE/"models/metrics.json").write_text(json.dumps(metrics,indent=2))
print(metrics)
