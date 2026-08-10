from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "hiring_model.pkl"
model = joblib.load(MODEL_PATH)


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Hiring Prediction System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 650px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #222;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 6px;
        }

        button {
            width: 100%;
            padding: 12px;
            margin-top: 25px;
            border: none;
            border-radius: 6px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            background: #eef2ff;
            border-radius: 8px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>AI Hiring Prediction System</h1>

    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" placeholder="e.g. Siya Pradhan">

        <label>Skills</label>
        <input type="text" name="skills"
               placeholder="e.g. Python, Machine Learning"
               required>

        <label>Experience (Years)</label>
        <input type="number" name="experience"
               min="0" step="0.1" required>

        <label>Education</label>
        <input type="text" name="education"
               placeholder="e.g. B.Tech"
               required>

        <label>Certifications</label>
        <input type="number" name="certifications"
               min="0" step="1" required>

        <label>Job Role</label>
        <input type="text" name="job_role"
               placeholder="e.g. Software Engineer"
               required>

        <label>Salary Expectation ($)</label>
        <input type="number" name="salary"
               min="0" step="1" required>

        <label>Projects Count</label>
        <input type="number" name="projects"
               min="0" step="1" required>

        <button type="submit">Predict Hiring Decision</button>

    </form>

    {% if result %}
        <div class="result">
            Prediction: {{ result }}
        </div>
    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        candidate = pd.DataFrame([{
            "Skills": request.form["skills"],
            "Experience (Years)": float(request.form["experience"]),
            "Education": request.form["education"],
            "Certifications": int(request.form["certifications"]),
            "Job Role": request.form["job_role"],
            "Salary Expectation ($)": float(request.form["salary"]),
            "Projects Count": int(request.form["projects"])
        }])

        prediction = model.predict(candidate)[0]

        if prediction == 1:
            result = "Hire"
        else:
            result = "Reject"

    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)