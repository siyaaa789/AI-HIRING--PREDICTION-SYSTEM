from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "hiring_model.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "hiring_model.pkl"

model = joblib.load(MODEL_PATH)


# =========================================================
# CERTIFICATION CONVERSION
# =========================================================

def certification_to_number(certification):
    certification = certification.strip().lower()

    certification_map = {
        "none": 0,
        "no certification": 0,

        "google ml": 1,

        "aws certified": 2,
        "aws certificate": 2,

        "deep learning spec": 3,
        "deep learning specialist": 3
    }

    return certification_map.get(certification, 0)


# =========================================================
# HTML
# =========================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>AI Hiring Prediction System</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.12);
        }

        h1 {
            text-align: center;
            color: #222;
            margin-bottom: 30px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-top: 18px;
            margin-bottom: 7px;
        }

        input,
        select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 7px;
            font-size: 15px;
        }

        input:focus,
        select:focus {
            outline: none;
            border-color: #2563eb;
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 25px;
            border: none;
            border-radius: 7px;
            background: #2563eb;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background: #eef4ff;
        }

        .result h2 {
            margin-top: 0;
        }

        .score {
            font-size: 28px;
            font-weight: bold;
            margin-top: 10px;
        }

        .info {
            margin-top: 25px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            color: #555;
            font-size: 14px;
        }

        .error {
            color: #b91c1c;
            background: #fee2e2;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>AI Hiring Prediction System</h1>


    <form method="POST">

        <!-- NAME -->

        <label>Name</label>

        <input
            type="text"
            name="name"
            placeholder="e.g. Siya Pradhan"
            required
        >


        <!-- SKILLS -->

        <label>Skills</label>

        <input
            type="text"
            name="skills"
            placeholder="e.g. Python, Machine Learning"
            required
        >


        <!-- EXPERIENCE -->

        <label>Experience (Years)</label>

        <input
            type="number"
            name="experience"
            step="0.1"
            min="0"
            placeholder="e.g. 3"
            required
        >


        <!-- EDUCATION -->

        <label>Education</label>

        <input
            type="text"
            name="education"
            placeholder="e.g. B.Tech"
            required
        >


        <!-- CERTIFICATIONS -->

        <label>Certifications</label>

        <select name="certifications" required>

            <option value="">Select Certification</option>

            <option value="None">
                None
            </option>

            <option value="Google ML">
                Google ML
            </option>

            <option value="AWS Certified">
                AWS Certified
            </option>

            <option value="Deep Learning Spec">
                Deep Learning Spec
            </option>

        </select>


        <!-- JOB ROLE -->

        <label>Job Role</label>

        <input
            type="text"
            name="job_role"
            placeholder="e.g. Software Engineer"
            required
        >


        <!-- SALARY -->

        <label>Salary Expectation ($)</label>

        <input
            type="number"
            name="salary"
            step="0.01"
            min="0"
            placeholder="e.g. 60000"
            required
        >


        <!-- PROJECTS -->

        <label>Projects Count</label>

        <input
            type="number"
            name="projects"
            min="0"
            placeholder="e.g. 5"
            required
        >


        <!-- BUTTON -->

        <button type="submit">
            Predict Hiring Decision
        </button>

    </form>


    {% if result is not none %}

        <div class="result">

            <h2>
                Prediction: {{ result }}
            </h2>

            {% if ai_score is not none %}

                <div class="score">
                    AI Score: {{ ai_score }}%
                </div>

            {% endif %}

        </div>

    {% endif %}


    <div class="info">

        <strong>How it works:</strong>

        <br><br>

        The system uses:

        <br>

        • Skills

        <br>

        • Experience

        <br>

        • Education

        <br>

        • Certifications

        <br>

        • Job Role

        <br>

        • Salary Expectation

        <br>

        • Projects Count

        <br><br>

        to predict whether a candidate is likely to be hired.

    </div>

</div>

</body>

</html>
"""


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    ai_score = None

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    if request.method == "GET":

        return render_template_string(
            HTML,
            result=result,
            ai_score=ai_score
        )


    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------

    try:

        # Convert certification text into number
        certification_value = certification_to_number(
            request.form["certifications"]
        )


        # Create candidate dataframe
        candidate = pd.DataFrame([{

            "Skills":
                request.form["skills"],

            "Experience (Years)":
                float(request.form["experience"]),

            "Education":
                request.form["education"],

            "Certifications":
                certification_value,

            "Job Role":
                request.form["job_role"],

            "Salary Expectation ($)":
                float(request.form["salary"]),

            "Projects Count":
                int(request.form["projects"])

        }])


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(candidate)[0]


        if prediction == 1:

            result = "Hire"

        else:

            result = "Reject"


        # -------------------------------------------------
        # AI SCORE
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(candidate)[0]

            ai_score = round(
                probability[1] * 100,
                2
            )

        else:

            ai_score = None


        # -------------------------------------------------
        # SHOW RESULT
        # -------------------------------------------------

        return render_template_string(
            HTML,
            result=result,
            ai_score=ai_score
        )


    except Exception as e:

        return render_template_string(
            HTML,
            result="Error: " + str(e),
            ai_score=None
        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

           