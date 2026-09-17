import joblib
import pandas as pd


MODEL_PATH = "model/student_model.pkl"


def predict_student(features):
    model = joblib.load(MODEL_PATH)

    columns = [
        "StudyHours",
        "Attendance",
        "Resources",
        "Extracurricular",
        "Motivation",
        "Internet",
        "Gender",
        "Age",
        "LearningStyle",
        "OnlineCourses",
        "Discussions",
        "AssignmentCompletion",
        "ExamScore",
        "EduTech",
        "StressLevel"
    ]

    data = pd.DataFrame([features], columns=columns)

    prediction = model.predict(data)

    return prediction[0]


if __name__ == "__main__":

    sample_student = [
        20,   # StudyHours
        80,   # Attendance
        1,    # Resources
        1,    # Extracurricular
        1,    # Motivation
        1,    # Internet
        0,    # Gender
        20,   # Age
        1,    # LearningStyle
        1,    # OnlineCourses
        1,    # Discussions
        85,   # AssignmentCompletion
        80,   # ExamScore
        1,    # EduTech
        1     # StressLevel
    ]

    result = predict_student(sample_student)

    print("Student Features:")
    print(sample_student)

    print("\nPredicted Final Grade:", result)