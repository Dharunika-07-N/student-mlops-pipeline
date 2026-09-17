import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.predict import predict_student


def test_prediction():
    sample_student = [
        20,
        80,
        1,
        1,
        1,
        1,
        0,
        20,
        1,
        1,
        1,
        85,
        80,
        1,
        1
    ]

    prediction = predict_student(sample_student)

    assert prediction in [0, 1, 2, 3]