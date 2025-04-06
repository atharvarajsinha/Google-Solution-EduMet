import pandas as pd
import joblib
from utils import MODEL_PATH, SCALER_PATH
from utils import SCHOOL_ENCODER_PATH, CLASS_ENCODER_PATH, SECTION_ENCODER_PATH, PARENT_EDU_ENCODER_PATH

# Load encoders and models globally
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

school_encoder = joblib.load(SCHOOL_ENCODER_PATH)
class_encoder = joblib.load(CLASS_ENCODER_PATH)
section_encoder = joblib.load(SECTION_ENCODER_PATH)
parent_edu_encoder = joblib.load(PARENT_EDU_ENCODER_PATH)


def safe_label_encode(encoder, value, default=0):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return default


def preprocess_input(df):
    """Preprocess the DataFrame for prediction."""

    df['Sex'] = df['Sex'].str.lower().map({'male': 1, 'female': 0})
    df['Extra_Curricular'] = df['Extra_Curricular'].astype(int)

    # Safe encoding
    df['School'] = df['School'].apply(lambda x: safe_label_encode(school_encoder, x))
    df['Class'] = df['Class'].astype(str).apply(lambda x: safe_label_encode(class_encoder, x))
    df['Section'] = df['Section'].astype(str).apply(lambda x: safe_label_encode(section_encoder, x))
    df['Parental_Education'] = df['Parental_Education'].astype(str).apply(lambda x: safe_label_encode(parent_edu_encoder, x))

    # Fill missing fields if any
    defaults = {
        'Participation_Score': 0,
        'Teacher_Rating': 0,
        'Discipline_Issues': 0,
        'Late_Submissions': 0,
        'Previous_Grade_1': 0,
        'Previous_Grade_2': 0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    # Scale numeric columns
    scaled_cols = [
        'Attendance_Percentage',
        'Study_Hours_Per_Week',
        'Participation_Score',
        'Teacher_Rating',
        'Discipline_Issues',
        'Late_Submissions',
        'Previous_Grade_1',
        'Previous_Grade_2'
    ]
    df[scaled_cols] = scaler.transform(df[scaled_cols])

    return df


def predict_grade(input_data):
    """Predict final grade for a single student dictionary input."""

    input_df = pd.DataFrame([input_data])
    processed_df = preprocess_input(input_df)
    processed_df = processed_df.drop(['School', 'Student_ID', 'Name'], axis=1, errors='ignore')

    predicted_grade = model.predict(processed_df)[0]
    return round(predicted_grade, 2)


def predict_bulk(csv_path):
    """Predict grades for multiple students from a CSV."""
    
    df = pd.read_csv(csv_path)
    original_ids = df[['Student_ID', 'Name']].copy()

    processed_df = preprocess_input(df)
    features = processed_df.drop(['School', 'Student_ID', 'Name'], axis=1, errors='ignore')

    preds = model.predict(features)
    original_ids['Predicted_Grade'] = preds.round(2)
    return original_ids


# Example usage
if __name__ == "__main__":
    sample_student = {
        'School': 'ABC School',
        'Student_ID': 'S001',
        'Name': 'John Doe',
        'Sex': 'Male',
        'Class': '5',
        'Section': 'A',
        'Attendance_Percentage': 85.0,
        'Homework_Completed': 0,
        'Parental_Education': 'Secondary',
        'Study_Hours_Per_Week': 16,
        'Failures': 2,
        'Extra_Curricular': 0,
        'Participation_Score': 1,
        'Teacher_Rating': 2,
        'Discipline_Issues': 2,
        'Late_Submissions': 9,
        'Previous_Grade_1': 100,
        'Previous_Grade_2': 70
    }

    print("Predicted Grade:", predict_grade(sample_student))

    # bulk_result = predict_bulk("data/class5A_students.csv")
    # bulk_result.to_csv("data/class5A_predictions.csv", index=False)
    # print("Bulk predictions saved to: data/class5A_predictions.csv")
