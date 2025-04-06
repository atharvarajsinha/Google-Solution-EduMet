import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from utils import RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH, SCHOOL_ENCODER_PATH, CLASS_ENCODER_PATH, SECTION_ENCODER_PATH, PARENT_EDU_ENCODER_PATH

# Loading raw data
df = pd.read_csv(RAW_DATA_PATH)

# Filtering only Classes 1 to 8
df = df[df['Class'].astype(str).isin([str(i) for i in range(1, 9)])]

# Initializing LabelEncoder
le = LabelEncoder()

# Encoding categorical features
df['Sex'] = le.fit_transform(df['Sex'])
df['Extra_Curricular'] = df['Extra_Curricular'].astype(int)
df['Parental_Education'] = le.fit_transform(df['Parental_Education'])
df['School'] = le.fit_transform(df['School'])
df['Class'] = le.fit_transform(df['Class'].astype(str))
df['Section'] = le.fit_transform(df['Section'].astype(str))

def encode_and_save(column_name, path):
    le = LabelEncoder() 
    df[column_name] = le.fit_transform(df[column_name].astype(str))
    joblib.dump(le, path)

encode_and_save('School', SCHOOL_ENCODER_PATH)
encode_and_save('Class', CLASS_ENCODER_PATH)
encode_and_save('Section', SECTION_ENCODER_PATH)
encode_and_save('Parental_Education', PARENT_EDU_ENCODER_PATH)

# Handling missing values
df.fillna({
    'Attendance_Percentage': df['Attendance_Percentage'].mean(),
    'Study_Hours_Per_Week': df['Study_Hours_Per_Week'].median(),
    'Participation_Score': df['Participation_Score'].mean(),
    'Teacher_Rating': df['Teacher_Rating'].mean(),
    'Discipline_Issues': 0,
    'Late_Submissions': 0,
    'Previous_Grade_1': df['Previous_Grade_1'].mean(),
    'Previous_Grade_2': df['Previous_Grade_2'].mean(),
    'Final_Grade': df['Final_Grade'].mean(),
}, inplace=True)

# Featuring scaling
scaler = MinMaxScaler()
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
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Saving processed data and scaler
Path(PROCESSED_DATA_PATH).parent.mkdir(exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)
joblib.dump(scaler, SCALER_PATH)

print("Preprocessing complete! Saved:", PROCESSED_DATA_PATH)
