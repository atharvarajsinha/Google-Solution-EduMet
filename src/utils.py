from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

# File paths
RAW_DATA_PATH = DATA_DIR / "raw_students.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "model.joblib"

# Encoders
SCHOOL_ENCODER_PATH = "model/enc_school.joblib"
CLASS_ENCODER_PATH = "model/enc_class.joblib"
SECTION_ENCODER_PATH = "model/enc_section.joblib"
PARENT_EDU_ENCODER_PATH = "model/enc_parent_edu.joblib"
