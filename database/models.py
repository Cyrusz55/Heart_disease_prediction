from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class CleanedHeartDisease(Base):
    __tablename__ = "cleaned_heart_disease"
    id = Column(Integer, primary_key = True, autoincrement=True)

    """all_cols = [
        "age",
        "sex",
        "chest_pains",
        "resting_blood_pressure",
        "cholesterol",
        "fasting_blood_sugar",
        "rest_ecg",
        "max_heart_rate",
        "exercise_induced_angina",
        "oldpeak",
        "slope",
        "num_major_vessels",
        "thalassemia",
        "target",
    ]"""
    age = Column(Integer, nullable = True)
    sex = Column(String, nullable = True)
    chest_pains = Column(Integer, nullable = True)
    resting_blood_pressure = Column(Integer, nullable = True)
    cholesterol = Column(Integer, nullable=True)
    fasting_blood_sugar = Column(Integer, nullable = True)
    rest_ecg = Column(Integer, nullable = True)
    max_heart_rate = Column(Integer, nullable = True)
    exercise_induced_angina = Column(Integer, nullable=True)
    oldpeak = Column(Integer, nullable = True)
    slope = Column(Integer, nullable = True)
    thalassemia = Column(Integer, nullable = True)
    target = Column(Integer, nullable = True)

def create_tables(engine):
    Base.metadata.create_all(engine)
    print("[db] Tables created")