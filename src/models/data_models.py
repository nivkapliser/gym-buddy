from enum import Enum

# Stores the exercises names (for better naming convention) --> need to add more exercises and connect to database
class Exercise(Enum):
    """Class for defining exercises for better naming convention"""
    UNKNOWN = "Unknown Movement"
    SQUAT = "Squat"
    PUSHUP = "Push-up"
