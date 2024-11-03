from enum import Enum

# Stores the exercises names (for better naming convention) --> need to add more exercises and connect to database
class Exercise(Enum):
    """Class for defining exercises for better naming convention"""
    UNKNOWN = "Unknown Movement"
    SQUAT = "Squat"
    PUSHUP = "Push-up"
    DEADLIFT = "Deadlift"
    LUNGE = "Lunge"
    PLANK = "Plank"
    PULLUP = "Pull-up"
    BENCH_PRESS = "Bench Press"
    OVERHEAD_PRESS = "Overhead Press"
    BICEP_CURL = "Bicep Curl"
    SHOULDER_PRESS = "Shoulder Press"
