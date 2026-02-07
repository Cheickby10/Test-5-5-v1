import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model Settings
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "ensemble")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "xgboost": 0.4,
        "lightgbm": 0.3,
        "neural_network": 0.2,
        "statistical": 0.1
    }
    
    # Feature Engineering
    FEATURE_WINDOW_SIZE: int = 10  # Last N matches for feature calculation
    ADVANCED_METRICS: bool = True
    INCLUDE_TREND_ANALYSIS: bool = True
    
    # Data Processing
    MAX_HISTORICAL_MATCHES: int = 1000
    DATA_VALIDATION_ENABLED: bool = True
    
    # Cache Settings
    CACHE_TTL: int = 3600  # seconds
    MAX_CACHE_SIZE: int = 1000
    
    # Paths
    MODEL_SAVE_PATH: str = "models/trained/"
    DATA_PATH: str = "data/processed/"
    LOG_PATH: str = "logs/"
    
    # Championship Formats Configuration
    CHAMPIONSHIP_CONFIGS: Dict[str, Dict] = {
        "FC25_5x5_Rush": {
            "players_per_team": 5,
            "match_duration": 12,
            "special_rules": ["rush_mode", "powerups"],
            "weight_factor": 1.2
        },
        "FC24_4x4": {
            "players_per_team": 4,
            "match_duration": 10,
            "special_rules": ["volley_mode"],
            "weight_factor": 1.0
        },
        "FCFC25_3x3": {
            "players_per_team": 3,
            "match_duration": 8,
            "special_rules": ["small_pitch", "fast_pace"],
            "weight_factor": 0.9
        }
    }
    
    # Performance Metrics Weights
    METRIC_WEIGHTS: Dict[str, float] = {
        "win_rate": 0.25,
        "goal_difference": 0.20,
        "recent_form": 0.15,
        "head_to_head": 0.15,
        "momentum": 0.10,
        "home_advantage": 0.10,
        "fatigue_factor": 0.05
    }

settings = Settings()
