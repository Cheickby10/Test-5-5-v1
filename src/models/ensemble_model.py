import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
import json
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleModel:
    """Ultra-performant ensemble model for FIFA virtual match predictions"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.version = "2.0.0"
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize multiple specialized models"""
        
        # XGBoost for complex patterns
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM for speed and efficiency
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Neural Network for non-linear relationships
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # Statistical model for baseline
        self.models['statistical'] = StatisticalModel()
        
    def prepare_features(self, raw_features: Dict) -> np.ndarray:
        """Prepare and engineer features"""
        
        # Basic features
        features = []
        
        # Team performance metrics
        features.extend([
            raw_features.get('team1_win_rate', 0.5),
            raw_features.get('team2_win_rate', 0.5),
            raw_features.get('team1_goals_avg', 1.5),
            raw_features.get('team2_goals_avg', 1.5),
            raw_features.get('team1_goals_conceded_avg', 1.5),
            raw_features.get('team2_goals_conceded_avg', 1.5),
        ])
        
        # Advanced metrics
        if 'advanced_metrics' in raw_features:
            adv = raw_features['advanced_metrics']
            features.extend([
                adv.get('momentum_score', 0),
                adv.get('form_score', 0),
                adv.get('consistency_score', 0),
                adv.get('clutch_performance', 0),
            ])
        
        # Head-to-head features
        if 'head_to_head' in raw_features:
            h2h = raw_features['head_to_head']
            features.extend([
                h2h.get('total_matches', 0),
                h2h.get('team1_wins', 0),
                h2h.get('team2_wins', 0),
                h2h.get('draws', 0),
                h2h.get('avg_goal_difference', 0),
            ])
        
        # Contextual features
        features.extend([
            raw_features.get('home_advantage', 0),
            raw_features.get('fatigue_factor', 0),
            raw_features.get('championship_weight', 1.0),
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train all ensemble models"""
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models
        for name, model in self.models.items():
            if name != 'statistical':  # Statistical model uses different interface
                model.fit(X_scaled, y)
                print(f"Trained {name} model")
        
        # Calculate feature importance
        self.calculate_feature_importance(X_scaled, y)
        
        self.is_trained = True
        
    def predict(self, features: Dict) -> Dict[str, Any]:
        """Make ensemble prediction with confidence scores"""
        
        # Prepare features
        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        probabilities = []
        
        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                predictions[f"{name}_probabilities"] = proba.tolist()
                probabilities.append(proba)
            elif hasattr(model, 'predict'):
                pred = model.predict(X_scaled)[0]
                predictions[f"{name}_prediction"] = int(pred)
        
        # Ensemble prediction (weighted average)
        if probabilities:
            weights = self.config.get('weights', [0.4, 0.3, 0.2, 0.1])
            weighted_proba = np.average(probabilities[:len(weights)], axis=0, weights=weights)
            
            # Final prediction
            final_prediction = np.argmax(weighted_proba)
            confidence = weighted_proba[final_prediction]
            
            predictions.update({
                "ensemble_prediction": int(final_prediction),
                "confidence_score": float(confidence),
                "probabilities": weighted_proba.tolist(),
                "recommendation": self.get_recommendation(final_prediction, confidence, features)
            })
        
        return predictions
    
    def calculate_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """Calculate and store feature importance"""
        if 'xgboost' in self.models:
            self.models['xgboost'].fit(X, y)
            self.feature_importance['xgboost'] = self.models['xgboost'].feature_importances_.tolist()
    
    def get_recommendation(self, prediction: int, confidence: float, features: Dict) -> str:
        """Generate betting recommendation"""
        
        if confidence > 0.8:
            return "STRONG_BET"
        elif confidence > 0.65:
            return "MODERATE_BET"
        elif confidence > 0.55:
            return "CAUTIOUS_BET"
        else:
            return "NO_BET"
    
    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'version': self.version
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.models = data['models']
        instance.scaler = data['scaler']
        instance.feature_importance = data['feature_importance']
        instance.version = data['version']
        instance.is_trained = True
        
        return instance

class StatisticalModel:
    """Statistical model based on Poisson distribution and team strengths"""
    
    def predict(self, features: Dict) -> Tuple[int, float]:
        """Predict using statistical methods"""
        
        # Calculate expected goals using Poisson distribution
        lambda1 = features.get('team1_expected_goals', 1.5)
        lambda2 = features.get('team2_expected_goals', 1.5)
        
        # Simulate match outcomes
        team1_goals = np.random.poisson(lambda1)
        team2_goals = np.random.poisson(lambda2)
        
        # Determine winner
        if team1_goals > team2_goals:
            prediction = 0  # Team 1 wins
        elif team2_goals > team1_goals:
            prediction = 1  # Team 2 wins
        else:
            prediction = 2  # Draw
        
        # Calculate confidence
        goal_diff = abs(team1_goals - team2_goals)
        confidence = min(0.9, 0.5 + goal_diff * 0.1)
        
        return prediction, confidence
