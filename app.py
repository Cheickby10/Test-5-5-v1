import os
import sys
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field

# Import modules
sys.path.append('src')
from src.data.extractor import MatchDataExtractor
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.features.metrics_calculator import PerformanceMetricsCalculator
from src.models.predictor import MatchPredictor
from src.models.ensemble_model import AdvancedEnsembleModel
from src.analysis.match_analyzer import MatchAnalyzer
from src.utils.logger import setup_logger
from src.utils.cache_manager import CacheManager
from config import settings

# Setup
logger = setup_logger(__name__)
app = FastAPI(title="FIFA Virtual Analytics Bot", version="2.0")
cache = CacheManager()

# Pydantic models
class MatchData(BaseModel):
    raw_text: str
    championship_format: str = Field(..., regex="^(FC25_5x5_Rush|FC24_4x4|FCFC25_3x3)$")
    timestamp: Optional[str] = None

class PredictionRequest(BaseModel):
    team1: str
    team2: str
    championship_format: str
    historical_context: Optional[Dict] = None

class BatchPredictionRequest(BaseModel):
    matches: List[Tuple[str, str]]
    championship_format: str

class BotConfig(BaseModel):
    model_type: str = "ensemble"
    confidence_threshold: float = 0.7
    include_advanced_metrics: bool = True

# Global instances
predictor = None
ensemble_model = None
feature_engineer = None

@app.on_event("startup")
async def startup_event():
    """Initialize bot components on startup"""
    global predictor, ensemble_model, feature_engineer
    
    logger.info("Initializing FIFA Virtual Analytics Bot...")
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    
    # Load or train models
    predictor = MatchPredictor()
    ensemble_model = AdvancedEnsembleModel()
    
    # Warm up cache with historical data if available
    await cache.warmup()
    
    logger.info("Bot initialization complete")

@app.post("/extract-match-data")
async def extract_match_data(match_data: MatchData):
    """Extract structured data from raw match text"""
    try:
        extractor = MatchDataExtractor(match_data.championship_format)
        extracted_data = extractor.parse(match_data.raw_text)
        
        # Calculate performance metrics
        metrics_calc = PerformanceMetricsCalculator()
        enhanced_data = metrics_calc.calculate_all_metrics(extracted_data)
        
        # Cache the data
        cache_key = f"match_{datetime.now().timestamp()}"
        await cache.set(cache_key, enhanced_data, ttl=3600)
        
        return {
            "status": "success",
            "data": enhanced_data,
            "cache_key": cache_key
        }
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-single-match")
async def predict_single_match(request: PredictionRequest):
    """Predict outcome for a single match"""
    try:
        # Prepare features
        features = feature_engineer.prepare_features(
            request.team1, 
            request.team2, 
            request.championship_format,
            request.historical_context
        )
        
        # Get predictions from ensemble model
        predictions = ensemble_model.predict(features)
        
        # Analyze confidence
        analyzer = MatchAnalyzer()
        analysis = analyzer.analyze_prediction(
            predictions, 
            request.team1, 
            request.team2
        )
        
        return {
            "status": "success",
            "predictions": predictions,
            "analysis": analysis,
            "confidence_score": analysis.get("confidence_score", 0),
            "recommendation": analysis.get("recommendation", "")
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Batch prediction for multiple matches"""
    try:
        results = []
        
        for team1, team2 in request.matches:
            features = feature_engineer.prepare_features(
                team1, team2, request.championship_format
            )
            prediction = ensemble_model.predict(features)
            results.append({
                "match": f"{team1} vs {team2}",
                "prediction": prediction
            })
        
        # Store results in cache for later retrieval
        batch_id = f"batch_{datetime.now().timestamp()}"
        await cache.set(batch_id, results, ttl=7200)
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "matches_processed": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
async def train_model(background_tasks: BackgroundTasks):
    """Retrain models with latest data"""
    background_tasks.add_task(retrain_models)
    return {"status": "training_started", "message": "Model retraining initiated in background"}

@app.get("/bot-stats")
async def get_bot_stats():
    """Get bot performance statistics"""
    stats = {
        "uptime": cache.get("bot_uptime"),
        "predictions_made": cache.get("total_predictions", 0),
        "accuracy_rate": predictor.get_accuracy() if predictor else None,
        "model_version": ensemble_model.version if ensemble_model else "1.0",
        "cache_hit_rate": await cache.get_hit_rate()
    }
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

async def retrain_models():
    """Background task to retrain models"""
    logger.info("Starting model retraining...")
    
    # Implementation for retraining
    # This would load new data, retrain models, and update the ensemble
    
    logger.info("Model retraining complete")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
)
