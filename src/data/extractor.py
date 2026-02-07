import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

class MatchDataExtractor:
    """Advanced data extractor for virtual FIFA match text"""
    
    def __init__(self, championship_format: str):
        self.format = championship_format
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict:
        """Initialize regex patterns for different formats"""
        
        patterns = {
            "FC25_5x5_Rush": {
                "match": r'(\w+(?:\s+\w+)*)\s+(\d+)\s*-\s*(\d+)\s+(\w+(?:\s+\w+)*)',
                "details": r'\((\w+)\s+vs\s+(\w+)\)',
                "timestamp": r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})'
            },
            "FC24_4x4": {
                "match": r'([A-Za-z0-9_]+)\s*:\s*(\d+)\s*-\s*([A-Za-z0-9_]+)\s*:\s*(\d+)',
                "details": r'Match\s+#(\d+)',
                "timestamp": r'@\s*(\d{2}:\d{2})'
            },
            "FCFC25_3x3": {
                "match": r'(\S+)\s+(\d+)\s*/\s*(\d+)\s+(\S+)',
                "details": r'\[(.*?)\]',
                "timestamp": r'Time:\s*(\d{2}h\d{2})'
            }
        }
        
        return patterns.get(self.format, patterns["FC25_5x5_Rush"])
    
    def parse(self, text: str) -> Dict:
        """Parse raw text and extract match data"""
        
        result = {
            "teams": [],
            "scores": [],
            "timestamp": None,
            "extracted_data": {},
            "raw_text": text,
            "format": self.format
        }
        
        # Extract match result
        match_pattern = self.patterns["match"]
        match_result = re.search(match_pattern, text)
        
        if match_result:
            groups = match_result.groups()
            
            if len(groups) >= 4:
                result["teams"] = [groups[0].strip(), groups[3].strip()]
                result["scores"] = [int(groups[1]), int(groups[2])]
                result["extracted_data"]["raw_match"] = groups
            elif len(groups) == 2:
                result["teams"] = [groups[0].strip()]
                result["scores"] = [int(groups[1])]
        
        # Extract timestamp
        timestamp_pattern = self.patterns.get("timestamp")
        if timestamp_pattern:
            timestamp_match = re.search(timestamp_pattern, text)
            if timestamp_match:
                result["timestamp"] = timestamp_match.group(1)
        
        # Extract additional details
        details_pattern = self.patterns.get("details")
        if details_pattern:
            details_match = re.search(details_pattern, text)
            if details_match:
                result["extracted_data"]["details"] = details_match.groups()
        
        # Calculate derived metrics
        result = self._calculate_derived_metrics(result)
        
        return result
    
    def _calculate_derived_metrics(self, data: Dict) -> Dict:
        """Calculate derived metrics from extracted data"""
        
        if len(data["scores"]) >= 2:
            score1, score2 = data["scores"][:2]
            
            data["derived_metrics"] = {
                "goal_difference": abs(score1 - score2),
                "total_goals": score1 + score2,
                "winner": data["teams"][0] if score1 > score2 else data["teams"][1] if score2 > score1 else "Draw",
                "is_draw": score1 == score2,
                "winning_margin": abs(score1 - score2)
            }
        
        return data
    
    def batch_parse(self, texts: List[str]) -> List[Dict]:
        """Parse multiple text entries"""
        return [self.parse(text) for text in texts]
    
    def validate_extraction(self, extracted_data: Dict) -> bool:
        """Validate extracted data"""
        
        required_fields = ["teams", "scores"]
        
        for field in required_fields:
            if field not in extracted_data or not extracted_data[field]:
                return False
        
        if len(extracted_data["teams"]) < 1 or len(extracted_data["scores"]) < 1:
            return False
        
        # Validate scores are positive
        if any(score < 0 for score in extracted_data["scores"]):
            return False
        
        return True
