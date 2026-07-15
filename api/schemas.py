from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Optional

class METARObservation(BaseModel):
    timestamp:  str   = Field(..., description="UTC timestamp YYYY-MM-DD HH:MM:SS")
    wind_dir:   float = Field(..., ge=0,   le=360,  description="degrees")
    wind_speed: float = Field(..., ge=0,   le=80,   description="knots")
    gust:       float = Field(..., ge=0,   le=100,  description="knots")
    visibility: float = Field(..., ge=0,   le=9999, description="metres")
    temp:       float = Field(..., ge=-10, le=50,   description="celsius")
    dewpoint:   float = Field(..., ge=-20, le=40,   description="celsius")
    pressure:   float = Field(..., ge=900, le=1050, description="hPa")

    @field_validator('wind_dir')
    @classmethod
    def wind_dir_valid(cls, v):
        if not (0 <= v <= 360):
            raise ValueError(f"wind_dir {v} out of range [0, 360]")
        return v

class BatchMETARRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "observations": [{
                    "timestamp": "2024-01-01 00:00:00",
                    "wind_dir": 270, "wind_speed": 10, "gust": 15,
                    "visibility": 5000, "temp": 28, "dewpoint": 22, "pressure": 1008
                }]
            }
        }
    )
    observations: list[METARObservation] = Field(
        ..., min_length=1, description="List of METAR observations, oldest first, 30-min intervals"
    )

class HorizonPrediction(BaseModel):
    temperature_c:  float
    wind_speed_kt:  float
    wind_dir_deg:   float
    gust_kt:        float
    pressure_hpa:   float
    visibility_m:   float

class FogAlert(BaseModel):
    low_vis_flag:  bool
    probability:   float
    threshold_m:   int
    icao_category: str

class ModelInfo(BaseModel):
    pressure_model:  str
    other_variables: str
    fog_classifier:  str
    horizon_method:  str

class ForecastResponse(BaseModel):
    station:       str
    station_name:  str
    forecast_time: str
    issued_for:    str
    horizons:      Dict[str, HorizonPrediction]
    fog_alert:     FogAlert
    model_info:    ModelInfo

class HealthResponse(BaseModel):
    status:        str
    models_loaded: bool
    station:       str
    timestamp:     str
    version:       str
