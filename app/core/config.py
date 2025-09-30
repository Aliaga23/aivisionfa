from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    # Vision/ML settings
    vehicle_model_path: str = "yolov8n.pt"
    orientation_model_path: str = "car_orientation_model.pth"
    image_format: str = "webp"  # webp | jpg
    cors_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()