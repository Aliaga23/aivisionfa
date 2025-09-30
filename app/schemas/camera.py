from pydantic import BaseModel
from typing import Optional


class CameraBase(BaseModel):
    name_camera: str
    location_camera: Optional[str] = None


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name_camera: Optional[str] = None
    location_camera: Optional[str] = None
    active: Optional[bool] = None


class CameraResponse(CameraBase):
    id_camera: int
    active: bool
    
    class Config:
        from_attributes = True