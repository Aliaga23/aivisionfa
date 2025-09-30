from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class VehicleBase(BaseModel):
    license_plate: str
    brand: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None


class VehicleCreate(VehicleBase):
    resident_id: Optional[int] = None
    visitor_id: Optional[int] = None


class VehicleUpdate(BaseModel):
    license_plate: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None
    active: Optional[bool] = None
    resident_id: Optional[int] = None
    visitor_id: Optional[int] = None


class VehicleResponse(VehicleBase):
    id_vehicle: int
    created_at: datetime
    active: bool
    resident_id: Optional[int]
    visitor_id: Optional[int]
    
    class Config:
        from_attributes = True