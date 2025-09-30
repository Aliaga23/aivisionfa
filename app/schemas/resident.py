from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


class ResidentBase(BaseModel):
    name_resident: str
    email: EmailStr
    phone: Optional[str] = None
    address: Optional[str] = None


class ResidentCreate(ResidentBase):
    pass


class ResidentUpdate(BaseModel):
    name_resident: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    active: Optional[bool] = None


class ResidentResponse(ResidentBase):
    id_resident: int
    created_at: datetime
    active: bool
    
    class Config:
        from_attributes = True