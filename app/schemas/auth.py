from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


class UserBase(BaseModel):
    name_user: str
    email: EmailStr
    active: Optional[bool] = True


class UserCreate(UserBase):
    password: str
    role_id: int


class UserUpdate(BaseModel):
    name_user: Optional[str] = None
    email: Optional[EmailStr] = None
    active: Optional[bool] = None
    role_id: Optional[int] = None


class UserResponse(UserBase):
    id_user: int
    created_at: datetime
    role_id: int
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class RolBase(BaseModel):
    name_rol: str


class RolCreate(RolBase):
    pass


class RolResponse(RolBase):
    id_rol: int
    
    class Config:
        from_attributes = True