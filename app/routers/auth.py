from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.auth import UserCreate, UserLogin, UserResponse, Token, RolCreate, RolResponse
from ..services.auth_service import AuthService
from typing import List

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account"""
    return AuthService.create_user(db, user)


@router.post("/login", response_model=Token)
def login(user_login: UserLogin, db: Session = Depends(get_db)):
    """Login user and return access token"""
    result = AuthService.login_user(db, user_login)
    return {
        "access_token": result["access_token"],
        "token_type": result["token_type"]
    }


@router.post("/roles", response_model=RolResponse, status_code=status.HTTP_201_CREATED)
def create_role(role: RolCreate, db: Session = Depends(get_db)):
    """Create a new role"""
    return AuthService.create_role(db, role.name_rol)


@router.get("/roles", response_model=List[RolResponse])
def get_roles(db: Session = Depends(get_db)):
    """Get all roles"""
    return AuthService.get_roles(db)