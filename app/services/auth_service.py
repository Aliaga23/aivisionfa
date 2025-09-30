from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.user import User, Rol
from ..schemas.auth import UserCreate, UserLogin
from ..core.security import get_password_hash, verify_password, create_access_token
from ..core.config import settings
from datetime import timedelta
from typing import Optional


class AuthService:
    
    @staticmethod
    def create_user(db: Session, user: UserCreate) -> User:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Verify role exists
        role = db.query(Rol).filter(Rol.id_rol == user.role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role not found"
            )
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        db_user = User(
            name_user=user.name_user,
            email=user.email,
            password_hash=hashed_password,
            role_id=user.role_id
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        user = db.query(User).filter(User.email == email, User.active == True).first()
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user
    
    @staticmethod
    def login_user(db: Session, user_login: UserLogin):
        user = AuthService.authenticate_user(db, user_login.email, user_login.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user
        }
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email, User.active == True).first()
    
    @staticmethod
    def create_role(db: Session, name_rol: str) -> Rol:
        existing_role = db.query(Rol).filter(Rol.name_rol == name_rol).first()
        if existing_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role already exists"
            )
        
        role = Rol(name_rol=name_rol)
        db.add(role)
        db.commit()
        db.refresh(role)
        return role
    
    @staticmethod
    def get_roles(db: Session):
        return db.query(Rol).all()