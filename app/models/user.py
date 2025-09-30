from sqlalchemy import Column, BigInteger, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class Rol(Base):
    __tablename__ = "rol"
    
    id_rol = Column(BigInteger, primary_key=True, index=True)
    name_rol = Column(String(60), nullable=False)
    
    # Relationships
    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "user"
    
    id_user = Column(BigInteger, primary_key=True, index=True)
    name_user = Column(String(120), nullable=False)
    email = Column(String(200), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    role_id = Column(BigInteger, ForeignKey("rol.id_rol"), nullable=False)
    
    # Relationships
    role = relationship("Rol", back_populates="users")
    infractions = relationship("Infraction", back_populates="user")