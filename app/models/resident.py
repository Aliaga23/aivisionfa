from sqlalchemy import Column, BigInteger, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class Resident(Base):
    __tablename__ = "resident"
    
    id_resident = Column(BigInteger, primary_key=True, index=True)
    name_resident = Column(String(150), nullable=False)
    email = Column(String(200), nullable=False)
    phone = Column(String(50))
    address = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    
    # Relationships
    visits = relationship("Visit", back_populates="resident")
    vehicles = relationship("Vehicle", back_populates="resident")
    notifications = relationship("Notification", back_populates="resident")