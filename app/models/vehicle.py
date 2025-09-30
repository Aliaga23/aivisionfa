from sqlalchemy import Column, BigInteger, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class Vehicle(Base):
    __tablename__ = "vehicle"
    
    id_vehicle = Column(BigInteger, primary_key=True, index=True)
    license_plate = Column(String(20), nullable=False, unique=True, index=True)
    brand = Column(String(60))
    model = Column(String(60))
    color = Column(String(40))
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    resident_id = Column(BigInteger, ForeignKey("resident.id_resident"))
    visitor_id = Column(BigInteger, ForeignKey("visitor.id_visitor"))
    
    # Relationships
    resident = relationship("Resident", back_populates="vehicles")
    visitor = relationship("Visitor", back_populates="vehicles")
    infractions = relationship("Infraction", back_populates="vehicle")