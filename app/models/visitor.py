from sqlalchemy import Column, BigInteger, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class Visitor(Base):
    __tablename__ = "visitor"
    
    id_visitor = Column(BigInteger, primary_key=True, index=True)
    name_visitor = Column(String(150), nullable=False)
    document_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
    
    # Relationships
    visits = relationship("Visit", back_populates="visitor")
    vehicles = relationship("Vehicle", back_populates="visitor")


class Visit(Base):
    __tablename__ = "visit"
    
    id_visit = Column(BigInteger, primary_key=True, index=True)
    entry_datetime = Column(DateTime, nullable=False)
    departure_datetime = Column(DateTime)
    resident_id = Column(BigInteger, ForeignKey("resident.id_resident"), nullable=False)
    visitor_id = Column(BigInteger, ForeignKey("visitor.id_visitor"), nullable=False)
    
    # Relationships
    resident = relationship("Resident", back_populates="visits")
    visitor = relationship("Visitor", back_populates="visits")