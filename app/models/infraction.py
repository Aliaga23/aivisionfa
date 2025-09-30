from sqlalchemy import Column, BigInteger, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class TipoInfraccion(Base):
    __tablename__ = "tipo_infraccion"
    
    id_tipo_infraccion = Column(BigInteger, primary_key=True, index=True)
    cod = Column(String(60), nullable=False)
    description = Column(Text, nullable=False)
    
    # Relationships
    infractions = relationship("Infraction", back_populates="tipo_infraccion")


class Infraction(Base):
    __tablename__ = "infraction"
    
    id_infraction = Column(BigInteger, primary_key=True, index=True)
    description_event = Column(Text)
    datetime_infraction = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(BigInteger, ForeignKey("user.id_user"), nullable=False)
    vehicle_id = Column(BigInteger, ForeignKey("vehicle.id_vehicle"), nullable=False)
    tipo_infraccion_id = Column(BigInteger, ForeignKey("tipo_infraccion.id_tipo_infraccion"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="infractions")
    vehicle = relationship("Vehicle", back_populates="infractions")
    tipo_infraccion = relationship("TipoInfraccion", back_populates="infractions")
    evidences = relationship("Evidence", back_populates="infraction")
    notifications = relationship("Notification", back_populates="infraction")


class Evidence(Base):
    __tablename__ = "evidence"
    
    id_evidence = Column(BigInteger, primary_key=True, index=True)
    description_event = Column(Text)
    url_arch = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    infraction_id = Column(BigInteger, ForeignKey("infraction.id_infraction"), nullable=False)
    camera_id = Column(BigInteger, ForeignKey("camera.id_camera"), nullable=False)
    
    # Relationships
    infraction = relationship("Infraction", back_populates="evidences")
    camera = relationship("Camera", back_populates="evidences")