from sqlalchemy import Column, BigInteger, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from ..core.database import Base


class Notification(Base):
    __tablename__ = "notification"
    
    id_notification = Column(BigInteger, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    state_notification = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)
    resident_id = Column(BigInteger, ForeignKey("resident.id_resident"), nullable=False)
    infraction_id = Column(BigInteger, ForeignKey("infraction.id_infraction"), nullable=False)
    
    # Relationships
    resident = relationship("Resident", back_populates="notifications")
    infraction = relationship("Infraction", back_populates="notifications")