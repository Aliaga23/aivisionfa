from sqlalchemy import Column, BigInteger, String, Boolean
from sqlalchemy.orm import relationship
from ..core.database import Base


class Camera(Base):
    __tablename__ = "camera"
    
    id_camera = Column(BigInteger, primary_key=True, index=True)
    name_camera = Column(String(100), nullable=False)
    location_camera = Column(String(200))
    active = Column(Boolean, default=True)
    
    # Relationships
    evidences = relationship("Evidence", back_populates="camera")