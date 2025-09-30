from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TipoInfraccionBase(BaseModel):
    cod: str
    description: str


class TipoInfraccionCreate(TipoInfraccionBase):
    pass


class TipoInfraccionResponse(TipoInfraccionBase):
    id_tipo_infraccion: int
    
    class Config:
        from_attributes = True


class InfractionBase(BaseModel):
    description_event: Optional[str] = None
    datetime_infraction: datetime
    vehicle_id: int
    tipo_infraccion_id: int


class InfractionCreate(InfractionBase):
    pass


class InfractionUpdate(BaseModel):
    description_event: Optional[str] = None
    datetime_infraction: Optional[datetime] = None


class InfractionResponse(InfractionBase):
    id_infraction: int
    created_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True


class EvidenceBase(BaseModel):
    description_event: Optional[str] = None
    url_arch: str
    infraction_id: int
    camera_id: int


class EvidenceCreate(EvidenceBase):
    pass


class EvidenceResponse(EvidenceBase):
    id_evidence: int
    created_at: datetime
    
    class Config:
        from_attributes = True