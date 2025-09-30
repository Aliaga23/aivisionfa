from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class VisitorBase(BaseModel):
    name_visitor: str
    document_id: Optional[str] = None


class VisitorCreate(VisitorBase):
    pass


class VisitorUpdate(BaseModel):
    name_visitor: Optional[str] = None
    document_id: Optional[str] = None
    active: Optional[bool] = None


class VisitorResponse(VisitorBase):
    id_visitor: int
    created_at: datetime
    active: bool
    
    class Config:
        from_attributes = True


class VisitBase(BaseModel):
    entry_datetime: datetime
    departure_datetime: Optional[datetime] = None
    resident_id: int
    visitor_id: int


class VisitCreate(VisitBase):
    pass


class VisitUpdate(BaseModel):
    departure_datetime: Optional[datetime] = None


class VisitResponse(VisitBase):
    id_visit: int
    
    class Config:
        from_attributes = True