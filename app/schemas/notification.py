from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class NotificationBase(BaseModel):
    title: str
    state_notification: Optional[str] = None
    resident_id: int
    infraction_id: int


class NotificationCreate(NotificationBase):
    pass


class NotificationUpdate(BaseModel):
    title: Optional[str] = None
    state_notification: Optional[str] = None


class NotificationResponse(NotificationBase):
    id_notification: int
    created_at: datetime
    
    class Config:
        from_attributes = True