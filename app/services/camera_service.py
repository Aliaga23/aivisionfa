from sqlalchemy.orm import Session
from ..models.camera import Camera
from ..schemas.camera import CameraCreate, CameraUpdate
from typing import List, Optional


class CameraService:
    
    @staticmethod
    def create_camera(db: Session, camera: CameraCreate) -> Camera:
        db_camera = Camera(**camera.dict())
        db.add(db_camera)
        db.commit()
        db.refresh(db_camera)
        return db_camera
    
    @staticmethod
    def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
        return db.query(Camera).filter(Camera.id_camera == camera_id).first()
    
    @staticmethod
    def get_cameras(db: Session, active_only: bool = True) -> List[Camera]:
        query = db.query(Camera)
        if active_only:
            query = query.filter(Camera.active == True)
        return query.all()
    
    @staticmethod
    def update_camera(db: Session, camera_id: int, camera_update: CameraUpdate) -> Optional[Camera]:
        db_camera = db.query(Camera).filter(Camera.id_camera == camera_id).first()
        if not db_camera:
            return None
        
        update_data = camera_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_camera, field, value)
        
        db.commit()
        db.refresh(db_camera)
        return db_camera
    
    @staticmethod
    def delete_camera(db: Session, camera_id: int) -> bool:
        db_camera = db.query(Camera).filter(Camera.id_camera == camera_id).first()
        if not db_camera:
            return False
        
        db_camera.active = False
        db.commit()
        return True