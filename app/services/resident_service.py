from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.resident import Resident
from ..schemas.resident import ResidentCreate, ResidentUpdate
from typing import List, Optional


class ResidentService:
    
    @staticmethod
    def create_resident(db: Session, resident: ResidentCreate) -> Resident:
        # Check if email already exists
        existing_resident = db.query(Resident).filter(Resident.email == resident.email).first()
        if existing_resident:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        db_resident = Resident(**resident.dict())
        db.add(db_resident)
        db.commit()
        db.refresh(db_resident)
        return db_resident
    
    @staticmethod
    def get_resident(db: Session, resident_id: int) -> Optional[Resident]:
        return db.query(Resident).filter(Resident.id_resident == resident_id).first()
    
    @staticmethod
    def get_residents(db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[Resident]:
        query = db.query(Resident)
        if active_only:
            query = query.filter(Resident.active == True)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def update_resident(db: Session, resident_id: int, resident_update: ResidentUpdate) -> Optional[Resident]:
        db_resident = db.query(Resident).filter(Resident.id_resident == resident_id).first()
        if not db_resident:
            return None
        
        update_data = resident_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_resident, field, value)
        
        db.commit()
        db.refresh(db_resident)
        return db_resident
    
    @staticmethod
    def delete_resident(db: Session, resident_id: int) -> bool:
        db_resident = db.query(Resident).filter(Resident.id_resident == resident_id).first()
        if not db_resident:
            return False
        
        db_resident.active = False
        db.commit()
        return True
    
    @staticmethod
    def search_residents_by_name(db: Session, name: str) -> List[Resident]:
        return db.query(Resident).filter(
            Resident.name_resident.ilike(f"%{name}%"),
            Resident.active == True
        ).all()