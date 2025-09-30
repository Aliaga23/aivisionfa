from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.visitor import Visitor, Visit
from ..schemas.visitor import VisitorCreate, VisitorUpdate, VisitCreate, VisitUpdate
from typing import List, Optional


class VisitorService:
    
    @staticmethod
    def create_visitor(db: Session, visitor: VisitorCreate) -> Visitor:
        db_visitor = Visitor(**visitor.dict())
        db.add(db_visitor)
        db.commit()
        db.refresh(db_visitor)
        return db_visitor
    
    @staticmethod
    def get_visitor(db: Session, visitor_id: int) -> Optional[Visitor]:
        return db.query(Visitor).filter(Visitor.id_visitor == visitor_id).first()
    
    @staticmethod
    def get_visitors(db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[Visitor]:
        query = db.query(Visitor)
        if active_only:
            query = query.filter(Visitor.active == True)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def update_visitor(db: Session, visitor_id: int, visitor_update: VisitorUpdate) -> Optional[Visitor]:
        db_visitor = db.query(Visitor).filter(Visitor.id_visitor == visitor_id).first()
        if not db_visitor:
            return None
        
        update_data = visitor_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_visitor, field, value)
        
        db.commit()
        db.refresh(db_visitor)
        return db_visitor
    
    @staticmethod
    def delete_visitor(db: Session, visitor_id: int) -> bool:
        db_visitor = db.query(Visitor).filter(Visitor.id_visitor == visitor_id).first()
        if not db_visitor:
            return False
        
        db_visitor.active = False
        db.commit()
        return True
    
    # Visit methods
    @staticmethod
    def create_visit(db: Session, visit: VisitCreate) -> Visit:
        # Verify visitor and resident exist
        visitor = db.query(Visitor).filter(Visitor.id_visitor == visit.visitor_id).first()
        if not visitor:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Visitor not found"
            )
        
        db_visit = Visit(**visit.dict())
        db.add(db_visit)
        db.commit()
        db.refresh(db_visit)
        return db_visit
    
    @staticmethod
    def get_visit(db: Session, visit_id: int) -> Optional[Visit]:
        return db.query(Visit).filter(Visit.id_visit == visit_id).first()
    
    @staticmethod
    def get_visits(db: Session, skip: int = 0, limit: int = 100) -> List[Visit]:
        return db.query(Visit).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_visits_by_visitor(db: Session, visitor_id: int) -> List[Visit]:
        return db.query(Visit).filter(Visit.visitor_id == visitor_id).all()
    
    @staticmethod
    def get_visits_by_resident(db: Session, resident_id: int) -> List[Visit]:
        return db.query(Visit).filter(Visit.resident_id == resident_id).all()
    
    @staticmethod
    def update_visit(db: Session, visit_id: int, visit_update: VisitUpdate) -> Optional[Visit]:
        db_visit = db.query(Visit).filter(Visit.id_visit == visit_id).first()
        if not db_visit:
            return None
        
        update_data = visit_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_visit, field, value)
        
        db.commit()
        db.refresh(db_visit)
        return db_visit