from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.infraction import Infraction, TipoInfraccion, Evidence
from ..schemas.infraction import InfractionCreate, InfractionUpdate, TipoInfraccionCreate, EvidenceCreate
from typing import List, Optional


class InfractionService:
    
    @staticmethod
    def create_infraction(db: Session, infraction: InfractionCreate, user_id: int) -> Infraction:
        db_infraction = Infraction(
            **infraction.dict(),
            user_id=user_id
        )
        db.add(db_infraction)
        db.commit()
        db.refresh(db_infraction)
        return db_infraction
    
    @staticmethod
    def get_infraction(db: Session, infraction_id: int) -> Optional[Infraction]:
        return db.query(Infraction).filter(Infraction.id_infraction == infraction_id).first()
    
    @staticmethod
    def get_infractions(db: Session, skip: int = 0, limit: int = 100) -> List[Infraction]:
        return db.query(Infraction).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_infractions_by_vehicle(db: Session, vehicle_id: int) -> List[Infraction]:
        return db.query(Infraction).filter(Infraction.vehicle_id == vehicle_id).all()
    
    @staticmethod
    def get_infractions_by_user(db: Session, user_id: int) -> List[Infraction]:
        return db.query(Infraction).filter(Infraction.user_id == user_id).all()
    
    @staticmethod
    def update_infraction(db: Session, infraction_id: int, infraction_update: InfractionUpdate) -> Optional[Infraction]:
        db_infraction = db.query(Infraction).filter(Infraction.id_infraction == infraction_id).first()
        if not db_infraction:
            return None
        
        update_data = infraction_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_infraction, field, value)
        
        db.commit()
        db.refresh(db_infraction)
        return db_infraction
    
    # Tipo Infraccion methods
    @staticmethod
    def create_tipo_infraccion(db: Session, tipo_infraccion: TipoInfraccionCreate) -> TipoInfraccion:
        # Check if code already exists
        existing_tipo = db.query(TipoInfraccion).filter(TipoInfraccion.cod == tipo_infraccion.cod).first()
        if existing_tipo:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Infraction type code already exists"
            )
        
        db_tipo = TipoInfraccion(**tipo_infraccion.dict())
        db.add(db_tipo)
        db.commit()
        db.refresh(db_tipo)
        return db_tipo
    
    @staticmethod
    def get_tipos_infraccion(db: Session) -> List[TipoInfraccion]:
        return db.query(TipoInfraccion).all()
    
    @staticmethod
    def get_tipo_infraccion(db: Session, tipo_id: int) -> Optional[TipoInfraccion]:
        return db.query(TipoInfraccion).filter(TipoInfraccion.id_tipo_infraccion == tipo_id).first()
    
    # Evidence methods
    @staticmethod
    def create_evidence(db: Session, evidence: EvidenceCreate) -> Evidence:
        db_evidence = Evidence(**evidence.dict())
        db.add(db_evidence)
        db.commit()
        db.refresh(db_evidence)
        return db_evidence
    
    @staticmethod
    def get_evidence_by_infraction(db: Session, infraction_id: int) -> List[Evidence]:
        return db.query(Evidence).filter(Evidence.infraction_id == infraction_id).all()
    
    @staticmethod
    def get_evidence(db: Session, evidence_id: int) -> Optional[Evidence]:
        return db.query(Evidence).filter(Evidence.id_evidence == evidence_id).first()