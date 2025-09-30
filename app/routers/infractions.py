from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.infraction import (
    InfractionCreate, InfractionUpdate, InfractionResponse,
    TipoInfraccionCreate, TipoInfraccionResponse,
    EvidenceCreate, EvidenceResponse
)
from ..services.infraction_service import InfractionService
from ..utils.dependencies import get_current_active_user
from ..models.user import User
from typing import List

router = APIRouter(prefix="/infractions", tags=["Infractions"])


# Infraction endpoints
@router.post("/", response_model=InfractionResponse, status_code=status.HTTP_201_CREATED)
def create_infraction(
    infraction: InfractionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new infraction"""
    return InfractionService.create_infraction(db, infraction, current_user.id_user)


@router.get("/", response_model=List[InfractionResponse])
def get_infractions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all infractions"""
    return InfractionService.get_infractions(db, skip=skip, limit=limit)


@router.get("/{infraction_id}", response_model=InfractionResponse)
def get_infraction(
    infraction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get an infraction by ID"""
    infraction = InfractionService.get_infraction(db, infraction_id)
    if not infraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Infraction not found"
        )
    return infraction


@router.get("/vehicle/{vehicle_id}", response_model=List[InfractionResponse])
def get_infractions_by_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all infractions for a specific vehicle"""
    return InfractionService.get_infractions_by_vehicle(db, vehicle_id)


@router.get("/user/{user_id}", response_model=List[InfractionResponse])
def get_infractions_by_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all infractions created by a specific user"""
    return InfractionService.get_infractions_by_user(db, user_id)


@router.put("/{infraction_id}", response_model=InfractionResponse)
def update_infraction(
    infraction_id: int,
    infraction_update: InfractionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update an infraction"""
    infraction = InfractionService.update_infraction(db, infraction_id, infraction_update)
    if not infraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Infraction not found"
        )
    return infraction


# Tipo Infraccion endpoints
@router.post("/types", response_model=TipoInfraccionResponse, status_code=status.HTTP_201_CREATED)
def create_tipo_infraccion(
    tipo_infraccion: TipoInfraccionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new infraction type"""
    return InfractionService.create_tipo_infraccion(db, tipo_infraccion)


@router.get("/types", response_model=List[TipoInfraccionResponse])
def get_tipos_infraccion(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all infraction types"""
    return InfractionService.get_tipos_infraccion(db)


@router.get("/types/{tipo_id}", response_model=TipoInfraccionResponse)
def get_tipo_infraccion(
    tipo_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get an infraction type by ID"""
    tipo = InfractionService.get_tipo_infraccion(db, tipo_id)
    if not tipo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Infraction type not found"
        )
    return tipo


# Evidence endpoints
@router.post("/evidence", response_model=EvidenceResponse, status_code=status.HTTP_201_CREATED)
def create_evidence(
    evidence: EvidenceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create new evidence for an infraction"""
    return InfractionService.create_evidence(db, evidence)


@router.get("/{infraction_id}/evidence", response_model=List[EvidenceResponse])
def get_evidence_by_infraction(
    infraction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all evidence for a specific infraction"""
    return InfractionService.get_evidence_by_infraction(db, infraction_id)


@router.get("/evidence/{evidence_id}", response_model=EvidenceResponse)
def get_evidence(
    evidence_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get evidence by ID"""
    evidence = InfractionService.get_evidence(db, evidence_id)
    if not evidence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found"
        )
    return evidence