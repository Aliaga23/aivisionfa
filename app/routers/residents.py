from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.resident import ResidentCreate, ResidentUpdate, ResidentResponse
from ..services.resident_service import ResidentService
from ..utils.dependencies import get_current_active_user
from ..models.user import User
from typing import List

router = APIRouter(prefix="/residents", tags=["Residents"])


@router.post("/", response_model=ResidentResponse, status_code=status.HTTP_201_CREATED)
def create_resident(
    resident: ResidentCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new resident"""
    return ResidentService.create_resident(db, resident)


@router.get("/", response_model=List[ResidentResponse])
def get_residents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all residents"""
    return ResidentService.get_residents(db, skip=skip, limit=limit, active_only=active_only)


@router.get("/{resident_id}", response_model=ResidentResponse)
def get_resident(
    resident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a resident by ID"""
    resident = ResidentService.get_resident(db, resident_id)
    if not resident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resident not found"
        )
    return resident


@router.put("/{resident_id}", response_model=ResidentResponse)
def update_resident(
    resident_id: int,
    resident_update: ResidentUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a resident"""
    resident = ResidentService.update_resident(db, resident_id, resident_update)
    if not resident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resident not found"
        )
    return resident


@router.delete("/{resident_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resident(
    resident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete (deactivate) a resident"""
    success = ResidentService.delete_resident(db, resident_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resident not found"
        )


@router.get("/search/{name}", response_model=List[ResidentResponse])
def search_residents_by_name(
    name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Search residents by name"""
    return ResidentService.search_residents_by_name(db, name)