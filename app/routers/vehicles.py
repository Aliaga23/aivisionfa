from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.vehicle import VehicleCreate, VehicleUpdate, VehicleResponse
from ..services.vehicle_service import VehicleService
from ..utils.dependencies import get_current_active_user
from ..models.user import User
from typing import List

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])


@router.post("/", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
def create_vehicle(
    vehicle: VehicleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new vehicle"""
    return VehicleService.create_vehicle(db, vehicle)


@router.get("/", response_model=List[VehicleResponse])
def get_vehicles(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all vehicles"""
    return VehicleService.get_vehicles(db, skip=skip, limit=limit, active_only=active_only)


@router.get("/{vehicle_id}", response_model=VehicleResponse)
def get_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a vehicle by ID"""
    vehicle = VehicleService.get_vehicle(db, vehicle_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )
    return vehicle


@router.get("/license-plate/{license_plate}", response_model=VehicleResponse)
def get_vehicle_by_license_plate(
    license_plate: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a vehicle by license plate"""
    vehicle = VehicleService.get_vehicle_by_license_plate(db, license_plate)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )
    return vehicle


@router.get("/resident/{resident_id}", response_model=List[VehicleResponse])
def get_vehicles_by_resident(
    resident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all vehicles belonging to a resident"""
    return VehicleService.get_vehicles_by_resident(db, resident_id)


@router.get("/visitor/{visitor_id}", response_model=List[VehicleResponse])
def get_vehicles_by_visitor(
    visitor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all vehicles belonging to a visitor"""
    return VehicleService.get_vehicles_by_visitor(db, visitor_id)


@router.put("/{vehicle_id}", response_model=VehicleResponse)
def update_vehicle(
    vehicle_id: int,
    vehicle_update: VehicleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a vehicle"""
    vehicle = VehicleService.update_vehicle(db, vehicle_id, vehicle_update)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )
    return vehicle


@router.delete("/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_vehicle(
    vehicle_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete (deactivate) a vehicle"""
    success = VehicleService.delete_vehicle(db, vehicle_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )