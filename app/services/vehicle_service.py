from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from ..models.vehicle import Vehicle
from ..schemas.vehicle import VehicleCreate, VehicleUpdate
from typing import List, Optional


class VehicleService:
    
    @staticmethod
    def create_vehicle(db: Session, vehicle: VehicleCreate) -> Vehicle:
        # Check if license plate already exists
        existing_vehicle = db.query(Vehicle).filter(Vehicle.license_plate == vehicle.license_plate).first()
        if existing_vehicle:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="License plate already registered"
            )
        
        # Validate that either resident_id or visitor_id is provided, but not both
        if vehicle.resident_id and vehicle.visitor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vehicle cannot belong to both resident and visitor"
            )
        
        if not vehicle.resident_id and not vehicle.visitor_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vehicle must belong to either a resident or visitor"
            )
        
        db_vehicle = Vehicle(**vehicle.dict())
        db.add(db_vehicle)
        db.commit()
        db.refresh(db_vehicle)
        return db_vehicle
    
    @staticmethod
    def get_vehicle(db: Session, vehicle_id: int) -> Optional[Vehicle]:
        return db.query(Vehicle).filter(Vehicle.id_vehicle == vehicle_id).first()
    
    @staticmethod
    def get_vehicle_by_license_plate(db: Session, license_plate: str) -> Optional[Vehicle]:
        return db.query(Vehicle).filter(
            Vehicle.license_plate == license_plate,
            Vehicle.active == True
        ).first()
    
    @staticmethod
    def get_vehicles(db: Session, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[Vehicle]:
        query = db.query(Vehicle)
        if active_only:
            query = query.filter(Vehicle.active == True)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_vehicles_by_resident(db: Session, resident_id: int) -> List[Vehicle]:
        return db.query(Vehicle).filter(
            Vehicle.resident_id == resident_id,
            Vehicle.active == True
        ).all()
    
    @staticmethod
    def get_vehicles_by_visitor(db: Session, visitor_id: int) -> List[Vehicle]:
        return db.query(Vehicle).filter(
            Vehicle.visitor_id == visitor_id,
            Vehicle.active == True
        ).all()
    
    @staticmethod
    def update_vehicle(db: Session, vehicle_id: int, vehicle_update: VehicleUpdate) -> Optional[Vehicle]:
        db_vehicle = db.query(Vehicle).filter(Vehicle.id_vehicle == vehicle_id).first()
        if not db_vehicle:
            return None
        
        update_data = vehicle_update.dict(exclude_unset=True)
        
        # Validate license plate uniqueness if being updated
        if "license_plate" in update_data:
            existing_vehicle = db.query(Vehicle).filter(
                Vehicle.license_plate == update_data["license_plate"],
                Vehicle.id_vehicle != vehicle_id
            ).first()
            if existing_vehicle:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="License plate already registered"
                )
        
        for field, value in update_data.items():
            setattr(db_vehicle, field, value)
        
        db.commit()
        db.refresh(db_vehicle)
        return db_vehicle
    
    @staticmethod
    def delete_vehicle(db: Session, vehicle_id: int) -> bool:
        db_vehicle = db.query(Vehicle).filter(Vehicle.id_vehicle == vehicle_id).first()
        if not db_vehicle:
            return False
        
        db_vehicle.active = False
        db.commit()
        return True