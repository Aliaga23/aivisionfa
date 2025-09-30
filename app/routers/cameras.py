from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.camera import CameraCreate, CameraUpdate, CameraResponse
from ..services.camera_service import CameraService
from ..utils.dependencies import get_current_active_user
from ..models.user import User
from typing import List

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(
    camera: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new camera"""
    return CameraService.create_camera(db, camera)


@router.get("/", response_model=List[CameraResponse])
def get_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all cameras"""
    return CameraService.get_cameras(db)


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a camera by ID"""
    camera = CameraService.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    return camera


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: int,
    camera_update: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a camera"""
    camera = CameraService.update_camera(db, camera_id, camera_update)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete (deactivate) a camera"""
    success = CameraService.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found"
        )