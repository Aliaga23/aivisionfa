from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.visitor import VisitorCreate, VisitorUpdate, VisitorResponse, VisitCreate, VisitUpdate, VisitResponse
from ..services.visitor_service import VisitorService
from ..utils.dependencies import get_current_active_user
from ..models.user import User
from typing import List

router = APIRouter(prefix="/visitors", tags=["Visitors"])


@router.post("/", response_model=VisitorResponse, status_code=status.HTTP_201_CREATED)
def create_visitor(
    visitor: VisitorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new visitor"""
    return VisitorService.create_visitor(db, visitor)


@router.get("/", response_model=List[VisitorResponse])
def get_visitors(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all visitors"""
    return VisitorService.get_visitors(db, skip=skip, limit=limit, active_only=active_only)


@router.get("/{visitor_id}", response_model=VisitorResponse)
def get_visitor(
    visitor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a visitor by ID"""
    visitor = VisitorService.get_visitor(db, visitor_id)
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visitor not found"
        )
    return visitor


@router.put("/{visitor_id}", response_model=VisitorResponse)
def update_visitor(
    visitor_id: int,
    visitor_update: VisitorUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a visitor"""
    visitor = VisitorService.update_visitor(db, visitor_id, visitor_update)
    if not visitor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visitor not found"
        )
    return visitor


@router.delete("/{visitor_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_visitor(
    visitor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete (deactivate) a visitor"""
    success = VisitorService.delete_visitor(db, visitor_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visitor not found"
        )


# Visit endpoints
@router.post("/visits", response_model=VisitResponse, status_code=status.HTTP_201_CREATED)
def create_visit(
    visit: VisitCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new visit"""
    return VisitorService.create_visit(db, visit)


@router.get("/visits", response_model=List[VisitResponse])
def get_visits(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all visits"""
    return VisitorService.get_visits(db, skip=skip, limit=limit)


@router.get("/visits/{visit_id}", response_model=VisitResponse)
def get_visit(
    visit_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a visit by ID"""
    visit = VisitorService.get_visit(db, visit_id)
    if not visit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visit not found"
        )
    return visit


@router.put("/visits/{visit_id}", response_model=VisitResponse)
def update_visit(
    visit_id: int,
    visit_update: VisitUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a visit (usually to set departure time)"""
    visit = VisitorService.update_visit(db, visit_id, visit_update)
    if not visit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visit not found"
        )
    return visit