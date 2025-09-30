import os
# Force CPU usage for all ML frameworks before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
from .core.database import engine, get_db
from .models import user, infraction, resident, visitor, vehicle, camera, notification
from .routers import auth, residents, vehicles, infractions
from .core.config import settings

# Create database tables
user.Base.metadata.create_all(bind=engine)
infraction.Base.metadata.create_all(bind=engine)
resident.Base.metadata.create_all(bind=engine)
visitor.Base.metadata.create_all(bind=engine)
vehicle.Base.metadata.create_all(bind=engine)
camera.Base.metadata.create_all(bind=engine)
notification.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Vision Artificial API",
    description="API for Vision Artificial surveillance system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers import visitors, cameras
from .routers import vision

app.include_router(auth.router, prefix="/api/v1")
app.include_router(residents.router, prefix="/api/v1")
app.include_router(vehicles.router, prefix="/api/v1")
app.include_router(infractions.router, prefix="/api/v1")
app.include_router(visitors.router, prefix="/api/v1")
app.include_router(cameras.router, prefix="/api/v1")
app.include_router(vision.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "Vision Artificial API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Startup event to create initial data
@app.on_event("startup")
async def startup_event():
    """Create initial roles if they don't exist"""
    from .services.auth_service import AuthService
    from .core.database import SessionLocal
    # Initialize vision models
    try:
        from .services.vision_service import vision_service
        # Get the project root (parent of 'app' directory)
        # __file__ is app/main.py, so parent is 'app', parent.parent is project root
        project_root = Path(__file__).resolve().parent.parent
        expected_zone_path = project_root / "models" / "best.pt"
        
        if not expected_zone_path.exists():
            raise FileNotFoundError(f"Required zone model not found: {expected_zone_path}")

        # Use absolute paths for all models
        vehicle_model_path = str(project_root / "models" / "yolov8n.pt")
        orientation_model_path = str(project_root / "models" / "car_orientation_model.pth")
        
        print(f"[startup] Loading models from:")
        print(f"  - Zone model: {expected_zone_path}")
        print(f"  - Vehicle model: {vehicle_model_path}")
        print(f"  - Orientation model: {orientation_model_path}")
        vision_service.init_models(str(expected_zone_path), vehicle_model_path, orientation_model_path)
        print(f"[startup] Vision models initialized from {expected_zone_path}")
    except Exception as e:
        print(f"[startup] Vision models init failed: {e}")
    
    db = SessionLocal()
    try:
        # Check if roles exist, if not create default ones
        roles = AuthService.get_roles(db)
        if not roles:
            # Create default roles
            AuthService.create_role(db, "Admin")
            AuthService.create_role(db, "Security")
            AuthService.create_role(db, "Operator")
            print("Default roles created successfully")
    except Exception as e:
        print(f"Error creating default roles: {e}")
    finally:
        db.close()