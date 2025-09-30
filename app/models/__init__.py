# Import all models here
from .user import User, Rol
from .infraction import Infraction, TipoInfraccion, Evidence
from .resident import Resident
from .visitor import Visitor, Visit
from .vehicle import Vehicle
from .camera import Camera
from .notification import Notification

__all__ = [
    "User", "Rol", "Infraction", "TipoInfraccion", "Evidence", 
    "Resident", "Visitor", "Visit", "Vehicle", "Camera", "Notification"
]