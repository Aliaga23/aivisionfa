import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from ..services.vision_service import vision_service
from ..core.config import settings

router = APIRouter(prefix="/vision", tags=["vision"])


@router.on_event("startup")
async def init_detector_on_startup():
    try:
        # Configure service from settings
        vision_service.configure(image_format=settings.image_format)
        # Models are initialized in app.main startup using settings
        print("[vision_router] Vision service configured successfully")
    except Exception as e:
        print(f"[vision_router] Startup configuration failed: {e}")


@router.get("/video_feed")
async def video_feed():
    """Legacy HTTP MJPEG stream - mantenido para compatibilidad"""
    return StreamingResponse(
        vision_service.mjpeg_generator(), 
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@router.get("/stats")
async def stats():
    """Get current statistics"""
    return vision_service.get_stats()


@router.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Upload video file to server"""
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / video.filename
        with open(file_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        print(f"[vision_router] Video uploaded: {video.filename}")
        return {"status": "success", "filename": video.filename}
    except Exception as e:
        print(f"[vision_router] Upload error: {e}")
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})


@router.post("/start_upload")
async def start_from_upload(video: UploadFile = File(...)):
    """Upload and immediately start processing video"""
    try:
        content = await video.read()
        result = vision_service.start_from_bytes(content, suffix=Path(video.filename).suffix)
        if result.get("status") == "error":
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})


@router.post("/start")
async def start_processing(request: Request):
    """Start video processing (HTTP endpoint - legacy)"""
    try:
        data = await request.json()
        source_type = data.get('source_type', 'webcam')
        video_path = data.get('video_path') or data.get('video_filename')
        
        result = vision_service.start(source_type, video_path)
        print(f"[vision_router] Started processing: {result}")
        return result
    except Exception as e:
        print(f"[vision_router] Start error: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/stop")
async def stop_processing():
    """Stop video processing (HTTP endpoint - legacy)"""
    result = vision_service.stop()
    print(f"[vision_router] Stopped processing: {result}")
    return result


@router.websocket("/ws")
async def vision_ws(ws: WebSocket):
    """
    WebSocket principal para streaming de video y control en tiempo real.
    
    Protocolo:
    - Envía frames como bytes binarios (JPEG o WebP)
    - Envía stats/status como JSON
    - Recibe comandos como JSON: {"action": "start/stop/stats"}
    """
    await ws.accept()
    print("[vision_router] WebSocket connected")
    
    try:
        # Enviar saludo inicial con configuración
        await ws.send_json({
            "type": "hello",
            "image_format": settings.image_format,
            "message": "WebSocket connected successfully"
        })
        
        # Timers para throttling
        last_stats = 0.0
        last_frame = 0.0
        frame_interval = 0.033   # ~30 FPS (ajustable según GPU)
        stats_interval = 0.5     # 2 Hz para estadísticas
        
        while True:
            # Procesar mensajes de control sin bloquear
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                try:
                    msg = json.loads(text)
                    action = msg.get("action")
                    
                    if action == "start":
                        source_type = msg.get("source_type", "webcam")
                        video_path = msg.get("video_path") or msg.get("video_filename")
                        
                        print(f"[vision_router] WS start command: {source_type}, {video_path}")
                        result = vision_service.start(source_type, video_path)
                        await ws.send_json({"type": "status", **result})
                        
                    elif action == "stop":
                        print("[vision_router] WS stop command")
                        result = vision_service.stop()
                        await ws.send_json({"type": "status", **result})
                        
                    elif action == "stats":
                        # Forzar envío inmediato de stats
                        await ws.send_json({"type": "stats", **vision_service.get_stats()})
                        
                    elif action == "config":
                        new_fmt = msg.get("image_format")
                        if new_fmt:
                            vision_service.configure(image_format=new_fmt)
                            await ws.send_json({"type": "config", "ok": True, "format": new_fmt})
                            
                    elif action == "ping":
                        # Heartbeat para mantener conexión viva
                        await ws.send_json({"type": "pong"})
                        
                    else:
                        print(f"[vision_router] Unknown action: {action}")
                        
                except json.JSONDecodeError as e:
                    print(f"[vision_router] Invalid JSON: {e}")
                    await ws.send_json({"type": "error", "message": "Invalid JSON format"})
                    
            except asyncio.TimeoutError:
                # No hay mensajes pendientes, continuar con el loop
                pass

            now = asyncio.get_event_loop().time()

            # Enviar frames de video con throttling
            if now - last_frame >= frame_interval:
                frame_bytes = vision_service.get_encoded_frame()
                if frame_bytes is not None:
                    try:
                        await ws.send_bytes(frame_bytes)
                        last_frame = now
                    except Exception as e:
                        print(f"[vision_router] Error sending frame: {e}")
                        break

            # Enviar estadísticas con throttling
            if now - last_stats >= stats_interval:
                try:
                    stats_data = vision_service.get_stats()
                    await ws.send_json({"type": "stats", **stats_data})
                    last_stats = now
                except Exception as e:
                    print(f"[vision_router] Error sending stats: {e}")

            # Sleep breve para no saturar CPU
            await asyncio.sleep(0.005)
            
    except WebSocketDisconnect:
        print("[vision_router] WebSocket disconnected by client")
        # Detener procesamiento si estaba activo
        vision_service.stop()
        
    except Exception as e:
        print(f"[vision_router] WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vision",
        "processing": vision_service.processing_active,
        "detector_loaded": vision_service.detector is not None
    }