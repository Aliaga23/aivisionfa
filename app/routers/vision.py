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
    except Exception as e:
        print(f"[vision] Startup configuration failed: {e}")


@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(vision_service.mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')


@router.get("/stats")
async def stats():
    return vision_service.get_stats()


@router.post("/start_upload")
async def start_from_upload(video: UploadFile = File(...)):
    content = await video.read()
    result = vision_service.start_from_bytes(content, suffix=Path(video.filename).suffix)
    if result.get("status") == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/start")
async def start_processing(request: Request):
    data = await request.json()
    source_type = data.get('source_type', 'webcam')
    # Prefer explicit 'video_path'; keep legacy 'video_filename' for compatibility
    video_path = data.get('video_path') or data.get('video_filename')
    return vision_service.start(source_type, video_path)


@router.post("/stop")
async def stop_processing():
    return vision_service.stop()


@router.websocket("/ws")
async def vision_ws(ws: WebSocket):
    await ws.accept()
    try:
        # Initial hello/config
        await ws.send_json({
            "type": "hello",
            "image_format": settings.image_format,
        })
        last_stats = 0.0
        while True:
            # Try to read control message without blocking
            try:
                text = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                try:
                    msg = json.loads(text)
                except Exception:
                    msg = {"action": "noop"}
                action = msg.get("action")
                if action == "start":
                    source_type = msg.get("source_type", "webcam")
                    # Accept both 'video_path' and legacy 'video_filename'
                    video_path = msg.get("video_path") or msg.get("video_filename")
                    result = vision_service.start(source_type, video_path)
                    await ws.send_json({"type": "status", **result})
                elif action == "stop":
                    result = vision_service.stop()
                    await ws.send_json({"type": "status", **result})
                elif action == "stats":
                    await ws.send_json({"type": "stats", **vision_service.get_stats()})
                elif action == "config":
                    new_fmt = msg.get("image_format")
                    if new_fmt:
                        vision_service.configure(image_format=new_fmt)
                        await ws.send_json({"type": "config", "ok": True})
                # ignore unknown actions
            except asyncio.TimeoutError:
                pass

            # Push latest frame if available
            frame_bytes = vision_service.get_encoded_frame()
            if frame_bytes is not None:
                await ws.send_bytes(frame_bytes)
            # Throttle stats to ~5 Hz
            now = asyncio.get_event_loop().time()
            if now - last_stats > 0.2:
                await ws.send_json({"type": "stats", **vision_service.get_stats()})
                last_stats = now
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        finally:
            await ws.close()
