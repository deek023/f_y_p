"""
KalaKriti — FastAPI Backend Server
Implements the exact JSON API Contract:
  - GET /health
  - POST /stylize

Strict Authenticity Rule:
  If genuine trained weights are not found, /health reports modelReady=false,
  and /stylize returns HTTP 503 Service Unavailable with an explanation.
  NO fake, filter-based, or placeholder images are ever returned.
"""

import base64
import io
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image


# Import TransformerNet if torch is installed
try:
    import torch
    import torchvision.transforms.functional as TF
    from backend.models.transformer_net import TransformerNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

app = FastAPI(
    title="KalaKriti Neural Style Transfer API",
    description="Real-Time Multi-Style Feed-Forward CNN for Indian Traditional Art",
    version="1.0.0",
)

# CORS Middleware to support React frontend across local and ngrok tunnels
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration & Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
WARLI_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "kalakriti_cin_warli.pth")
WARLI_PILOT_PATH = os.path.join(CHECKPOINT_DIR, "kalakriti_cin_warli_pilot.pth")

# Global Model State
device = None
model: Optional[object] = None
model_ready = False
loaded_styles = []
active_checkpoint_name = ""

STYLE_MAP = {
    "warli": 0,
    "madhubani": 1,
    "kalamkari": 2,
    "gond": 3,
    "pattachitra": 4,
}


def load_model_if_available():
    """Attempts to load genuine trained weights. Never fabricates ready status."""
    global device, model, model_ready, loaded_styles, active_checkpoint_name

    if not TORCH_AVAILABLE:
        model_ready = False
        loaded_styles = []
        active_checkpoint_name = ""
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine which checkpoint is available (full or pilot)
    target_path = None
    if os.path.exists(WARLI_CHECKPOINT_PATH):
        target_path = WARLI_CHECKPOINT_PATH
    elif os.path.exists(WARLI_PILOT_PATH):
        target_path = WARLI_PILOT_PATH

    if target_path:
        try:
            print(f"Loading weights from {target_path} on {device}...")
            net = TransformerNet(num_styles=len(STYLE_MAP), base_channels=32)
            state_dict = torch.load(target_path, map_location=device)
            net.load_state_dict(state_dict)
            net.to(device)
            net.eval()
            model = net
            model_ready = True
            loaded_styles = ["warli"]
            active_checkpoint_name = os.path.basename(target_path)
            print(f"Successfully loaded Warli model weights ({active_checkpoint_name}).")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            model = None
            model_ready = False
            loaded_styles = []
            active_checkpoint_name = ""
    else:
        model = None
        model_ready = False
        loaded_styles = []
        active_checkpoint_name = ""
        print(f"No checkpoint found at {WARLI_CHECKPOINT_PATH} or {WARLI_PILOT_PATH}. Model marked as NOT ready.")


# Trigger model loading at startup
@app.on_event("startup")
def on_startup():
    load_model_if_available()


# Request / Response Schemas
class StylizeRequest(BaseModel):
    contentImageBase64: str = Field(..., description="Base64 encoded JPEG/PNG content image")
    styleId: str = Field(..., description="Art style identifier (e.g. 'warli')")
    styleStrength: float = Field(0.85, ge=0.0, le=1.0, description="Style blending ratio [0.0 - 1.0]")
    resolution: int = Field(512, ge=128, le=2048, description="Target maximum dimension")


class StylizeResponse(BaseModel):
    stylizedImageBase64: str
    inferenceTimeMs: float
    styleId: str
    resolution: list[int]


class HealthResponse(BaseModel):
    status: str
    engine: str
    device: str
    modelReady: bool
    loadedStyles: list[str]
    message: Optional[str] = None
    version: str


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Checks backend health and reports model readiness.
    Accurately indicates whether trained weights are loaded.
    """
    if model_ready and model is not None:
        return HealthResponse(
            status="healthy",
            engine="Feed-Forward Multi-Style CNN (KalaKriti)",
            device=str(device),
            modelReady=True,
            loadedStyles=loaded_styles,
            message=f"Model weights loaded and operational ({active_checkpoint_name}).",
            version="1.0.0",
        )
    else:
        return HealthResponse(
            status="model_not_ready",
            engine="Feed-Forward Multi-Style CNN (KalaKriti)",
            device=str(device) if device else "unknown",
            modelReady=False,
            loadedStyles=[],
            message=(
                "PyTorch scaffold is initialized, but trained model checkpoint "
                f"'{os.path.basename(WARLI_CHECKPOINT_PATH)}' is not present. "
                "Complete Stage 5 training on Google Colab and place the weights file "
                "in backend/checkpoints/ to enable genuine stylization."
            ),
            version="1.0.0",
        )


@app.post("/stylize", response_model=StylizeResponse)
def stylize_image(req: StylizeRequest):
    """
    Inference endpoint: executes a single forward pass on the uploaded content image.
    Strictly refuses to serve placeholder or mock artwork if weights are unavailable.
    """
    # 1. Model Readiness Check
    if not model_ready or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model weights for style '{req.styleId}' are not available on the server. "
                "KalaKriti adheres to strict authenticity and does not return fake or filtered images. "
                "Train the generator using train_colab.py and place kalakriti_cin_warli.pth in backend/checkpoints/."
            ),
        )

    # 2. Style Identifier Validation
    style_key = req.styleId.lower().strip()
    if style_key not in STYLE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown style '{req.styleId}'. Supported styles: {list(STYLE_MAP.keys())}",
        )

    if style_key not in loaded_styles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Style '{req.styleId}' is defined in the architecture, but weights have not been trained yet. "
                f"Currently loaded and operational styles: {loaded_styles}"
            ),
        )

    style_idx = STYLE_MAP[style_key]

    # 3. Decode Base64 Content Image
    try:
        header = ""
        raw_b64 = req.contentImageBase64
        if "," in raw_b64:
            header, raw_b64 = raw_b64.split(",", 1)

        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 image data: {str(e)}",
        )

    # 4. Aspect-Ratio Preserving Resize
    orig_w, orig_h = pil_img.size
    max_dim = req.resolution
    scale = min(max_dim / orig_w, max_dim / orig_h, 1.0)
    target_w = int(orig_w * scale)
    target_h = int(orig_h * scale)

    # Ensure dimensions are divisible by 4 (generator downsamples 2x then 2x)
    target_w = (target_w // 4) * 4
    target_h = (target_h // 4) * 4
    target_w = max(target_w, 64)
    target_h = max(target_h, 64)

    resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 5. Tensor Conversion & Single Forward Pass
    try:
        start_time = time.perf_counter()

        # Convert to tensor in range [0, 255]
        content_tensor = TF.to_tensor(resized_img).unsqueeze(0).to(device) * 255.0
        style_tensor = torch.tensor([style_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            stylized_tensor = model(content_tensor, style_tensor)

            # Style strength blending: linear interpolation between content and stylized output
            if req.styleStrength < 1.0:
                alpha = max(0.0, min(1.0, req.styleStrength))
                stylized_tensor = alpha * stylized_tensor + (1.0 - alpha) * content_tensor

            # Clamp to valid uint8 range
            stylized_tensor = torch.clamp(stylized_tensor, 0.0, 255.0).byte()

        # Convert back to PIL
        out_np = stylized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_pil = Image.fromarray(out_np)

        # Encode to Base64 JPEG
        buf = io.BytesIO()
        out_pil.save(buf, format="JPEG", quality=92)
        out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return StylizeResponse(
            stylizedImageBase64=out_b64,
            inferenceTimeMs=round(elapsed_ms, 2),
            styleId=style_key,
            resolution=[target_w, target_h],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
