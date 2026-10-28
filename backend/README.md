# KalaKriti — Backend Engineering Guide

This directory contains the Python/PyTorch modules and FastAPI server for **KalaKriti — Real-Time Multi-Style Neural Style Transfer for Visual Domain Adaptation and Indian Traditional Art**.

---

## 1. Directory Structure

```text
backend/
├── models/
│   ├── __init__.py
│   └── transformer_net.py     # Generator with Conditional Instance Normalization (CIN)
├── checkpoints/              # Directory for trained weights (kalakriti_cin_warli.pth)
├── server.py                  # FastAPI server implementing GET /health and POST /stylize
├── train_colab.py             # Reproducible training pipeline for Google Colab GPU
├── test_model.py              # Technical unit tests (tensor shape, CIN, gradient flow)
├── test_api.py                # API contract unit tests (unready state, 503 error, validation)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 2. API Contract

The server implements the exact JSON API contract:

### `GET /health`
- **Model Ready Response (HTTP 200):**
  ```json
  {
    "status": "healthy",
    "engine": "Feed-Forward Multi-Style CNN (KalaKriti)",
    "device": "cuda:0",
    "modelReady": true,
    "loadedStyles": ["warli"],
    "version": "1.0.0"
  }
  ```
- **Model Pending Response (HTTP 200):**
  ```json
  {
    "status": "model_not_ready",
    "engine": "Feed-Forward Multi-Style CNN (KalaKriti)",
    "device": "cpu",
    "modelReady": false,
    "loadedStyles": [],
    "message": "Model weights checkpoint not found...",
    "version": "1.0.0"
  }
  ```

### `POST /stylize`
- **Request Body (JSON):**
  ```json
  {
    "contentImageBase64": "data:image/jpeg;base64,...",
    "styleId": "warli",
    "styleStrength": 0.85,
    "resolution": 512
  }
  ```
- **Authenticity Guard:**
  If weights are not found, returns **HTTP 503 Service Unavailable** with an explicit error message. It **refuses** to return simulated or filtered images.

---

## 3. Running the FastAPI Server Locally

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run the FastAPI server
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

---

## 4. Google Colab GPU Workflow (Stage 5A Pilot)

Open `KalaKriti_Warli_Pilot.ipynb` directly in Google Colab (with a free T4 GPU enabled).
The notebook guides you through:
1. **GPU Environment Verification:** Checks CUDA device name and allocated VRAM.
2. **Kaggle Dataset Acquisition:** Downloads `ajg117/indian-paintings-dataset` (which contains 8 labeled classes including authentic **Warli**, **Madhubani**, and **Gond** paintings).
3. **Contact Sheet Review:** Renders a 12-image visual thumbnail grid for human inspection before copying verified Warli art to `data/warli_train`.
4. **Content & Unseen Validation Photographs:** Downloads a 50-image MS-COCO subset and 4 distinct benchmark photos (`portrait.jpg`, `monument.jpg`, `landscape.jpg`, `animal.jpg`).
5. **Pre-Flight Validation:** Executes `backend/validate_dataset.py` to confirm image integrity, dimensions, and visual variance.
6. **Reproducible Pilot Run:** Runs `backend/run_colab_pilot.py` (50 content photos, 10–20 Warli references, 2 epochs, batch size 4, seed 42) and saves `kalakriti_cin_warli_pilot.pth`.
7. **Visual Evaluation:** Displays 3-panel comparison plates (`[Original Photo | Warli Reference | KalaKriti Stylized]`).
8. **Live FastAPI Backend:** Starts the server and provides a public ngrok tunnel URL to paste into the React frontend.
