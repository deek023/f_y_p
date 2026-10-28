# KalaKriti — Google Colab Warli Pilot Setup Guide

This package contains the complete, self-contained files to execute the **KalaKriti Stage 5A Warli Pilot** on a Google Colab GPU instance.

---

## Package Contents

```text
kalakriti_colab_package/
├── KalaKriti_Warli_Pilot.ipynb   # Master executable Colab notebook (11 cells)
├── README_COLAB.md               # This setup and execution guide
└── backend/
    ├── models/
    │   ├── __init__.py
    │   └── transformer_net.py    # TransformerNet with Conditional Instance Normalization (CIN)
    ├── checkpoints/             # Directory for saved checkpoints
    ├── train_colab.py            # Training pipeline with domain-averaged Gram loss
    ├── run_colab_pilot.py        # Pilot orchestration runner (seed 42, 2 epochs, batch size 4)
    ├── validate_dataset.py       # Pre-flight dataset integrity & variance validator
    ├── server.py                 # FastAPI server with /health and /stylize endpoints
    ├── test_api.py               # Unit tests verifying API contracts and refusal of fake outputs
    ├── test_model.py             # Unit tests verifying CIN, tensor shapes, and gradient flow
    └── requirements.txt          # Python runtime requirements
```

---

## 3-Step Execution Instructions

### Step 1: Open Google Colab with GPU
1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook or click **Upload**.
2. Upload `KalaKriti_Warli_Pilot.ipynb`.
3. In the Colab menu, go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ select **T4 GPU** $\rightarrow$ click **Save**.

### Step 2: Upload Package Files
1. In Colab's left sidebar, click the **Folder icon (Files)**.
2. Drag and drop `kalakriti_colab_package.zip` into the files area, or upload the `backend` folder directly.
3. If `kalakriti_colab_package.zip` is uploaded, **Cell 2** of the notebook will automatically extract it into the working directory.

### Step 3: Run All Cells
1. In Colab, select **Runtime** $\rightarrow$ **Run all** (or execute cells 1 through 11 sequentially).
2. **Cell 4:** Authorize Google Drive when prompted so checkpoints and evaluation plates persist in `/content/drive/MyDrive/KalaKriti`.
3. **Cell 5 & 6:** Downloads `ajg117/indian-paintings-dataset` via `kagglehub` and displays the **Warli Contact Sheet**.
   - Type `YES` in the input prompt to confirm that the candidate artworks depict authentic Warli grammar (white rice-paste figures on terracotta backgrounds).
4. **Cell 7 & 8:** Acquires 50 MS-COCO content photos, acquires the 4 unseen benchmark photos, and executes pre-flight validation.
5. **Cell 9:** Trains the 2-epoch pilot on the T4 GPU (takes ~3–5 minutes).
6. **Cell 10:** Displays the 3-panel comparison plates (`[Content | Warli Reference | KalaKriti Stylized]`).
7. **Cell 11:** Starts the FastAPI server and launches an ngrok tunnel. Copy the generated public URL:
   ```text
   Public ngrok URL: https://xxxx.ngrok-free.app
   ```
8. In the KalaKriti web application, click the **Settings icon** in the top navigation bar, paste this URL, and click **Save & Apply** to perform real-time neural style transfer inference!
