# KalaKriti — Neural Style Transfer

KalaKriti is a web-based Neural Style Transfer (NST) application that transforms user-uploaded photographs into artistic images using deep learning. The project focuses on Indian traditional art styles, with the goal of preserving the structure and recognisable features of the original content image while transferring artistic characteristics.

The application uses a React + Vite frontend and a Python backend built with PyTorch and FastAPI. Google Colab can be used for GPU-backed inference and training.

## Features

- Upload a content image for stylization.
- Select an artistic style and configure the available style-transfer settings.
- Generate stylized outputs through a deep-learning backend.
- Use Google Colab GPU resources instead of requiring a powerful local GPU.
- Separate model training from inference.

**Current model:** The repository includes a Warli v4 checkpoint with edge-preservation loss in the training script. Other styles shown in the interface may not yet have trained checkpoints connected to the backend.

## Technology Stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python, FastAPI
- **Deep learning:** PyTorch, convolutional neural network
- **Inference and training:** Local Python environment or Google Colab GPU
- **API tunnelling:** ngrok, when the backend runs in Colab

## Project Structure

```text
f_y_p/
├── src/                         # Frontend source code
├── public/                      # Static assets
├── backend/
│   ├── server.py                # FastAPI application
│   ├── train_colab.py           # Model training script
│   ├── models/
│   │   └── transformer_net.py   # Neural network architecture
│   ├── checkpoints/
│   │   └── kalakriti_cin_warli_color_v4.pth
│   └── requirements.txt         # Python dependencies
├── KalaKriti_Warli_Pilot.ipynb  # Colab notebook, if present
├── README_COLAB.md              # Additional Colab instructions
├── package.json
└── README.md
```

Some notebook or archive files may vary between repository versions.

## Requirements

### Frontend
- Node.js and npm

### Backend
- Python 3.10 or newer recommended
- Packages listed in `backend/requirements.txt`
- A compatible PyTorch installation
- Google Colab GPU recommended for faster inference and training

## 1. Clone the Repository

Install Git if it is not already installed, then run:

```bash
git clone https://github.com/deek023/f_y_p.git
cd f_y_p
```

To download the latest changes into an existing clone:

```bash
git pull origin main
```

## 2. Run the Frontend

From the project root, install the Node.js dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal. With the current Vite configuration, this is typically:

```text
http://localhost:3000
```

If Vite displays a different port, use the URL it prints.

To create a production build:

```bash
npm run build
```

## 3. Run the Backend

There are two options. Use Google Colab for GPU-backed inference, or run the backend locally.

### Option A: Google Colab + ngrok (Recommended)

This option is useful when the local computer does not have a suitable GPU.

1. Open [Google Colab](https://colab.research.google.com/).
2. Open or upload the project's Colab notebook, if included in the repository.
3. Select **Runtime → Change runtime type** and choose a GPU runtime if available.
4. Follow the notebook cells in order to install dependencies, load the model checkpoint, and start the FastAPI server.
5. Start the ngrok tunnel using the notebook's instructions.
6. Copy the generated public HTTPS URL.
7. Open KalaKriti in your browser and navigate to its backend/API settings.
8. Paste the public URL, save the setting, and test the connection.

Verify that the backend is reachable by opening:

```text
<YOUR_BACKEND_URL>/health
```

The endpoint should return a JSON health/status response.

**Important:** Keep the Colab runtime and ngrok tunnel running while using the application. The public URL may change when the tunnel restarts; update the URL in the frontend settings if it changes.

See [`README_COLAB.md`](README_COLAB.md) for additional notebook instructions, if that file is present.

### Option B: Run the Backend Locally

From the repository root, create a virtual environment.

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Start the FastAPI application from the repository root:

```bash
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

If the import path does not work, try running from the backend directory:

```bash
cd backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open the health endpoint:

```text
http://localhost:8000/health
```

In the frontend's backend/API settings, configure the backend URL as:

```text
http://localhost:8000
```

Local CPU inference may be slower than GPU inference. If the model fails to load, check that the checkpoint file exists at the path expected by the backend.

## 4. Using KalaKriti

1. Start the frontend.
2. Start the backend using one of the options above.
3. Configure the backend URL in the frontend's backend/API settings.
4. Confirm that the backend connection is successful.
5. Upload a content image.
6. Select a supported style and adjust the available settings.
7. Run the stylization process and review the output.

The uploaded content image is used as an inference-time input. It is not used to train the model.

## API Endpoints

The frontend is designed to communicate with the following backend routes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend and model status |
| POST | `/stylize` | Generate a stylized image |

The frontend request uses fields including:

- `contentImageBase64`
- `styleId`
- `styleStrength`
- `resolution`

The frontend and backend must use compatible request and response formats.

## Training the Model (Optional)

Training is separate from inference. Users do not need to retrain the model every time they upload a photograph.

The training workflow generally involves:

1. Preparing content images and authentic style-reference images.
2. Installing the required Python and PyTorch dependencies.
3. Running the training script in a suitable GPU environment, such as Google Colab.
4. Saving the trained checkpoint to the expected location.
5. Restarting the backend so it loads the intended checkpoint.
6. Evaluating outputs on content images that were not used for training.

The current Warli v4 checkpoint is:

```text
backend/checkpoints/kalakriti_cin_warli_color_v4.pth
```

The uploaded content image must remain an inference-time input and should not be added to the training dataset automatically.

## Troubleshooting

### Frontend shows “Backend Disconnected” or “Failed to fetch”

- Confirm that the FastAPI server is running.
- Open the backend's `/health` endpoint directly.
- Check that the frontend's saved backend URL is correct.
- If using Colab, verify that the runtime and ngrok tunnel are still active.
- If the browser reports a CORS error, check the FastAPI CORS configuration and restart the backend after making changes.

### `localhost:8000` returns “Connection refused”

The backend is not running at that address or is using a different port. Start the API and verify the URL again.

### The checkpoint is missing or the model does not load

Confirm that the checkpoint exists:

```text
backend/checkpoints/kalakriti_cin_warli_color_v4.pth
```

Check the backend startup logs for model-loading errors. Ensure that the checkpoint and model architecture are compatible.

### Python dependencies fail to install

- Confirm that the virtual environment is activated.
- Upgrade pip.
- Install the dependencies listed in `backend/requirements.txt`.
- If PyTorch installation fails, use the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your Python and hardware configuration.

## Responsible Use

- Use images and artwork that you have permission to use.
- Respect the licensing and attribution requirements of artistic reference images.
- Do not commit API tokens, credentials, private datasets, or environment files.
- Do not treat a preview or disconnected state as a genuine model-generated result.

## Repository

[https://github.com/deek023/f_y_p](https://github.com/deek023/f_y_p)

## Project Status

KalaKriti is under development. Model quality, supported styles, and deployment configuration may change as training and validation continue.
