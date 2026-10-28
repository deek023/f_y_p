/**
 * KalaKriti — Real-Time Multi-Style Neural Style Transfer
 * Frontend Interface connecting to FastAPI / Colab backend.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Sliders,
  Settings,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  RefreshCw,
  ExternalLink,
  ChevronRight,
  Maximize2,
  Download,
  Info,
  ShieldAlert,
} from 'lucide-react';
import {
  checkHealth,
  stylizeImage,
  getBackendUrl,
  setBackendUrl,
  HealthResponse,
  StylizeResponse,
} from './services/nstApi';

interface StyleItem {
  id: string;
  name: string;
  origin: string;
  status: 'active' | 'upcoming';
  description: string;
  motifs: string[];
  palette: string[];
}

const SUPPORTED_STYLES: StyleItem[] = [
  {
    id: 'warli',
    name: 'Warli Art',
    origin: 'Maharashtra, India',
    status: 'active',
    description: 'Earthy terracotta backdrop with stark white rice-paste geometric figures.',
    motifs: ['Inverted-triangle torsos', 'Circular heads', 'Rhythmic linear limbs', 'Tarpa circle dance'],
    palette: ['#7A2F1E', '#F4F0EA', '#4A1E14'],
  },
  {
    id: 'madhubani',
    name: 'Madhubani (Mithila)',
    origin: 'Bihar, India',
    status: 'upcoming',
    description: 'Double outlines, geometric hatching fills, and vibrant natural pigments.',
    motifs: ['Fish & peacock symbols', 'Lotus flowers', 'Intricate line borders'],
    palette: ['#D62828', '#F77F00', '#FCBF49', '#003049'],
  },
  {
    id: 'kalamkari',
    name: 'Kalamkari',
    origin: 'Andhra Pradesh, India',
    status: 'upcoming',
    description: 'Hand-drawn organic textile dye art with narrative figurative motifs.',
    motifs: ['Flowing vine borders', 'Earthy ochre & indigo dyes', 'Mythological scenes'],
    palette: ['#3A5A40', '#A3B18A', '#582F0E', '#E6CCB2'],
  },
  {
    id: 'gond',
    name: 'Gond Folk Art',
    origin: 'Madhya Pradesh, India',
    status: 'upcoming',
    description: 'Dots, fine dashes, and vibrant repeated patterns illuminating nature and wildlife.',
    motifs: ['Dotted animal silhouettes', 'Tree of life', 'Vibrant pattern textures'],
    palette: ['#E63946', '#F1FAEE', '#A8DADC', '#457B9D'],
  },
  {
    id: 'pattachitra',
    name: 'Pattachitra',
    origin: 'Odisha & Bengal, India',
    status: 'upcoming',
    description: 'Ornate scroll paintings with rich natural colors, crisp contours, and floral borders.',
    motifs: ['Sharply contoured profiles', 'Intricate lotus borders', 'Rich mineral pigments'],
    palette: ['#BC4749', '#F2E8CF', '#386641', '#6A994E'],
  },
];

// Curated royalty-free test samples across the 4 key evaluation categories
const SAMPLE_PREVIEWS = [
  {
    category: 'Portrait',
    name: 'Human Subject',
    url: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=400&q=80',
    description: 'Evaluates facial silhouette and pose preservation',
  },
  {
    category: 'Monument',
    name: 'Historic Architecture',
    url: 'https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=400&q=80',
    description: 'Evaluates linear geometric contours and architectural lines',
  },
  {
    category: 'Landscape',
    name: 'Natural Scenery',
    url: 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=80',
    description: 'Evaluates organic texture and background domain adaptation',
  },
  {
    category: 'Animal',
    name: 'Wildlife Silhouette',
    url: 'https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?auto=format&fit=crop&w=400&q=80',
    description: 'Evaluates anatomical outline and high-contrast stroke capture',
  },
];

export default function App() {
  // Backend & Connection State
  const [backendUrl, setBackendUrlState] = useState<string>(getBackendUrl());
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [tempBackendUrl, setTempBackendUrl] = useState(backendUrl);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Content Image State
  const [contentBase64, setContentBase64] = useState<string | null>(null);
  const [imageMetadata, setImageMetadata] = useState<{ width: number; height: number; sizeKb: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Stylize Parameters
  const [selectedStyleId, setSelectedStyleId] = useState<string>('warli');
  const [styleStrength, setStyleStrength] = useState<number>(0.85);
  const [resolution, setResolution] = useState<number>(512);

  // Inference State
  const [isProcessing, setIsProcessing] = useState(false);
  const [stylizedResult, setStylizedResult] = useState<StylizeResponse | null>(null);
  const [inferenceError, setInferenceError] = useState<string | null>(null);

  // Comparison Slider State
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const comparisonContainerRef = useRef<HTMLDivElement>(null);

  // Poll backend health on initial mount and when URL changes
  useEffect(() => {
    checkBackendHealth();
  }, [backendUrl]);

  const checkBackendHealth = async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await checkHealth(backendUrl);
      setHealth(data);
    } catch (err: unknown) {
      setHealth(null);
      setHealthError(err instanceof Error ? err.message : String(err));
    } finally {
      setHealthLoading(false);
    }
  };

  const handleSaveSettings = () => {
    setBackendUrl(tempBackendUrl);
    setBackendUrlState(tempBackendUrl);
    setIsSettingsOpen(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPEG, PNG, or WebP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const b64 = event.target?.result as string;
      setContentBase64(b64);
      setStylizedResult(null);
      setInferenceError(null);

      // Extract image dimensions
      const img = new Image();
      img.onload = () => {
        setImageMetadata({
          width: img.width,
          height: img.height,
          sizeKb: Math.round(file.size / 1024),
        });
      };
      img.src = b64;
    };
    reader.readAsDataURL(file);
  };

  const loadSampleImage = async (url: string) => {
    try {
      setInferenceError(null);
      setStylizedResult(null);
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = () => {
        const b64 = reader.result as string;
        setContentBase64(b64);
        const img = new Image();
        img.onload = () => {
          setImageMetadata({
            width: img.width,
            height: img.height,
            sizeKb: Math.round(blob.size / 1024),
          });
        };
        img.src = b64;
      };
      reader.readAsDataURL(blob);
    } catch {
      setInferenceError('Failed to load sample image. Please upload a local file instead.');
    }
  };

  const handleRunInference = async () => {
    if (!contentBase64) return;

    // Strict authenticity guard
    if (health && !health.modelReady) {
      setInferenceError(
        'Inference cannot proceed: model weights for Warli art are pending Stage 5 training on Google Colab. ' +
        'KalaKriti strictly adheres to authenticity and refuses to return fake or filtered images.'
      );
      return;
    }

    setIsProcessing(true);
    setInferenceError(null);

    try {
      const response = await stylizeImage({
        contentImageBase64: contentBase64,
        styleId: selectedStyleId,
        styleStrength,
        resolution,
      });
      setStylizedResult(response);
    } catch (err: unknown) {
      setInferenceError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  // Drag handler for comparison split viewer
  const handleSliderMove = (clientX: number) => {
    if (!comparisonContainerRef.current) return;
    const rect = comparisonContainerRef.current.getBoundingClientRect();
    const pos = ((clientX - rect.left) / rect.width) * 100;
    setSliderPosition(Math.max(0, Math.min(100, pos)));
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (e.buttons === 1) {
      handleSliderMove(e.clientX);
    }
  };

  return (
    <div className="min-h-screen bg-stone-900 text-stone-100 flex flex-col font-sans">
      {/* Top Academic Header */}
      <header className="border-b border-stone-800 bg-stone-950/80 backdrop-blur sticky top-0 z-30 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-amber-700/30 border border-amber-600/50 flex items-center justify-center text-amber-400 font-bold">
                क
              </div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                KalaKriti
                <span className="text-xs font-mono font-normal uppercase px-2 py-0.5 rounded bg-stone-800 text-stone-300 border border-stone-700">
                  v1.0 (Stage 4)
                </span>
              </h1>
            </div>
            <p className="text-xs text-stone-400 mt-1 max-w-2xl">
              Real-Time Multi-Style Neural Style Transfer for Visual Domain Adaptation and Indian Traditional Art
            </p>
          </div>

          {/* Backend Status & Config Indicator */}
          <div className="flex items-center gap-3">
            <a
              href="/kalakriti_colab_package.zip"
              download="kalakriti_colab_package.zip"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition cursor-pointer bg-amber-950/50 border-amber-700 hover:border-amber-500 text-amber-300 shadow-sm"
              title="Download Colab Notebook & Backend Package (.zip)"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Colab Package</span>
            </a>

            <button
              onClick={() => setIsSettingsOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition cursor-pointer bg-stone-900 border-stone-700 hover:border-stone-500 text-stone-300"
              title="Click to configure backend URL (FastAPI / ngrok)"
            >
              {healthLoading ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-stone-400" />
              ) : health?.modelReady ? (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
              ) : health ? (
                <span className="h-2 w-2 rounded-full bg-amber-400"></span>
              ) : (
                <span className="h-2 w-2 rounded-full bg-rose-500"></span>
              )}

              <span>
                {healthLoading
                  ? 'Connecting...'
                  : health?.modelReady
                  ? 'Backend Ready'
                  : health
                  ? 'Weights Pending'
                  : 'Backend Offline'}
              </span>
              <Settings className="w-3.5 h-3.5 ml-1 text-stone-500" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Style & Upload Controls (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Diagnostic Status Alert when model is not ready */}
          {health && !health.modelReady && (
            <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-200 text-xs space-y-2">
              <div className="flex items-center gap-2 font-semibold text-amber-400">
                <ShieldAlert className="w-4 h-4 shrink-0" />
                <span>Stage 4 Scaffolding Active: Weights Awaiting Training</span>
              </div>
              <p className="leading-relaxed text-stone-300">
                The FastAPI backend server is operational, but the genuine Warli model checkpoint{' '}
                <code className="text-amber-300 font-mono">kalakriti_cin_warli.pth</code> has not been placed in{' '}
                <code className="text-amber-300 font-mono">backend/checkpoints/</code>. KalaKriti does not fake output
                with synthetic filters.
              </p>
              <div className="pt-2 flex flex-col gap-1.5">
                <a
                  href="/kalakriti_colab_package.zip"
                  download="kalakriti_colab_package.zip"
                  className="inline-flex items-center gap-1.5 font-semibold text-amber-300 hover:text-amber-200 underline text-xs"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Complete Colab Pilot Package (KalaKriti_Warli_Pilot.ipynb + backend)</span>
                </a>
                <span className="text-[11px] text-stone-400">
                  Run the notebook on Google Colab with a T4 GPU to train the model and generate the ngrok tunnel.
                </span>
              </div>
            </div>
          )}

          {/* Backend Offline Warning */}
          {!health && !healthLoading && (
            <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800/50 text-rose-200 text-xs space-y-2">
              <div className="flex items-center gap-2 font-semibold text-rose-400">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>Backend Disconnected</span>
              </div>
              <p className="leading-relaxed text-stone-300">
                Cannot reach FastAPI at <code className="text-rose-300 font-mono">{backendUrl}</code>. Ensure the backend
                is started locally or an ngrok tunnel from Google Colab is active.
              </p>
              <button
                onClick={() => setIsSettingsOpen(true)}
                className="text-[11px] font-semibold text-rose-300 underline hover:text-rose-200 cursor-pointer"
              >
                Configure Backend URL &rarr;
              </button>
            </div>
          )}

          {/* Section 1: Style Selection */}
          <div className="bg-stone-950/60 border border-stone-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold tracking-wide uppercase text-stone-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-500" />
                1. Select Traditional Art Style
              </h2>
              <span className="text-xs text-stone-500">Style Index: 0</span>
            </div>

            <div className="space-y-3">
              {SUPPORTED_STYLES.map((style) => {
                const isSelected = selectedStyleId === style.id;
                const isActive = style.status === 'active';

                return (
                  <div
                    key={style.id}
                    onClick={() => isActive && setSelectedStyleId(style.id)}
                    className={`relative p-3.5 rounded-xl border transition-all ${
                      isActive
                        ? isSelected
                          ? 'border-amber-600 bg-amber-950/20 shadow-lg shadow-amber-950/30 cursor-pointer'
                          : 'border-stone-800 bg-stone-900/50 hover:border-stone-700 cursor-pointer'
                        : 'border-stone-800/40 bg-stone-900/20 opacity-60 cursor-not-allowed'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-stone-100 text-sm">{style.name}</span>
                          <span
                            className={`text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${
                              isActive
                                ? 'bg-amber-900/60 text-amber-300 border border-amber-700/50'
                                : 'bg-stone-800 text-stone-400'
                            }`}
                          >
                            {isActive ? 'Primary Focus' : 'Stage 6 Target'}
                          </span>
                        </div>
                        <p className="text-xs text-stone-400 mt-1">{style.description}</p>
                      </div>

                      {/* Palette circles */}
                      <div className="flex -space-x-1 shrink-0 ml-3">
                        {style.palette.map((c, i) => (
                          <div
                            key={i}
                            className="w-4 h-4 rounded-full border border-stone-900"
                            style={{ backgroundColor: c }}
                          />
                        ))}
                      </div>
                    </div>

                    {/* Cultural Anatomy Notes */}
                    <div className="mt-3 pt-2.5 border-t border-stone-800/60 flex flex-wrap gap-1.5">
                      {style.motifs.map((motif, i) => (
                        <span
                          key={i}
                          className="text-[10px] px-2 py-0.5 rounded-full bg-stone-800/80 text-stone-300 border border-stone-700/40"
                        >
                          {motif}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Section 2: Upload Content Image */}
          <div className="bg-stone-950/60 border border-stone-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold tracking-wide uppercase text-stone-300 flex items-center gap-2">
                <Upload className="w-4 h-4 text-amber-500" />
                2. Input Content Photograph
              </h2>
              {imageMetadata && (
                <span className="text-xs font-mono text-stone-400">
                  {imageMetadata.width} × {imageMetadata.height} ({imageMetadata.sizeKb} KB)
                </span>
              )}
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
            />

            {!contentBase64 ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-stone-700 hover:border-amber-600/70 bg-stone-900/30 rounded-xl p-8 text-center cursor-pointer transition group"
              >
                <div className="w-12 h-12 rounded-full bg-stone-800 group-hover:bg-amber-950/50 flex items-center justify-center mx-auto mb-3 transition">
                  <Upload className="w-5 h-5 text-stone-400 group-hover:text-amber-400" />
                </div>
                <p className="text-sm font-medium text-stone-200">Upload photograph to stylize</p>
                <p className="text-xs text-stone-500 mt-1">Portrait, monument, landscape, or animal (JPEG, PNG, WebP)</p>
              </div>
            ) : (
              <div className="relative group rounded-xl overflow-hidden border border-stone-700">
                <img
                  src={contentBase64}
                  alt="Content Preview"
                  className="w-full max-h-56 object-contain bg-stone-950"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="absolute bottom-3 right-3 bg-stone-900/90 hover:bg-stone-800 text-xs px-3 py-1.5 rounded-lg border border-stone-600 text-stone-200 backdrop-blur transition cursor-pointer"
                >
                  Change Image
                </button>
              </div>
            )}

            {/* Quick Benchmark Category Samples */}
            <div className="space-y-2 pt-2">
              <span className="text-[11px] font-medium uppercase text-stone-400 tracking-wider">
                Or select benchmark evaluation sample:
              </span>
              <div className="grid grid-cols-2 gap-2">
                {SAMPLE_PREVIEWS.map((sample) => (
                  <button
                    key={sample.category}
                    onClick={() => loadSampleImage(sample.url)}
                    className="p-2 rounded-lg bg-stone-900/60 border border-stone-800 hover:border-stone-700 flex items-center gap-2 text-left transition cursor-pointer"
                  >
                    <img
                      src={sample.url}
                      alt={sample.category}
                      className="w-9 h-9 rounded object-cover shrink-0"
                    />
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-stone-200 truncate">{sample.name}</p>
                      <p className="text-[10px] text-stone-500 truncate">{sample.category}</p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Section 3: Fine-Tuning & Execution */}
          <div className="bg-stone-950/60 border border-stone-800 rounded-2xl p-5 space-y-4">
            <h2 className="text-sm font-semibold tracking-wide uppercase text-stone-300 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-amber-500" />
              3. Inference Parameters
            </h2>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-stone-400">Style Strength ($\alpha$ blend)</span>
                  <span className="font-mono text-amber-400">{(styleStrength * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={styleStrength}
                  onChange={(e) => setStyleStrength(parseFloat(e.target.value))}
                  className="w-full accent-amber-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-stone-400">Target Long-Edge Resolution</span>
                  <span className="font-mono text-stone-300">{resolution} px</span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {[256, 512, 768, 1024].map((res) => (
                    <button
                      key={res}
                      onClick={() => setResolution(res)}
                      className={`py-1.5 rounded text-xs font-mono transition cursor-pointer border ${
                        resolution === res
                          ? 'bg-amber-950/40 border-amber-600 text-amber-300'
                          : 'bg-stone-900 border-stone-800 text-stone-400 hover:border-stone-700'
                      }`}
                    >
                      {res}p
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Button */}
              <button
                onClick={handleRunInference}
                disabled={!contentBase64 || isProcessing || (health !== null && !health.modelReady)}
                className={`w-full py-3 rounded-xl font-semibold text-sm transition flex items-center justify-center gap-2 ${
                  !contentBase64 || (health !== null && !health.modelReady)
                    ? 'bg-stone-800 text-stone-500 cursor-not-allowed border border-stone-700/50'
                    : isProcessing
                    ? 'bg-amber-600 text-stone-950 animate-pulse cursor-wait'
                    : 'bg-amber-600 hover:bg-amber-500 text-stone-950 cursor-pointer shadow-lg shadow-amber-900/30'
                }`}
              >
                {isProcessing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Executing Feed-Forward Pass...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Stylize with KalaKriti</span>
                  </>
                )}
              </button>

              {inferenceError && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs">
                  <p className="font-semibold mb-1">Execution Notice:</p>
                  <p className="leading-relaxed">{inferenceError}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Comparative Inspection & Visualizer (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-stone-950/60 border border-stone-800 rounded-2xl p-5 min-h-[580px] flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-stone-800">
              <div>
                <h2 className="text-sm font-semibold tracking-wide uppercase text-stone-300">
                  Visual Inspection & Comparison
                </h2>
                <p className="text-xs text-stone-500 mt-0.5">
                  Interactive before/after split slider for structural and domain adaptation verification
                </p>
              </div>

              {stylizedResult && (
                <div className="flex items-center gap-4 text-xs font-mono">
                  <div className="flex items-center gap-1.5 text-emerald-400">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{stylizedResult.inferenceTimeMs.toFixed(1)} ms</span>
                  </div>
                  <div className="text-stone-400">
                    {stylizedResult.resolution[0]} × {stylizedResult.resolution[1]}
                  </div>
                </div>
              )}
            </div>

            {/* Visualizer Canvas Area */}
            <div className="flex-1 flex items-center justify-center py-6">
              {!contentBase64 ? (
                <div className="text-center p-8 text-stone-500 space-y-3">
                  <div className="w-16 h-16 rounded-2xl bg-stone-900 border border-stone-800 flex items-center justify-center mx-auto">
                    <Maximize2 className="w-6 h-6 text-stone-600" />
                  </div>
                  <p className="text-sm font-medium text-stone-400">No content image selected</p>
                  <p className="text-xs max-w-sm mx-auto text-stone-600">
                    Upload an image or pick a benchmark sample to test content structure preservation against the Warli
                    domain adaptation.
                  </p>
                </div>
              ) : stylizedResult ? (
                /* Interactive Split Slider */
                <div
                  ref={comparisonContainerRef}
                  onMouseMove={handleMouseMove}
                  onTouchMove={handleTouchMove}
                  className="relative w-full max-w-xl aspect-4/3 rounded-xl overflow-hidden border border-stone-700 shadow-2xl select-none cursor-ew-resize"
                >
                  {/* Background: Original Content */}
                  <img
                    src={contentBase64}
                    alt="Original Content"
                    className="absolute inset-0 w-full h-full object-contain bg-stone-950 pointer-events-none"
                  />
                  <div className="absolute top-3 left-3 bg-stone-950/80 px-2.5 py-1 rounded text-[11px] font-mono text-stone-300 border border-stone-700">
                    Original
                  </div>

                  {/* Foreground: Stylized Output with clip-path */}
                  <div
                    className="absolute inset-0 overflow-hidden pointer-events-none"
                    style={{ clipPath: `inset(0 0 0 ${sliderPosition}%)` }}
                  >
                    <img
                      src={stylizedResult.stylizedImageBase64}
                      alt="Stylized Output"
                      className="absolute inset-0 w-full h-full object-contain bg-stone-950"
                    />
                    <div className="absolute top-3 right-3 bg-amber-950/80 px-2.5 py-1 rounded text-[11px] font-mono text-amber-300 border border-amber-700">
                      KalaKriti (Warli)
                    </div>
                  </div>

                  {/* Split Divider Line */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-amber-400 pointer-events-none shadow-lg"
                    style={{ left: `${sliderPosition}%` }}
                  >
                    <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-amber-500 text-stone-950 flex items-center justify-center text-xs font-bold shadow-lg">
                      &harr;
                    </div>
                  </div>
                </div>
              ) : (
                /* Pre-inference Preview with Structural Guidance */
                <div className="w-full max-w-xl space-y-4">
                  <div className="rounded-xl overflow-hidden border border-stone-800 bg-stone-950/80 p-2">
                    <img
                      src={contentBase64}
                      alt="Selected Content"
                      className="w-full max-h-80 object-contain rounded-lg mx-auto"
                    />
                  </div>
                  <div className="p-4 rounded-xl bg-stone-900/40 border border-stone-800/80 text-xs text-stone-400 space-y-1.5">
                    <div className="flex items-center gap-1.5 text-stone-200 font-medium">
                      <Info className="w-4 h-4 text-amber-500" />
                      <span>Stage 4 Validation Ready</span>
                    </div>
                    <p>
                      Input image validated. Once Stage 5 weights are loaded, the feed-forward generator will process this
                      image in a single forward pass without iterative backpropagation.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Actions if result exists */}
            {stylizedResult && (
              <div className="pt-4 border-t border-stone-800 flex justify-between items-center">
                <span className="text-xs text-stone-400">
                  Drag the slider to compare original contours against the synthesized Warli stroke grammar.
                </span>
                <a
                  href={stylizedResult.stylizedImageBase64}
                  download={`kalakriti_warli_${Date.now()}.jpg`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-stone-900 hover:bg-stone-800 border border-stone-700 text-xs text-stone-200 transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Artwork</span>
                </a>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Backend URL Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 bg-stone-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-stone-900 border border-stone-700 rounded-2xl max-w-md w-full p-6 space-y-5 shadow-2xl">
            <div>
              <h3 className="text-base font-bold text-stone-100 flex items-center gap-2">
                <Settings className="w-4 h-4 text-amber-500" />
                Backend Connection Settings
              </h3>
              <p className="text-xs text-stone-400 mt-1">
                Configure your FastAPI inference server URL (e.g. Google Colab ngrok tunnel or local endpoint).
              </p>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-medium text-stone-300">FastAPI Server URL</label>
              <input
                type="text"
                value={tempBackendUrl}
                onChange={(e) => setTempBackendUrl(e.target.value)}
                placeholder="http://localhost:8000 or https://xxxx.ngrok-free.app"
                className="w-full px-3 py-2 rounded-lg bg-stone-950 border border-stone-700 text-stone-200 text-xs font-mono focus:outline-none focus:border-amber-500"
              />
              <p className="text-[11px] text-stone-500">
                Default: <code className="text-stone-400">http://localhost:8000</code>. When using Colab, paste your
                public ngrok URL.
              </p>
            </div>

            {/* Modal Connection Test */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => checkBackendHealth()}
                disabled={healthLoading}
                className="w-full py-2 rounded-lg bg-stone-800 hover:bg-stone-700 text-stone-200 text-xs font-medium border border-stone-600 transition flex items-center justify-center gap-2 cursor-pointer"
              >
                {healthLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                <span>Test Connection Now</span>
              </button>

              {health && (
                <div className="mt-2 p-2 rounded bg-stone-950 text-[11px] border border-stone-800 space-y-1">
                  <div className="flex items-center gap-1.5 text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Ping Successful: {health.engine}</span>
                  </div>
                  <div className="text-stone-400">
                    Device: {health.device} | Weights Loaded: {health.modelReady ? 'Yes' : 'Pending'}
                  </div>
                </div>
              )}

              {healthError && (
                <div className="mt-2 p-2 rounded bg-rose-950/40 text-[11px] border border-rose-800/60 text-rose-300">
                  {healthError}
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="flex justify-end gap-3 pt-3 border-t border-stone-800">
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                className="px-4 py-2 rounded-lg text-xs font-medium text-stone-400 hover:text-stone-200 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveSettings}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-amber-600 hover:bg-amber-500 text-stone-950 cursor-pointer transition"
              >
                Save & Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
