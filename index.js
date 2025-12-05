const config = {
  video: {
    width: 640,
    height: 480,
    fps: 30,
  },
};

let videoWidth, videoHeight, overlayContext, drawingContext, model;
let isDrawing = false;
let lastX = 0,
  lastY = 0;

const _els = {};
function getEl(id) {
  if (_els[id] === undefined) _els[id] = document.getElementById(id);
  return _els[id];
}

let classificationTempCanvas = null;

const DRAWING_LINE_WIDTH = 5;
let currentDrawingColor = "white";

const PINCH_THRESHOLD = 50;

const EMA_ALPHA = 0.3;
let smoothedX = 0;
let smoothedY = 0;
let isFirstSmoothingFrame = true;
let isUISetupComplete = false;
let keyListenerAdded = false;

const indexFingerTipIndex = 8;
const thumbTipIndex = 4;

const CLASSIFICATION_SIZE = 64;
const DRAWING_CLASSES = ["circle", "door"];
let currentPromptIndex = 0;

let classificationModel = null;
const CLASSIFICATION_MODEL_PATH = "./models/doodle-classifier/model.json";

let uiElements = [];

const fingerLookupIndices = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [0, 5, 6, 7, 8],
  middleFinger: [0, 9, 10, 11, 12],
  ringFinger: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20],
};

const landmarkColors = {
  thumb: "#ff0000",
  indexFinger: "#0000ff",
  middleFinger: "#ffff00",
  ringFinger: "#00ff00",
  pinky: "#ff69b4",
  palmBase: "#ffffff",
};

function drawPoint(x, y, r, context) {
  context.beginPath();
  context.arc(x, y, r, 0, 2 * Math.PI);
  context.fill();
}

function drawPath(points, closePath = false, color, context) {
  context.strokeStyle = color;
  context.lineWidth = 2;
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  context.stroke(region);
}

function drawKeypoints(keypoints, context) {
  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i][0];
    const y = keypoints[i][1];
    context.fillStyle = landmarkColors.palmBase;
    drawPoint(x, y, 5, context);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map((idx) => {
      return [keypoints[idx][0], keypoints[idx][1]];
    });
    drawPath(points, false, landmarkColors[finger], context);
  }
}

async function loadWebcam(width, height, fps) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Webcam not supported in this browser.");
  }

  const webcamElement = getEl("webcam");
  if (!webcamElement) throw new Error("webcam element not found");
  webcamElement.width = width;
  webcamElement.height = height;
  webcamElement.muted = true;

  const mediaConfig = {
    audio: false,
    video: {
      facingMode: "user",
      width: width,
      height: height,
      frameRate: { ideal: fps },
    },
  };

  const stream = await navigator.mediaDevices.getUserMedia(mediaConfig);
  webcamElement.srcObject = stream;

  return new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      resolve(webcamElement);
    };
  });
}

async function loadVideo() {
  const video = await loadWebcam(
    config.video.width,
    config.video.height,
    config.video.fps
  );
  video.play();
  return video;
}

function getDistance(p1, p2) {
  return Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
}

function applyEMA(rawX, rawY) {
  if (isFirstSmoothingFrame) {
    smoothedX = rawX;
    smoothedY = rawY;
    isFirstSmoothingFrame = false;
  } else {
    smoothedX = EMA_ALPHA * rawX + (1 - EMA_ALPHA) * smoothedX;
    smoothedY = EMA_ALPHA * rawY + (1 - EMA_ALPHA) * smoothedY;
  }
  return [smoothedX, smoothedY];
}

function startDrawing(x, y) {
  if (!isDrawing) {
    isDrawing = true;
    [lastX, lastY] = [x, y];

    drawingContext.beginPath();
    drawingContext.fillStyle = currentDrawingColor;
    drawingContext.arc(x, y, DRAWING_LINE_WIDTH / 2, 0, 2 * Math.PI);
    drawingContext.fill();
  }

  const midX = (lastX + x) / 2;
  const midY = (lastY + y) / 2;

  drawingContext.beginPath();
  drawingContext.strokeStyle = currentDrawingColor;
  drawingContext.lineWidth = DRAWING_LINE_WIDTH;
  drawingContext.lineCap = "round";
  drawingContext.lineJoin = "round";

  drawingContext.moveTo(lastX, lastY);

  drawingContext.quadraticCurveTo(x, y, midX, midY);

  drawingContext.stroke();

  [lastX, lastY] = [midX, midY];
}

function stopDrawing() {
  isDrawing = false;
}

function clearCanvasAndResetPrompt() {
  drawingContext.clearRect(0, 0, drawingContext.canvas.width, drawingContext.canvas.height);
  const resultEl = getEl("prediction-result");
  if (resultEl) resultEl.textContent = "";

  currentPromptIndex = (currentPromptIndex + 1) % DRAWING_CLASSES.length;
  updatePrompt();
  const statusEl = getEl("app-status");
  if (statusEl) statusEl.textContent = "Ready to Draw";
}

function updatePrompt() {
  const promptEl = getEl("app-prompt");
  const currentClass = DRAWING_CLASSES[currentPromptIndex];
  promptEl.innerHTML = `Draw a '${currentClass}'`;
}

function setupUI() {
  if (isUISetupComplete) return;

  const containerWidth = config.video.width;
  const containerHeight = config.video.height;

  const overlay = getEl("selection-overlay");

  if (!overlay) {
    console.error("UI Overlay not found!");
    return;
  }

  const overlayRect = overlay.getBoundingClientRect();
  const containerRect = document
    .getElementById("video-container")
    .getBoundingClientRect();

  uiElements = Array.from(document.querySelectorAll("#selection-overlay .ui-button")).map((btn) => {
    const btnRect = btn.getBoundingClientRect();

    return {
      element: btn,
      canvasXMin: btnRect.left - containerRect.left,
      canvasXMax: btnRect.right - containerRect.left,
      canvasYMin: btnRect.top - containerRect.top,
      canvasYMax: btnRect.bottom - containerRect.top,
      color: btn.dataset.color,
      action: btn.dataset.action,
    };
  });

  const classifyBtn = getEl("classify-btn");
  if (classifyBtn) {
    classifyBtn.addEventListener("click", () => {
      const statusEl = getEl("app-status");
      if (statusEl) statusEl.textContent = "Classifying Drawing...";
      classifyDrawing();
    });
  }

  if (!keyListenerAdded) {
    document.addEventListener("keydown", (ev) => {
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (!ev.key) return;
      if (ev.key.toLowerCase() === "d") {
        const statusEl = getEl("app-status");
        if (statusEl) statusEl.textContent = "Classifying (key: D)...";
        classifyDrawing();
      }
    });
    keyListenerAdded = true;
  }

  const initialColor = getEl("white-btn");
  if (initialColor) {
    currentDrawingColor = initialColor.dataset.color || "white";
    initialColor.classList.add("selected");
  }

  isUISetupComplete = true;
}

function checkUISelection(indexCanvasX, indexCanvasY) {
  if (!isUISetupComplete) return false;

  const originalX = videoWidth - indexCanvasX;
  
  let actionTaken = false;

  uiElements.forEach((item) => {
    if (
      originalX > item.canvasXMin &&
      originalX < item.canvasXMax &&
      indexCanvasY > item.canvasYMin &&
      indexCanvasY < item.canvasYMax
    ) {
      if (item.action === "CLEAR") {
        clearCanvasAndResetPrompt();
      } else if (item.action === "CLASSIFY") {
        classifyDrawing();
      } else if (item.color) {
        currentDrawingColor = item.color;

        document
          .querySelectorAll("#selection-overlay .ui-button")
          .forEach((b) => b.classList.remove("selected"));
        item.element.classList.add("selected");
      }
      const statusEl = getEl("app-status");
      if (statusEl)
        statusEl.textContent = `Selected: ${item.action || item.color.toUpperCase()}`;
      actionTaken = true;
    }
  });
  return actionTaken;
}

async function classifyDrawing() {
  const statusEl = getEl("app-status");
  if (statusEl) statusEl.textContent = "Classifying Drawing...";

  if (!classificationModel) {
    const predEl = getEl("prediction-result");
    if (predEl)
      predEl.textContent = "Error: Classification model not loaded. Please ensure model files are in './models/doodle-classifier/'.";
    setTimeout(clearCanvasAndResetPrompt, 1500);
    return;
  }

  const drawingCanvas = getEl("drawing-canvas");
  if (!drawingCanvas) return;

  if (!classificationTempCanvas) {
    classificationTempCanvas = document.createElement("canvas");
    classificationTempCanvas.width = CLASSIFICATION_SIZE;
    classificationTempCanvas.height = CLASSIFICATION_SIZE;
  }
  const tempCanvas = classificationTempCanvas;
  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
  tempCtx.drawImage(drawingCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

  let resultElement = getEl("prediction-result");
  if (resultElement) resultElement.textContent = "Processing...";

  tf.tidy(() => {
    let tensor = tf.browser.fromPixels(tempCanvas, 1);

    tensor = tensor.cast("float32");
    tensor = tensor.div(255.0);
    tensor = tensor.expandDims(0);

    const prediction = classificationModel.predict(tensor);

    const values = prediction.dataSync();
    const predictedIndex = values.indexOf(Math.max(...values));
    const finalPrediction = DRAWING_CLASSES[predictedIndex];

    const isCorrect = finalPrediction === DRAWING_CLASSES[currentPromptIndex];

    setTimeout(() => {
      if (resultElement) {
        if (isCorrect) {
          resultElement.textContent = `✅ Predicted: ${finalPrediction.toUpperCase()}. Correct!`;
        } else {
          resultElement.textContent = `❌ Predicted: ${finalPrediction.toUpperCase()}. Try drawing a ${DRAWING_CLASSES[currentPromptIndex].toUpperCase()}.`;
        }
      }
      setTimeout(clearCanvasAndResetPrompt, 1000);
    }, 500);
  });
}

function handleDrawingGesture(landmarks) {
  const indexTip = landmarks[indexFingerTipIndex];
  const thumbTip = landmarks[thumbTipIndex];

  const pinchDistance = getDistance(indexTip, thumbTip);

  const indexCanvasX = indexTip[0];
  const indexCanvasY = indexTip[1];

  if (pinchDistance < PINCH_THRESHOLD) {
    stopDrawing();
    const statusEl = getEl("app-status");
    if (statusEl) statusEl.textContent = "Selection Mode (Pinch Active)";
    isFirstSmoothingFrame = true;

    checkUISelection(indexCanvasX, indexCanvasY);
  } else {
    if (indexCanvasY < videoHeight - 70) {
      const [smoothedDrawX, smoothedDrawY] = applyEMA(
        indexCanvasX,
        indexCanvasY
      );

      startDrawing(smoothedDrawX, smoothedDrawY);
      const statusEl2 = getEl("app-status");
      if (statusEl2) statusEl2.textContent = `Drawing Mode (${currentDrawingColor.toUpperCase()})`;
      isFirstSmoothingFrame = false;
    } else {
      stopDrawing();
      const statusEl3 = getEl("app-status");
      if (statusEl3) statusEl3.textContent = "Hand Detected (Near Controls)";
      isFirstSmoothingFrame = true;
    }
  }
}

class CustomUrlIOHandler {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
  }

  async load() {
    const url = this.modelUrl;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(
        `Failed to fetch model at ${url}. Status: ${response.status}`
      );
    }

    const modelJSON = await response.json();
    let modelTopology = null;

    if (modelJSON.modelTopology) {
      if (modelJSON.modelTopology.model_config) {
        modelTopology = modelJSON.modelTopology.model_config;
      } else if (modelJSON.modelTopology.modelConfig) {
        modelTopology = modelJSON.modelTopology.modelConfig;
      } else {
        modelTopology = modelJSON.modelTopology;
      }
    } else if (modelJSON.model_config) {
      modelTopology = modelJSON.model_config;
    } else if (modelJSON.modelConfig) {
      modelTopology = modelJSON.modelConfig;
    } else {
      modelTopology = modelJSON.config || modelJSON;
    }

    try {
      const layers =
        (modelTopology && modelTopology.config && modelTopology.config.layers) ||
        (modelTopology && modelTopology.config && modelTopology.config.layers) ||
        null;

      if (layers && layers.length > 0) {
        const inputLayer =
          layers.find((l) => {
            const name = l.className || l.class_name || "";
            return name.toLowerCase().includes("input");
          }) || layers[0];

        const cfg = inputLayer.config || inputLayer.layer_config || null;
        if (cfg) {
          if (cfg.batch_shape && !cfg.batch_input_shape) {
            cfg.batch_input_shape = cfg.batch_shape;
          }
          if (cfg.batch_shape && !cfg.batchInputShape) {
            cfg.batchInputShape = cfg.batch_shape;
          }

          if (!cfg.batch_input_shape && !cfg.batchInputShape && !cfg.input_shape) {
            console.warn(
              `TFJS IOHandler: injecting batch input shape [null,${CLASSIFICATION_SIZE},${CLASSIFICATION_SIZE},1] into model topology`
            );
            cfg.batch_input_shape = [null, CLASSIFICATION_SIZE, CLASSIFICATION_SIZE, 1];
            cfg.batchInputShape = cfg.batch_input_shape;
          }

          if (inputLayer.class_name && !inputLayer.className) {
            inputLayer.className = inputLayer.class_name;
          }
        }
      }
    } catch (e) {
      console.warn("Failed to inject input shape (non-fatal)", e);
    }

    try {
      const topology = modelJSON.modelTopology || modelTopology;

      const mc = topology.model_config || topology.modelConfig || null;
      if (mc) {
        if (mc.class_name && !mc.className) mc.className = mc.class_name;

        if (mc.config) {
          const layers = mc.config.layers || mc.config.layers;
          if (Array.isArray(layers)) {
            layers.forEach((layer) => {
              if (layer.class_name && !layer.className)
                layer.className = layer.class_name;

                if (layer.inbound_nodes && !layer.inboundNodes) {
                  try {
                    const normalized = layer.inbound_nodes.map((nodeEntry) => {
                      if (Array.isArray(nodeEntry)) return nodeEntry;

                      if (nodeEntry && Array.isArray(nodeEntry.args)) {
                        const mapped = nodeEntry.args.map((arg) => {
                          if (arg && arg.config && Array.isArray(arg.config.keras_history)) {
                            return arg.config.keras_history;
                          }
                          if (Array.isArray(arg)) return arg;
                          if (arg && arg.name) return [arg.name, 0, 0];
                          return null;
                        }).filter((x) => x != null);

                        return mapped;
                      }

                      return nodeEntry;
                    });

                    layer.inboundNodes = normalized;
                  } catch (e) {
                    layer.inboundNodes = layer.inbound_nodes;
                  }
                }

              const lc = layer.config || layer.layer_config || {};
              if (lc.batch_shape && !lc.batch_input_shape) {
                lc.batch_input_shape = lc.batch_shape;
              }
              if (lc.batch_shape && !lc.batchInputShape) {
                lc.batchInputShape = lc.batch_shape;
              }
              layer.config = lc;
            });
          }

          if (mc.config.input_layers && !mc.config.inputLayers)
            mc.config.inputLayers = mc.config.input_layers;
          if (mc.config.output_layers && !mc.config.outputLayers)
            mc.config.outputLayers = mc.config.output_layers;
        }
      }

      if (topology.model_config && topology.model_config.config) {
        topology.model_config = topology.model_config;
      }
    } catch (e) {
      console.warn("Normalization of modelTopology failed (non-fatal)", e);
    }

    const artifacts = {
      modelTopology: modelJSON.modelTopology || modelTopology,
      weightsManifest: modelJSON.weightsManifest,
    };

    return artifacts;
  }
}

async function loadClassificationModel() {
  const statusEl = getEl("app-status");
  if (statusEl) statusEl.textContent = "Loading Classification Model...";

  const customHandler = new CustomUrlIOHandler(CLASSIFICATION_MODEL_PATH);

  try {
    classificationModel = await tf.loadLayersModel(customHandler);

    console.log("Classification Model loaded successfully via manual override.");
    if (statusEl) statusEl.textContent = "Classification Model Loaded.";

    tf.tidy(() => {
      const dummyInput = tf.zeros([
        1,
        CLASSIFICATION_SIZE,
        CLASSIFICATION_SIZE,
        1,
      ]);
      const warmUpResult = classificationModel.predict(dummyInput);
      warmUpResult.dispose();
    });
  } catch (e) {
    if (statusEl) statusEl.textContent = "Error Loading Model!";
    console.warn(
      `Could not load classification model from ${CLASSIFICATION_MODEL_PATH}. Using simulation.`
    );
    console.error("TensorFlow.js Model Load Error:", e);
  }
}

async function continouslyDetectLandmarks(video) {
  model = await handpose.load();
  const statusEl2 = getEl("app-status");
  if (statusEl2) statusEl2.textContent = "Handpose Model Loaded. Ready to draw!";

  async function runDetection() {
    if (!isUISetupComplete) {
      setupUI();
    }

    overlayContext.clearRect(
      0,
      0,
      overlayContext.canvas.width,
      overlayContext.canvas.height
    );

    try {
      const predictions = await model.estimateHands(video, true);

      if (predictions && predictions.length > 0) {
        const landmarks = predictions[0].landmarks.map((kp) => [videoWidth - kp[0], kp[1]]);

        drawKeypoints(landmarks, overlayContext);
        handleDrawingGesture(landmarks);
      } else {
        stopDrawing();
        if (statusEl2) statusEl2.textContent = "No Hand Detected";
        isFirstSmoothingFrame = true;
      }
    } catch (e) {
      console.warn("Error estimating hands:", e);
      stopDrawing();
      if (statusEl2) statusEl2.textContent = "Detection Error";
      isFirstSmoothingFrame = true;
    }

    requestAnimationFrame(runDetection);
  }

  runDetection();
}

async function main() {
  const video = await loadVideo();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  const overlayCanvas = getEl("overlay-canvas");
  if (!overlayCanvas) throw new Error("overlay-canvas element not found");
  overlayCanvas.width = videoWidth;
  overlayCanvas.height = videoHeight;
  overlayContext = overlayCanvas.getContext("2d");

  overlayContext.translate(overlayCanvas.width, 0);
  overlayContext.scale(-1, 1);

  const drawingCanvas = getEl("drawing-canvas");
  if (!drawingCanvas) throw new Error("drawing-canvas element not found");
  drawingCanvas.width = videoWidth;
  drawingCanvas.height = videoHeight;
  drawingContext = drawingCanvas.getContext("2d");

  drawingContext.translate(drawingCanvas.width, 0);
  drawingContext.scale(-1, 1);

  drawingContext.lineCap = "round";
  drawingContext.lineJoin = "round";

  updatePrompt();

  await Promise.all([
    loadClassificationModel(),
    continouslyDetectLandmarks(video),
  ]);
}

main();
