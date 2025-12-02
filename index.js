// index.js (FINAL CODE WITH TENSORFLOW.JS MODEL INTEGRATION)

// --- CONFIGURATION ---
const config = {
  video: {
    width: 640,
    height: 480,
    fps: 30,
  },
};

// --- GLOBAL VARIABLES ---
let videoWidth, videoHeight, overlayContext, drawingContext, model;
let isDrawing = false;
let lastX = 0,
  lastY = 0;

// Drawing constants
const DRAWING_LINE_WIDTH = 5;
let currentDrawingColor = "white"; // Initial default color

// Gesture threshold (in pixels - rough estimates based on a 640x480 canvas)
const PINCH_THRESHOLD = 50;

// EMA Smoothing constants
const EMA_ALPHA = 0.3; // 0.3 is a good balance for smoothness vs lag
let smoothedX = 0;
let smoothedY = 0;
let isFirstSmoothingFrame = true;
let isUISetupComplete = false; // Ensures UI coordinates are calculated only once
let keyListenerAdded = false; // Ensure key listener is attached only once

// Landmark indices (MediaPipe Hand model)
const indexFingerTipIndex = 8;
const thumbTipIndex = 4;

// Classification setup
const CLASSIFICATION_SIZE = 64; // Model input size (must match Python training: 64x64)
// FIX: Updated DRAWING_CLASSES to match successfully trained model
const DRAWING_CLASSES = ["circle", "door"];
let currentPromptIndex = 0;

// --- MODEL INTEGRATION ---
let classificationModel = null;
const CLASSIFICATION_MODEL_PATH = "./models/doodle-classifier/model.json";
// -------------------------

// UI elements storage
let uiElements = [];

// --- UTILITY FUNCTIONS (For Keypoint Drawing) ---
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
  // Draw points
  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i][0];
    const y = keypoints[i][1];
    context.fillStyle = landmarkColors.palmBase;
    drawPoint(x, y, 5, context);
  }

  // Draw lines
  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map((idx) => {
      return [keypoints[idx][0], keypoints[idx][1]];
    });
    drawPath(points, false, landmarkColors[finger], context);
  }
}

// --- VIDEO SETUP ---

async function loadWebcam(width, height, fps) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Webcam not supported in this browser.");
  }

  const webcamElement = document.getElementById("webcam");
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

// --- DRAWING AND GESTURE LOGIC ---

function getDistance(p1, p2) {
  return Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2));
}

function applyEMA(rawX, rawY) {
  if (isFirstSmoothingFrame) {
    smoothedX = rawX;
    smoothedY = rawY;
    isFirstSmoothingFrame = false;
  } else {
    // Apply Exponential Moving Average (EMA)
    smoothedX = EMA_ALPHA * rawX + (1 - EMA_ALPHA) * smoothedX;
    smoothedY = EMA_ALPHA * rawY + (1 - EMA_ALPHA) * smoothedY;
  }
  return [smoothedX, smoothedY];
}

function startDrawing(x, y) {
  if (!isDrawing) {
    isDrawing = true;
    [lastX, lastY] = [x, y];

    // Start a new path for the initial point
    drawingContext.beginPath();
    drawingContext.fillStyle = currentDrawingColor;
    drawingContext.arc(x, y, DRAWING_LINE_WIDTH / 2, 0, 2 * Math.PI);
    drawingContext.fill();
  }

  // Calculate the midpoint between the last point and the new point
  const midX = (lastX + x) / 2;
  const midY = (lastY + y) / 2;

  // Draw a smooth Quadratic Bezier Curve
  drawingContext.beginPath();
  drawingContext.strokeStyle = currentDrawingColor;
  drawingContext.lineWidth = DRAWING_LINE_WIDTH;
  drawingContext.lineCap = "round";
  drawingContext.lineJoin = "round";

  // Move to the last point
  drawingContext.moveTo(lastX, lastY);

  // Draw a quadratic curve
  drawingContext.quadraticCurveTo(x, y, midX, midY);

  drawingContext.stroke();

  // Update lastX/lastY to the midpoint for the start of the next segment
  [lastX, lastY] = [midX, midY];
}

function stopDrawing() {
  isDrawing = false;
}

function clearCanvasAndResetPrompt() {
  // Clear the canvas immediately
  drawingContext.clearRect(
    0,
    0,
    drawingContext.canvas.width,
    drawingContext.canvas.height
  );
  document.getElementById("prediction-result").textContent = "";

  // Set to Ready state
  currentPromptIndex = (currentPromptIndex + 1) % DRAWING_CLASSES.length;
  updatePrompt();
  document.getElementById("app-status").textContent = "Ready to Draw";
}

function updatePrompt() {
  const promptEl = document.getElementById("app-prompt");
  const currentClass = DRAWING_CLASSES[currentPromptIndex];
  promptEl.innerHTML = `Draw a **${currentClass}** ✍️`;
}

// --- UI AND SELECTION LOGIC ---

// This function must be robust and called only after the layout is stable
function setupUI() {
  if (isUISetupComplete) return; // Only run once

  // Get the video container dimensions (fixed 640x480)
  const containerWidth = config.video.width;
  const containerHeight = config.video.height;

  // Get the selection overlay element
  const overlay = document.getElementById("selection-overlay");

  // Check if the overlay exists and is laid out
  if (!overlay) {
    console.error("UI Overlay not found!");
    return;
  }

  // Use getBoundingClientRect to calculate the dimensions of the overlay itself
  const overlayRect = overlay.getBoundingClientRect();
  const containerRect = document
    .getElementById("video-container")
    .getBoundingClientRect();

  // Collect all UI buttons
  uiElements = Array.from(
    document.querySelectorAll("#selection-overlay .ui-button")
  ).map((btn) => {
    // Calculate the button's position relative to the OVERLAY
    const btnRect = btn.getBoundingClientRect();

    // Final bounds are relative to the 640x480 canvas origin (0,0)
    return {
      element: btn,
      // X coord: button_x - container_x (since container is 640px wide)
      canvasXMin: btnRect.left - containerRect.left,
      canvasXMax: btnRect.right - containerRect.left,
      // Y coord: button_y - container_y
      canvasYMin: btnRect.top - containerRect.top,
      canvasYMax: btnRect.bottom - containerRect.top,
      color: btn.dataset.color,
      action: btn.dataset.action,
    };
  });

  // Attach click listener to the CLASSIFY button
  const classifyBtn = document.getElementById("classify-btn");
  if (classifyBtn) {
    classifyBtn.addEventListener("click", () => {
      document.getElementById("app-status").textContent =
        "Classifying Drawing...";
      classifyDrawing();
    });
  }

  // Attach keyboard shortcut: press 'd' to classify the current drawing
  if (!keyListenerAdded) {
    document.addEventListener("keydown", (ev) => {
      // Ignore modifier combos
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (!ev.key) return;
      if (ev.key.toLowerCase() === "d") {
        // Provide quick UI feedback and trigger classification
        document.getElementById("app-status").textContent = "Classifying (key: D)...";
        classifyDrawing();
      }
    });
    keyListenerAdded = true;
  }

  // Initialize color and highlight
  const initialColor = document.getElementById("white-btn");
  if (initialColor) {
    currentDrawingColor = initialColor.dataset.color || "white";
    initialColor.classList.add("selected");
  }

  isUISetupComplete = true;
}

function checkUISelection(indexCanvasX, indexCanvasY) {
  // Ensure setup is complete before checking
  if (!isUISetupComplete) return false;

  // 🌟 CRITICAL FIX: The indexCanvasX is the MIRRORED coordinate (videoWidth - originalX).
  // We must convert it back to the ORIGINAL coordinate system (0-640) 
  // to compare it against the static screen bounds (canvasXMin/Max) calculated in setupUI.
  const originalX = videoWidth - indexCanvasX;
  
  let actionTaken = false;

  uiElements.forEach((item) => {
    // Check if the ORIGINAL X coordinate is within the calculated static bounds.
    if (
      originalX > item.canvasXMin &&
      originalX < item.canvasXMax &&
      indexCanvasY > item.canvasYMin &&
      indexCanvasY < item.canvasYMax
    ) {
      // Selection is made: Trigger action
      if (item.action === "CLEAR") {
        clearCanvasAndResetPrompt();
      } else if (item.action === "CLASSIFY") {
        // We typically rely on the explicit CLASSIFY button click in this area
        // but allowing pinch selection here can also work.
        classifyDrawing();
      } else if (item.color) {
        currentDrawingColor = item.color;

        // Update CSS highlight
        document
          .querySelectorAll("#selection-overlay .ui-button")
          .forEach((b) => b.classList.remove("selected"));
        item.element.classList.add("selected");
      }
      document.getElementById("app-status").textContent = `Selected: ${
        item.action || item.color.toUpperCase()
      }`;
      actionTaken = true;
    }
  });
  return actionTaken;
}

// --- CLASSIFICATION LOGIC (REAL MODEL INTEGRATION) ---

async function classifyDrawing() {
  document.getElementById("app-status").textContent = "Classifying Drawing...";

  if (!classificationModel) {
    document.getElementById("prediction-result").textContent =
      "Error: Classification model not loaded. Please ensure model files are in './models/doodle-classifier/'.";
    setTimeout(clearCanvasAndResetPrompt, 1500);
    return;
  }

  const drawingCanvas = document.getElementById("drawing-canvas");

  // 1. Create a temporary canvas for resizing the drawing
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = CLASSIFICATION_SIZE;
  tempCanvas.height = CLASSIFICATION_SIZE;
  const tempCtx = tempCanvas.getContext("2d");

  // Draw the main drawing onto the temporary canvas (resizing/resampling)
  tempCtx.drawImage(drawingCanvas, 0, 0, tempCanvas.width, tempCanvas.height);

  let resultElement = document.getElementById("prediction-result");
  resultElement.textContent = "Processing...";

  // Use tf.tidy() to automatically clean up intermediate Tensors (best practice)
  tf.tidy(() => {
    // 2. Create and preprocess the tensor
    let tensor = tf.browser.fromPixels(tempCanvas, 1); // [H, W, 1] - Int32

    // Perform the operations step-by-step
    tensor = tensor.cast("float32"); // Convert to Float32
    tensor = tensor.div(255.0); // Normalize to 0-1 range
    tensor = tensor.expandDims(0); // Add batch dimension [1, H, W, 1]

    // 3. Run Prediction
    const prediction = classificationModel.predict(tensor);

    // Get the probabilities and find the highest one
    const values = prediction.dataSync();
    const predictedIndex = values.indexOf(Math.max(...values));
    const finalPrediction = DRAWING_CLASSES[predictedIndex]; // Use finalPrediction name

    const isCorrect = finalPrediction === DRAWING_CLASSES[currentPromptIndex];

    // Wait for prediction to be ready before displaying
    setTimeout(() => {
      if (isCorrect) {
        resultElement.textContent = `✅ Predicted: ${finalPrediction.toUpperCase()}. Correct!`;
      } else {
        resultElement.textContent = `❌ Predicted: ${finalPrediction.toUpperCase()}. Try drawing a ${DRAWING_CLASSES[
          currentPromptIndex
        ].toUpperCase()}.`;
      }
      // Clear canvas and set new prompt after feedback
      setTimeout(clearCanvasAndResetPrompt, 1000);
    }, 500); // Short delay for visual effect
  });

  // Manually remove the temporary canvas (optional)
  tempCanvas.remove();
}

function handleDrawingGesture(landmarks) {
  // Get landmark coordinates (video feed coordinates, 0-640/0-480)
  const indexTip = landmarks[indexFingerTipIndex];
  const thumbTip = landmarks[thumbTipIndex];

  // Calculate distances
  const pinchDistance = getDistance(indexTip, thumbTip);

  // The landmark coordinates are already normalized to 640x480
  const indexCanvasX = indexTip[0];
  const indexCanvasY = indexTip[1];

  // 1. --- Pinch/Selection Mode Logic (Pinch = Mode Toggle/Click) ---
  if (pinchDistance < PINCH_THRESHOLD) {
    // PINCH DETECTED: Stop drawing and check for UI selection (Click)
    stopDrawing();
    document.getElementById("app-status").textContent =
      "Selection Mode (Pinch Active)";
    isFirstSmoothingFrame = true; // Stop smoothing

    // Check if the pinch/click happened over a UI button
    checkUISelection(indexCanvasX, indexCanvasY);
  } else {
    // 2. --- Drawing Mode Logic (Default State: Index Finger Pointer) ---
    // NO PINCH DETECTED: Draw with the index finger.

    // IMPORTANT: Check if the index finger is NOT over the selection overlay
    // to prevent drawing lines through the buttons (approx Y > 420 for selection)
    if (indexCanvasY < videoHeight - 70) {
      // Apply EMA to smooth the raw index tip position
      const [smoothedDrawX, smoothedDrawY] = applyEMA(
        indexCanvasX,
        indexCanvasY
      );

      startDrawing(smoothedDrawX, smoothedY); // Use smoothed coordinates
      document.getElementById(
        "app-status"
      ).textContent = `Drawing Mode (${currentDrawingColor.toUpperCase()})`;
      isFirstSmoothingFrame = false;
    } else {
      stopDrawing(); // Stop drawing if the index finger is over the control panel area
      document.getElementById("app-status").textContent =
        "Hand Detected (Near Controls)";
      isFirstSmoothingFrame = true;
    }
  }
}

// --- MAIN EXECUTION ---
// index.js (Final Reliable Fix)

// --- A. Define the Custom IO Handler Class ---
// This class implements the IOHandler interface for tf.loadLayersModel.
// NOTE: Do NOT use `extends tf.io.IOHandler` in browser TFJS—IOHandler is
// a shape (interface) rather than a real base class. Instead provide an
// object/class with a `load()` method and pass its instance to
// `tf.loadLayersModel`.
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
    // The model.json produced by different converter versions can nest the
    // actual Keras model config in different places. Normalize the shape so
    // tf.loadLayersModel receives a proper `modelTopology` object.
    let modelTopology = null;

    if (modelJSON.modelTopology) {
      // tfjs-converter may nest Keras config under modelTopology.model_config
      // (snake_case) or modelTopology.modelConfig (camelCase). Prefer those
      // normalized objects when present.
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

    // Defensive injection: ensure the InputLayer has a batch input shape
    // (some converter variants use `batchShape` or `batch_input_shape`). We
    // attempt to find the InputLayer node and set both properties if missing.
    try {
      const layers =
        (modelTopology && modelTopology.config && modelTopology.config.layers) ||
        (modelTopology && modelTopology.config && modelTopology.config.layers) ||
        null;

      if (layers && layers.length > 0) {
        // Locate an InputLayer (or fallback to first layer). Support both
        // camelCase (`className`) and snake_case (`class_name`).
        const inputLayer =
          layers.find((l) => {
            const name = l.className || l.class_name || "";
            return name.toLowerCase().includes("input");
          }) || layers[0];

        const cfg = inputLayer.config || inputLayer.layer_config || null;
        // Normalize common snake_case -> expected tfjs fields
        if (cfg) {
          // If config provides `batch_shape` (snake_case), copy it to the
          // typical tfjs/keras keys so the loader recognizes the input shape.
          if (cfg.batch_shape && !cfg.batch_input_shape) {
            cfg.batch_input_shape = cfg.batch_shape;
          }
          if (cfg.batch_shape && !cfg.batchInputShape) {
            cfg.batchInputShape = cfg.batch_shape;
          }

          // If there's still no batch_input_shape or batchInputShape, inject one
          if (!cfg.batch_input_shape && !cfg.batchInputShape && !cfg.input_shape) {
            console.warn(
              `TFJS IOHandler: injecting batch input shape [null,${CLASSIFICATION_SIZE},${CLASSIFICATION_SIZE},1] into model topology`
            );
            cfg.batch_input_shape = [null, CLASSIFICATION_SIZE, CLASSIFICATION_SIZE, 1];
            cfg.batchInputShape = cfg.batch_input_shape;
          }

          // Also normalize layer naming if converter used snake_case `class_name`.
          if (inputLayer.class_name && !inputLayer.className) {
            inputLayer.className = inputLayer.class_name;
          }
        }
      }
    } catch (e) {
      console.warn("Failed to inject input shape (non-fatal)", e);
    }

    // If the original model.json used snake_case naming (common with
    // keras->tfjs converter), normalize a few well-known fields in-place so
    // tf.loadLayersModel receives the shape it expects. We RETURN the full
    // `modelJSON.modelTopology` object (not the nested model_config) because
    // tfjs expects that wrapper for layers-model format.
    try {
      const topology = modelJSON.modelTopology || modelTopology;

      // Normalize model_config key
      const mc = topology.model_config || topology.modelConfig || null;
      if (mc) {
        // class_name -> className
        if (mc.class_name && !mc.className) mc.className = mc.class_name;

        // Normalize input/output layer lists
        if (mc.config) {
          // layers
          const layers = mc.config.layers || mc.config.layers;
          if (Array.isArray(layers)) {
            layers.forEach((layer) => {
              // class_name -> className
              if (layer.class_name && !layer.className)
                layer.className = layer.class_name;

                // inbound_nodes -> inboundNodes
                if (layer.inbound_nodes && !layer.inboundNodes) {
                  try {
                    // Keras sometimes encodes inbound_nodes as an array of objects
                    // with `args`/`kwargs`. tfjs expects an array-of-arrays where
                    // each inner array contains tuples like [layerName, nodeIndex, tensorIndex].
                    const normalized = layer.inbound_nodes.map((nodeEntry) => {
                      // If nodeEntry already looks like the tfjs format (an array), keep it
                      if (Array.isArray(nodeEntry)) return nodeEntry;

                      // If it has an `args` array with `config.keras_history`, extract those
                      if (nodeEntry && Array.isArray(nodeEntry.args)) {
                        const mapped = nodeEntry.args.map((arg) => {
                          if (arg && arg.config && Array.isArray(arg.config.keras_history)) {
                            return arg.config.keras_history;
                          }
                          // Fallback: if arg is already a simple tuple/array
                          if (Array.isArray(arg)) return arg;
                          // Last-resort: try to construct from known fields
                          if (arg && arg.name) return [arg.name, 0, 0];
                          return null;
                        }).filter((x) => x != null);

                        return mapped;
                      }

                      // Unknown shape — fallback to original
                      return nodeEntry;
                    });

                    layer.inboundNodes = normalized;
                  } catch (e) {
                    // If normalization fails, copy original to avoid breaking loader
                    layer.inboundNodes = layer.inbound_nodes;
                  }
                }

              // normalize config.batch_shape -> batch_input_shape
              const lc = layer.config || layer.layer_config || {};
              if (lc.batch_shape && !lc.batch_input_shape) {
                lc.batch_input_shape = lc.batch_shape;
              }
              if (lc.batch_shape && !lc.batchInputShape) {
                lc.batchInputShape = lc.batch_shape;
              }
              // ensure the normalized config is written back
              layer.config = lc;
            });
          }

          // input_layers / output_layers -> inputLayers / outputLayers
          if (mc.config.input_layers && !mc.config.inputLayers)
            mc.config.inputLayers = mc.config.input_layers;
          if (mc.config.output_layers && !mc.config.outputLayers)
            mc.config.outputLayers = mc.config.output_layers;
        }
      }

      // Also normalize top-level arrays that some converters use
      if (topology.model_config && topology.model_config.config) {
        topology.model_config = topology.model_config; // no-op to indicate we've normalized
      }
    } catch (e) {
      console.warn("Normalization of modelTopology failed (non-fatal)", e);
    }

    // Return the full wrapper object as modelTopology (this matches
    // tfjs-converter output format: {keras_version, backend, model_config, ...})
    const artifacts = {
      modelTopology: modelJSON.modelTopology || modelTopology,
      weightsManifest: modelJSON.weightsManifest,
    };

    return artifacts;
  }
}

// --- B. The loadClassificationModel Function ---
async function loadClassificationModel() {
  document.getElementById("app-status").textContent =
    "Loading Classification Model...";

  // Create the custom IO handler using the path defined earlier
  const customHandler = new CustomUrlIOHandler(CLASSIFICATION_MODEL_PATH);

  try {
    // Use the custom handler with tf.loadLayersModel
    classificationModel = await tf.loadLayersModel(customHandler);

    console.log(
      "Classification Model loaded successfully via manual override."
    );
    document.getElementById("app-status").textContent =
      "Classification Model Loaded.";

    // Warm-up prediction
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
    document.getElementById("app-status").textContent = "Error Loading Model!";
    console.warn(
      `Could not load classification model from ${CLASSIFICATION_MODEL_PATH}. Using simulation.`
    );
    console.error("TensorFlow.js Model Load Error:", e);
  }
}

async function continouslyDetectLandmarks(video) {
  // Load Handpose model
  model = await handpose.load();
  document.getElementById("app-status").textContent =
    "Handpose Model Loaded. Ready to draw!";

  async function runDetection() {
    // --- Run setupUI here once to capture button bounds reliably ---
    if (!isUISetupComplete) {
      setupUI();
    }
    // --- END NEW ---

    // Clear overlay for fresh keypoints
    overlayContext.clearRect(
      0,
      0,
      overlayContext.canvas.width,
      overlayContext.canvas.height
    );

    const predictions = await model.estimateHands(video, true);

    if (predictions.length > 0) {
      // FIX: Use raw landmarks directly. The context flip handles mirroring.
      const landmarks = predictions[0].landmarks.map((kp) => [videoWidth - kp[0], kp[1]]);

      drawKeypoints(landmarks, overlayContext);
      handleDrawingGesture(landmarks);
    } else {
      stopDrawing();
      document.getElementById("app-status").textContent = "No Hand Detected";
      // NOTE: Reset EMA state when hand is completely gone
      isFirstSmoothingFrame = true;
    }

    requestAnimationFrame(runDetection);
  }

  runDetection();
}

async function main() {
  const video = await loadVideo();
  videoWidth = video.videoWidth;
  // FIX: videoHeight was using video.height, which is often 0. Use video.videoHeight.
  videoHeight = video.videoHeight;

  // 1. Initialize OVERLAY Canvas (for video feed and keypoints)
  const overlayCanvas = document.getElementById("overlay-canvas");
  overlayCanvas.width = videoWidth;
  overlayCanvas.height = videoHeight;
  overlayContext = overlayCanvas.getContext("2d");

  // FIX: Flip the overlay context ONCE for visual mirroring.
  overlayContext.translate(overlayCanvas.width, 0);
  overlayContext.scale(-1, 1);

  // 2. Initialize DRAWING Canvas (for persistent drawing)
  const drawingCanvas = document.getElementById("drawing-canvas");
  drawingCanvas.width = videoWidth;
  drawingCanvas.height = videoHeight;
  drawingContext = drawingCanvas.getContext("2d");

  // FIX: Flip the drawing context ONCE for drawing on the mirrored view.
  drawingContext.translate(drawingCanvas.width, 0);
  drawingContext.scale(-1, 1);

  // Add these lines for better drawing quality
  drawingContext.lineCap = "round";
  drawingContext.lineJoin = "round";

  // Set initial prompt
  updatePrompt();

  // Load models in parallel
  // ContinouslyDetectLandmarks will call runDetection which handles setupUI
  await Promise.all([
    loadClassificationModel(),
    continouslyDetectLandmarks(video),
  ]);
}

main();
