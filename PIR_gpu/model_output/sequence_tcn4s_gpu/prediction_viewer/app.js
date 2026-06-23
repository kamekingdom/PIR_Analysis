const BONES = [
  ["pelvis", "abdomen"], ["abdomen", "thorax"], ["thorax", "neck"], ["neck", "head"],
  ["pelvis", "l_thigh"], ["l_thigh", "l_shank"], ["l_shank", "l_foot"], ["l_foot", "l_toes"],
  ["pelvis", "r_thigh"], ["r_thigh", "r_shank"], ["r_shank", "r_foot"], ["r_foot", "r_toes"],
  ["thorax", "l_uarm"], ["l_uarm", "l_larm"], ["l_larm", "l_hand"],
  ["thorax", "r_uarm"], ["r_uarm", "r_larm"], ["r_larm", "r_hand"]
];

const state = {
  data: null,
  index: 0,
  playing: false,
  yaw: -0.62,
  pitch: 0.24,
  drag: null,
  lastTick: 0
};

const els = {
  status: document.getElementById("statusText"),
  canvas: document.getElementById("scene"),
  slider: document.getElementById("frameSlider"),
  play: document.getElementById("playBtn"),
  prev: document.getElementById("prevBtn"),
  next: document.getElementById("nextBtn"),
  export: document.getElementById("exportBtn"),
  trial: document.getElementById("trialText"),
  frame: document.getElementById("frameText"),
  error: document.getElementById("errorText"),
  worst: document.getElementById("worstText")
};

function fitCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * scale));
  const height = Math.max(1, Math.round(rect.height * scale));
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
  return { width: width / scale, height: height / scale, scale };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function fmt(value, digits = 1) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "-";
}

function jointsFromFlat(row) {
  const out = {};
  state.data.joints.forEach((joint, idx) => {
    out[joint] = row.slice(idx * 3, idx * 3 + 3);
  });
  return out;
}

function rotate(point) {
  const [x, y, z] = point;
  const cy = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const xYaw = x * cy - y * sy;
  const depthYaw = x * sy + y * cy;
  const zPitch = z * cp - depthYaw * sp;
  const depth = z * sp + depthYaw * cp;
  return { x: xYaw, y: -zPitch, depth };
}

function boundsCorners(bounds) {
  const b = bounds;
  return [
    [b.min[0], b.min[1], b.min[2]], [b.max[0], b.min[1], b.min[2]],
    [b.min[0], b.max[1], b.min[2]], [b.max[0], b.max[1], b.min[2]],
    [b.min[0], b.min[1], b.max[2]], [b.max[0], b.min[1], b.max[2]],
    [b.min[0], b.max[1], b.max[2]], [b.max[0], b.max[1], b.max[2]]
  ];
}

function makeProjection(bounds, rect) {
  const corners = boundsCorners(bounds).map(rotate);
  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = Math.min((rect.w - 80) / Math.max(1, maxX - minX), (rect.h - 90) / Math.max(1, maxY - minY));
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    scale
  };
}

function projectPoint(point, projection, rect) {
  const rotated = rotate(point);
  return {
    x: rect.x + rect.w / 2 + (rotated.x - projection.centerX) * projection.scale,
    y: rect.y + rect.h / 2 + (rotated.y - projection.centerY) * projection.scale,
    depth: rotated.depth
  };
}

function projectSets(trueJoints, predJoints, rect, projection) {
  const rotated = [];
  for (const points of [trueJoints, predJoints]) {
    const next = {};
    for (const [joint, point] of Object.entries(points)) {
      next[joint] = projectPoint(point, projection, rect);
    }
    rotated.push(next);
  }
  return rotated;
}

function drawBounds(ctx, bounds, projection, rect, label) {
  const corners = boundsCorners(bounds).map((point) => projectPoint(point, projection, rect));
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 3],
    [4, 5], [4, 6], [5, 7], [6, 7],
    [0, 4], [1, 5], [2, 6], [3, 7]
  ];
  ctx.strokeStyle = "rgba(24, 32, 38, 0.28)";
  ctx.lineWidth = 1.4;
  for (const [a, b] of edges) {
    ctx.beginPath();
    ctx.moveTo(corners[a].x, corners[a].y);
    ctx.lineTo(corners[b].x, corners[b].y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(24, 32, 38, 0.72)";
  ctx.font = "12px Segoe UI, sans-serif";
  ctx.textAlign = "left";
  const b = bounds;
  ctx.fillText(`${label}: X ${fmt(b.min[0], 0)}..${fmt(b.max[0], 0)} / Y ${fmt(b.min[1], 0)}..${fmt(b.max[1], 0)} / Z ${fmt(b.min[2], 0)}..${fmt(b.max[2], 0)} mm`, rect.x + 14, rect.y + rect.h - 14);
}

function drawSkeleton(ctx, points, color, width) {
  const links = BONES
    .filter(([a, b]) => points[a] && points[b])
    .map(([a, b]) => ({ a, b, depth: (points[a].depth + points[b].depth) / 2 }))
    .sort((a, b) => a.depth - b.depth);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (const link of links) {
    ctx.beginPath();
    ctx.moveTo(points[link.a].x, points[link.a].y);
    ctx.lineTo(points[link.b].x, points[link.b].y);
    ctx.stroke();
  }
  const joints = Object.values(points).sort((a, b) => a.depth - b.depth);
  ctx.fillStyle = color;
  for (const point of joints) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, width + 1.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawErrorLinks(ctx, truePoints, predPoints) {
  ctx.strokeStyle = "rgba(105, 116, 128, 0.5)";
  ctx.lineWidth = 1;
  for (const joint of state.data.joints) {
    if (!truePoints[joint] || !predPoints[joint]) continue;
    ctx.beginPath();
    ctx.moveTo(truePoints[joint].x, truePoints[joint].y);
    ctx.lineTo(predPoints[joint].x, predPoints[joint].y);
    ctx.stroke();
  }
}

function frameError(trueJoints, predJoints) {
  let sum = 0;
  let worst = { joint: "-", value: 0 };
  for (const joint of state.data.joints) {
    const a = trueJoints[joint];
    const b = predJoints[joint];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    sum += d;
    if (d > worst.value) worst = { joint, value: d };
  }
  return { mean: sum / state.data.joints.length, worst };
}

function relativeJoints(points) {
  const origin = points.pelvis || points[state.data.joints[0]];
  const out = {};
  for (const [joint, point] of Object.entries(points)) {
    out[joint] = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
  }
  return out;
}

function computeRelativeBounds() {
  const bounds = {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]
  };
  for (const row of state.data.frames) {
    for (const points of [relativeJoints(jointsFromFlat(row.true)), relativeJoints(jointsFromFlat(row.pred))]) {
      for (const point of Object.values(points)) {
        for (let axis = 0; axis < 3; axis += 1) {
          bounds.min[axis] = Math.min(bounds.min[axis], point[axis]);
          bounds.max[axis] = Math.max(bounds.max[axis], point[axis]);
        }
      }
    }
  }
  const padding = 120;
  return {
    min: bounds.min.map((value) => value - padding),
    max: bounds.max.map((value) => value + padding)
  };
}

function drawSceneGrid(ctx, rect) {
  ctx.strokeStyle = "#d8e0e6";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const y = rect.y + 48 + ((rect.h - 96) * i) / 8;
    ctx.beginPath();
    ctx.moveTo(rect.x + 24, y);
    ctx.lineTo(rect.x + rect.w - 24, y);
    ctx.stroke();
  }
}

function drawSceneTitle(ctx, rect, title) {
  ctx.fillStyle = "rgba(24, 32, 38, 0.86)";
  ctx.font = "700 15px Segoe UI, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(title, rect.x + 14, rect.y + 24);
}

function drawScene(ctx, trueJoints, predJoints, bounds, rect, title, boundsLabel) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.w, rect.h);
  ctx.clip();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  drawSceneGrid(ctx, rect);
  const projection = makeProjection(bounds, rect);
  const [truePoints, predPoints] = projectSets(trueJoints, predJoints, rect, projection);
  drawBounds(ctx, bounds, projection, rect, boundsLabel);
  drawErrorLinks(ctx, truePoints, predPoints);
  drawSkeleton(ctx, truePoints, "#1f5f9f", 4);
  drawSkeleton(ctx, predPoints, "#d87522", 3);
  drawSceneTitle(ctx, rect, title);
  ctx.restore();
}

function drawMetricsPanel(ctx, rect, row, err, statusText) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = "#d5dde3";
  ctx.lineWidth = 1;
  ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
  const items = [
    ["Trial", row.trial],
    ["Frame", String(row.frame)],
    ["Mean Joint Error", `${fmt(err.mean, 1)} mm`],
    ["Worst Joint", `${err.worst.joint} (${fmt(err.worst.value, 1)} mm)`]
  ];
  let y = rect.y + 34;
  for (const [label, value] of items) {
    ctx.fillStyle = "#63717b";
    ctx.font = "700 13px Segoe UI, sans-serif";
    ctx.fillText(label, rect.x + 16, y);
    ctx.fillStyle = "#182026";
    ctx.font = "700 22px Segoe UI, sans-serif";
    ctx.fillText(value, rect.x + 16, y + 30);
    y += 82;
    ctx.strokeStyle = "#d5dde3";
    ctx.beginPath();
    ctx.moveTo(rect.x + 16, y - 18);
    ctx.lineTo(rect.x + rect.w - 16, y - 18);
    ctx.stroke();
  }
  ctx.fillStyle = "#63717b";
  ctx.font = "13px Segoe UI, sans-serif";
  ctx.fillText(statusText, rect.x + 16, rect.y + rect.h - 22);
}

function drawFrame(ctx, width, height, includeMetrics = false) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const row = state.data.frames[state.index];
  const trueJoints = jointsFromFlat(row.true);
  const predJoints = jointsFromFlat(row.pred);
  const relTrueJoints = relativeJoints(trueJoints);
  const relPredJoints = relativeJoints(predJoints);
  const panelW = includeMetrics ? Math.max(250, Math.round(width * 0.18)) : 0;
  const sceneW = width - panelW;
  const gap = 10;
  const rightW = Math.max(220, Math.round((sceneW - gap) * 0.3));
  const leftW = sceneW - gap - rightW;
  const leftRect = { x: 0, y: 0, w: leftW, h: height };
  const rightRect = { x: leftW + gap, y: 0, w: rightW, h: height };
  drawScene(ctx, trueJoints, predJoints, state.data.true_bounds, leftRect, "Global coordinates", "Global bounds");
  drawScene(ctx, relTrueJoints, relPredJoints, state.data.relative_bounds, rightRect, "Pelvis-relative", "Relative bounds");

  ctx.fillStyle = "#eef2f4";
  ctx.fillRect(leftW, 0, gap, height);

  const err = frameError(trueJoints, predJoints);
  if (includeMetrics) {
    drawMetricsPanel(ctx, { x: sceneW, y: 0, w: panelW, h: height }, row, err, `${state.index + 1} / ${state.data.frames.length}`);
  }
  return { row, err };
}

function render() {
  if (!state.data) return;
  const { width, height, scale } = fitCanvas();
  const ctx = els.canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const { row, err } = drawFrame(ctx, width, height, false);

  els.trial.textContent = row.trial;
  els.frame.textContent = row.frame;
  els.error.textContent = `${fmt(err.mean, 1)} mm`;
  els.worst.textContent = `${err.worst.joint} (${fmt(err.worst.value, 1)} mm)`;
  els.status.textContent = `${state.index + 1} / ${state.data.frames.length}`;
  els.slider.value = String(state.index);
}

function videoMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];
  return candidates.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5000);
}

async function exportVideo() {
  if (!state.data || !window.MediaRecorder) {
    els.status.textContent = "このブラウザは動画書き出しに対応していません";
    return;
  }

  const previous = {
    index: state.index,
    playing: state.playing,
    yaw: state.yaw,
    pitch: state.pitch
  };
  const fps = 25;
  const frameMs = 1000 / fps;
  const chunks = [];
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = 1600;
  exportCanvas.height = 900;
  const exportCtx = exportCanvas.getContext("2d");
  const stream = exportCanvas.captureStream(fps);
  const mimeType = videoMimeType();
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

  state.playing = false;
  els.export.disabled = true;
  els.play.disabled = true;
  els.prev.disabled = true;
  els.next.disabled = true;
  els.slider.disabled = true;
  els.export.textContent = "Exporting";

  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };

  const finished = new Promise((resolve) => {
    recorder.onstop = resolve;
  });

  recorder.start();
  state.index = 0;
  render();
  drawFrame(exportCtx, exportCanvas.width, exportCanvas.height, true);

  let frame = 0;
  const timer = window.setInterval(() => {
    frame += 1;
    if (frame >= state.data.frames.length) {
      window.clearInterval(timer);
      recorder.stop();
      return;
    }
    state.index = frame;
    render();
    drawFrame(exportCtx, exportCanvas.width, exportCanvas.height, true);
    els.status.textContent = `書き出し中 ${frame + 1} / ${state.data.frames.length}`;
  }, frameMs);

  await finished;
  stream.getTracks().forEach((track) => track.stop());
  const blob = new Blob(chunks, { type: mimeType || "video/webm" });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  downloadBlob(blob, `pir_prediction_${stamp}.webm`);

  state.index = previous.index;
  state.playing = previous.playing;
  state.yaw = previous.yaw;
  state.pitch = previous.pitch;
  els.export.disabled = false;
  els.play.disabled = false;
  els.prev.disabled = false;
  els.next.disabled = false;
  els.slider.disabled = false;
  els.export.textContent = "Export";
  els.play.textContent = state.playing ? "Pause" : "Play";
  render();
}

async function load() {
  const res = await fetch("viewer_data.json");
  state.data = await res.json();
  state.data.relative_bounds = computeRelativeBounds();
  els.slider.max = String(state.data.frames.length - 1);
  render();
}

function step(delta) {
  state.index = clamp(state.index + delta, 0, state.data.frames.length - 1);
  render();
}

function tick(now) {
  if (state.playing && state.data) {
    const delta = state.lastTick ? now - state.lastTick : 0;
    if (delta > 40) {
      state.index = (state.index + 1) % state.data.frames.length;
      state.lastTick = now;
      render();
    }
  } else {
    state.lastTick = now;
  }
  requestAnimationFrame(tick);
}

els.slider.addEventListener("input", () => {
  state.index = Number(els.slider.value);
  render();
});
els.prev.addEventListener("click", () => step(-1));
els.next.addEventListener("click", () => step(1));
els.play.addEventListener("click", () => {
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "Pause" : "Play";
});
els.export.addEventListener("click", exportVideo);
els.canvas.addEventListener("pointerdown", (event) => {
  state.drag = { x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch };
  els.canvas.setPointerCapture?.(event.pointerId);
});
els.canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  state.yaw = state.drag.yaw + (event.clientX - state.drag.x) * 0.01;
  state.pitch = clamp(state.drag.pitch + (event.clientY - state.drag.y) * 0.006, -0.9, 0.9);
  render();
});
window.addEventListener("pointerup", () => { state.drag = null; });
window.addEventListener("resize", render);

load().catch((error) => {
  els.status.textContent = `読み込み失敗: ${error.message}`;
});
requestAnimationFrame(tick);
