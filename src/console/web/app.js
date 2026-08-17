"use strict";

const app = {
  state: null,
  selectedMode: "simulation",
  clearedBefore: 0,
  frameTick: 0,
  urIpInitialized: false,
  manusSettingsInitialized: false,
  manusCalibrationPolling: false,
  manusVisualPolling: false,
  backendMismatchWarned: false,
  teleopVisualPolling: false,
  teleopConfigMode: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = error ? "show error" : "show";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = ""; }, 2600);
}

async function api(path, options = {}) {
  const config = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    config.headers["Content-Type"] = "application/json";
    config.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, config);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function action(path, body = {}) {
  try {
    const payload = await api(path, { method: "POST", body });
    if (payload.state) render(payload.state);
    return payload;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

async function switchTab(name) {
  if (!await action("/api/console/tab", { tab: name, ...teleopInputSelection() })) return;
  $$(".tab").forEach(button => button.classList.toggle("active", button.dataset.tab === name));
  $$(".tab-page").forEach(page => {
    const active = page.id === `tab-${name}`;
    page.hidden = !active;
    page.classList.toggle("active", active);
  });
  if (name === "teleop") {
    pollTeleopVisual();
  }
  if (name === "manus") {
    requestAnimationFrame(drawManus);
    pollManusVisual();
  }
}

function statusCard(name, value, side = "") {
  const state = value?.state || "unknown";
  const label = value?.label || value?.state || "Unknown";
  const detail = value?.detail || "No detail";
  return `<div class="status-card ${state}"><span class="status-card-title">${escapeHtml(name)}</span><span class="metric-value">${escapeHtml(side || label)}</span><span class="metric-detail">${escapeHtml(detail)}</span></div>`;
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll('"', "&quot;");
}

function formatAge(value) {
  if (value === null || value === undefined) return "—";
  return value < 1 ? `${Math.round(value * 1000)} ms` : `${value.toFixed(1)} s`;
}

function hardwareCard(name, value) {
  const state = value?.state || "unknown";
  const label = value?.label || "Unknown";
  const detail = value?.detail || "No detail";
  const metadata = Object.entries(value?.metadata || {})
    .map(([key, item]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(item)}</dd></div>`)
    .join("");
  return `<article class="hardware-card ${state}"><div class="hardware-card-head"><span class="hardware-name">${escapeHtml(name)}</span><span class="hardware-state">${escapeHtml(label)}</span></div><div class="hardware-detail">${escapeHtml(detail)}</div><dl class="hardware-metadata">${metadata}</dl></article>`;
}

function renderSystem(state) {
  const system = state.system;
  const names = { ur: "UR5", ryhand: "RYHand", manus: "Manus", vive: "Vive" };
  $("#system-status-cards").innerHTML = Object.entries(names)
    .map(([key, name]) => statusCard(name, system[key]))
    .join("") + (system.cameras || []).map(camera => statusCard(camera.name, camera, "RGB")).join("");
  $("#hardware-inventory").innerHTML = Object.entries(names)
    .map(([key, name]) => hardwareCard(name, system[key]))
    .join("") + (system.cameras || []).map(camera => hardwareCard(camera.name, camera)).join("");
  $("#scan-time").textContent = system.updated_at
    ? `Checked ${new Date(system.updated_at * 1000).toLocaleTimeString()}`
    : "Not checked";
  if (!app.urIpInitialized) {
    $("#ur-ip").value = state.ur_ip || "";
    app.urIpInitialized = true;
  }

  const cameras = state.cameras || [];
  const strip = $("#camera-strip");
  const select = $("#camera-select");
  if (!cameras.length) {
    strip.className = "camera-strip empty";
    strip.textContent = "No RGB cameras configured";
    select.innerHTML = '<option value="">No cameras configured</option>';
  } else {
    strip.className = "camera-strip";
    strip.innerHTML = cameras.map(camera => `<span class="camera-chip">RGB · ${escapeHtml(camera.name)}</span>`).join("");
    const current = select.value;
    select.innerHTML = cameras.map(camera => `<option value="${escapeHtml(camera.serial)}">${escapeHtml(camera.name)} · ${escapeHtml(camera.serial)}</option>`).join("");
    if (cameras.some(camera => camera.serial === current)) select.value = current;
  }
  const cameraLive = state.camera.active && state.camera.frame_ready;
  $("#camera-preview-wrap").classList.toggle("live", cameraLive);
  if (cameraLive) $("#camera-preview").src = `/api/camera/frame?t=${app.frameTick}`;
}

function batteryState(item) {
  const raw = item?.batteryPercentage;
  if (raw === null || raw === undefined || raw === "") return ["—", "unknown"];
  const value = Number(raw);
  if (!item?.connected || !Number.isFinite(value) || value < 0 || value > 100) return ["—", "unknown"];
  return [`${Math.round(value)}%`, value <= 10 ? "critical" : value <= 30 ? "low" : "good"];
}

function renderManus(state) {
  const manus = state.manus || {
    active: false,
    state: "unavailable",
    detail: "Restart ssr-console to load the MANUS monitor backend",
    sdkVersion: "3.1.1",
    settings: {},
    calibration: { active: false, inProgress: false },
    hands: { left: { connected: false }, right: { connected: false } },
  };
  const hands = manus.hands || {};
  $("#manus-visual-state").textContent = manus.destination === "manus" ? manus.state : `Routed to ${manus.destination || "system"}`;
  ["left", "right"].forEach(hand => {
    const [label, batteryClass] = batteryState(hands[hand]);
    $(`#manus-${hand}-battery`).textContent = label;
    $(`#manus-${hand}-battery-badge`).className = `manus-battery ${batteryClass}`;
  });
  const connected = ["left", "right"].filter(hand => hands[hand]?.connected);
  $("#manus-offline").style.display = connected.length ? "none" : "grid";
  $("#manus-status").innerHTML = [
    statusCard("Bridge", { state: manus.state === "running" ? "connected" : manus.state, detail: manus.detail }, manus.state),
    statusCard("SDK", { state: manus.active ? "connected" : "unknown", detail: `Integrated MANUS Core ${manus.sdkVersion || "3.1.1"}` }, manus.settings?.coreVersion || manus.sdkVersion || "3.1.1"),
    ...["left", "right"].map(hand => {
      const item = hands[hand] || {};
      const family = item.deviceFamily?.name || "Family unknown";
      const detail = item.gloveId
        ? `ID ${item.gloveId} · ${item.boneCount || 0} bones · ${Number(item.fps || 0).toFixed(1)} Hz · ${item.ageMs == null ? "—" : Number(item.ageMs).toFixed(0) + " ms"} · ${family}`
        : "No glove detected";
      return statusCard(`${hand} glove`, { state: item.connected ? "connected" : "offline", detail }, item.calibrating ? "Calibrating" : item.connected ? "Connected" : "Disconnected");
    }),
    statusCard("Coordinates", { state: manus.active ? "connected" : "unknown", detail: "Raw SDK skeleton · OpenVR meters" }, "LH · Y-up · world"),
  ].join("");

  if (!app.manusSettingsInitialized && Object.keys(manus.settings || {}).length) {
    const settings = manus.settings;
    $("#manus-motion").value = String(settings.handMotion ?? 4);
    $("#manus-pinch").checked = Boolean(settings.pinchCompensation);
    $("#manus-casing").value = String(Number(settings.casingCompensation ?? 0));
    $("#manus-casing-value").textContent = Number(settings.casingCompensation ?? 0).toFixed(2);
    $("#manus-settings-note").textContent = `SDK ${settings.sdkVersion || manus.sdkVersion || "3.1.1"} · Core ${settings.coreVersion || "unknown"}`;
    app.manusSettingsInitialized = true;
  }

  const selectedHand = $("#manus-calibration-hand").value;
  const selected = hands[selectedHand] || {};
  $("#manus-calibration-family").textContent = selected.deviceFamily?.name || "Waiting…";
  const calibration = manus.calibration || {};
  const active = Boolean(calibration.active);
  const inProgress = Boolean(calibration.inProgress);
  const ownsControl = manus.destination === "manus";
  const tunables = selected.calibrationTunables || {};
  $("#manus-settings-load").disabled = !manus.active || !ownsControl || active;
  $("#manus-settings-apply").disabled = !manus.active || !ownsControl || !app.manusSettingsInitialized || active;
  $("#manus-motion").disabled = active;
  $("#manus-pinch").disabled = active || !tunables.pinchCompensation;
  $("#manus-casing").disabled = active || !tunables.casingCompensation;
  const stepCount = Number(calibration.stepCount || 0);
  const completed = Number(calibration.completedStepIndex ?? -1);
  const nextStep = Math.min(stepCount, completed + 1);
  const instructions = ["Flat reference", "Hold fist", "Fingertips to palm"];
  if (active) {
    $("#manus-calibration-title").textContent = inProgress
      ? `Sampling step ${Number(calibration.stepIndex || 0) + 1} of ${stepCount}`
      : completed >= stepCount - 1
        ? "Ready to save"
        : `Step ${nextStep + 1} of ${stepCount}: ${instructions[nextStep] || "Follow SDK pose"}`;
    $("#manus-calibration-copy").textContent = calibration.error || calibration.description || (inProgress ? "Hold the requested pose until sampling completes." : "Press Next Step when the pose is ready.");
  } else {
    $("#manus-calibration-title").textContent = calibration.saved ? "Calibration saved" : calibration.cancelled ? "Calibration cancelled" : "Calibration idle";
    $("#manus-calibration-copy").textContent = calibration.saved ? "MANUS Core accepted and stored the glove calibration." : "Select a connected glove, then start the official MANUS calibration pipeline.";
  }
  const duration = Number(calibration.samplingDuration || 0);
  const elapsed = Number(calibration.elapsedSeconds || 0);
  $("#manus-calibration-progress").value = inProgress && duration > 0 ? Math.min(100, elapsed / duration * 100) : completed >= stepCount - 1 && stepCount > 0 ? 100 : 0;
  $("#manus-calibration-hand").disabled = active;
  $("#manus-calibration-start").disabled = !manus.active || !ownsControl || active || !selected.connected;
  $("#manus-calibration-next").disabled = !ownsControl || !active || inProgress || stepCount === 0 || completed >= stepCount - 1;
  $("#manus-calibration-save").disabled = !ownsControl || !active || inProgress || completed < stepCount - 1;
  $("#manus-calibration-cancel").disabled = !ownsControl || !active || inProgress;
  $("#manus-start").disabled = manus.active;
  $("#manus-stop").disabled = !manus.active;
  if (inProgress && ownsControl) pollManusCalibration();
  drawManus();
  pollManusVisual();
}

function sliderGroup(name, values) {
  const specs = [
    ["scale", "Scale", .1, 3, .01],
    ["x", "X", -.2, .2, .001],
    ["y", "Y", -.2, .2, .001],
    ["z", "Z", -.2, .2, .001],
  ];
  return `<div class="slider-group" data-finger="${name}"><h3>${name}</h3>${specs.map(([key, label, min, max, step]) => `<label class="slider-line"><span>${label}</span><input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${values[key]}"><output>${Number(values[key]).toFixed(key === "scale" ? 2 : 3)}</output></label>`).join("")}</div>`;
}

function calibrationValues() {
  const result = {};
  $$(".slider-group").forEach(group => {
    const values = {};
    $$('input[type="range"]', group).forEach(input => { values[input.dataset.key] = Number(input.value); });
    result[group.dataset.finger] = values;
  });
  return result;
}

function renderCalibration(state) {
  const calibration = state.calibration;
  if (!$("#calibration-sliders").children.length) {
    $("#calibration-sliders").innerHTML = sliderGroup("thumb", calibration.thumb) + sliderGroup("index", calibration.index);
    $$('#calibration-sliders input[type="range"]').forEach(input => input.addEventListener("input", () => {
      input.nextElementSibling.textContent = Number(input.value).toFixed(input.dataset.key === "scale" ? 2 : 3);
    }));
  }
  $("#calibration-visual-state").textContent = calibration.state;
  const stage = $(".calibration-stage");
  stage.classList.toggle("live", calibration.active && calibration.frame_ready);
  if (calibration.active && calibration.frame_ready) $("#calibration-frame").src = `/api/calibration/frame?t=${app.frameTick}`;
  const angles = calibration.angles || new Array(15).fill(0);
  $("#calibration-angle-bars").innerHTML = angles.map((angle, index) => {
    const value = Math.max(0, Math.min(1.57, Math.abs(angle)));
    return `<progress class="angle-bar ${index >= 6 ? "disabled" : ""}" max="1.57" value="${value}" title="Joint ${index + 1}: ${angle.toFixed(3)} rad"></progress>`;
  }).join("");
  $("#calibration-status").innerHTML = [
    statusCard("Session", { state: calibration.state, detail: calibration.detail }, calibration.state),
    statusCard("Manus", { state: calibration.active ? "connected" : "unknown", detail: `${calibration.side} glove` }),
    statusCard("PyBullet", { state: calibration.frame_ready ? "connected" : "unknown", detail: "Official RYHand URDF" }),
    statusCard("Physical output", { state: calibration.live_output && calibration.active ? "connected" : "simulated", detail: calibration.live_output ? "RYHand commands enabled" : "Simulation only" }),
  ].join("");
}

const teleopModeDescriptions = {
  full: "Vive → physical UR5 and calibrated MANUS skeleton → physical RYHand.",
  arm: "Vive controls the physical UR5; MANUS and RYHand are not used.",
  hand: "Calibrated MANUS skeleton controls the physical RYHand; Vive and UR are not used.",
  simulation: "Vive and MANUS independently drive virtual UR and RYHand outputs.",
};

const teleopConfigFields = {
  ur_ip: { label: "UR IPv4", type: "text", wide: true },
  can_port: { label: "CAN interface", type: "text" },
  translation_scale: { label: "Translation scale", type: "number", min: .05, max: 3, step: .05 },
  servo_speed: { label: "servoL speed", type: "number", min: .01, max: 2, step: .01 },
  servo_acceleration: { label: "servoL acceleration", type: "number", min: .01, max: 5, step: .01 },
  update_rate: { label: "Update rate (Hz)", type: "number", min: 10, max: 250, step: 1 },
  input_timeout: { label: "Freshness timeout (s)", type: "number", min: .05, max: 2, step: .01 },
  hand_motor_speed: { label: "RYHand speed", type: "number", min: 1, max: 65535, step: 1 },
  max_linear_speed: { label: "Max linear speed", type: "number", min: .01, max: 1, step: .01 },
  max_angular_speed: { label: "Max angular speed", type: "number", min: .05, max: 5, step: .05 },
};

const modeConfigKeys = {
  full: ["ur_ip", "can_port", "translation_scale", "servo_speed", "servo_acceleration", "input_timeout", "hand_motor_speed"],
  arm: ["ur_ip", "translation_scale", "servo_speed", "servo_acceleration", "max_linear_speed", "max_angular_speed"],
  hand: ["can_port", "update_rate", "input_timeout", "hand_motor_speed"],
  simulation: ["translation_scale", "update_rate", "input_timeout", "hand_motor_speed", "max_linear_speed", "max_angular_speed"],
};

function renderTeleopConfig(config) {
  if (app.teleopConfigMode === app.selectedMode && $("#teleop-config-fields").children.length) return;
  app.teleopConfigMode = app.selectedMode;
  $("#teleop-mode-description").textContent = teleopModeDescriptions[app.selectedMode] || "";
  $("#teleop-config-fields").innerHTML = (modeConfigKeys[app.selectedMode] || []).map(key => {
    const field = teleopConfigFields[key];
    const limits = field.type === "number" ? ` min="${field.min}" max="${field.max}" step="${field.step}"` : "";
    return `<div class="ctl-field ${field.wide ? "wide" : ""}"><label for="teleop-config-${key}">${escapeHtml(field.label)}</label><input id="teleop-config-${key}" data-config-key="${key}" data-config-type="${field.type}" type="${field.type}" value="${escapeAttribute(config?.[key] ?? "")}"${limits}></div>`;
  }).join("");
}

function renderModes(modes, config) {
  const select = $("#teleop-mode");
  if (!select.options.length) select.innerHTML = modes.map(mode => `<option value="${mode.id}">${escapeHtml(mode.label)}</option>`).join("");
  select.value = app.selectedMode;
  renderTeleopConfig(config);
}

function selectedTeleopConfig() {
  const values = {};
  $$("#teleop-config-fields [data-config-key]").forEach(input => {
    values[input.dataset.configKey] = input.dataset.configType === "number" ? Number(input.value) : input.value.trim();
  });
  return values;
}

function teleopInputSelection() {
  return {
    vive_side: $("#teleop-vive-side").value,
    manus_side: $("#teleop-manus-side").value,
  };
}

function renderTeleop(state) {
  const teleop = state.teleop;
  renderModes(state.teleop_modes, state.teleop_config);
  $("#teleop-visual-state").textContent = teleop.state;
  const selectedMode = state.teleop_modes.find(mode => mode.id === app.selectedMode);
  $("#visual-mode").textContent = teleop.active ? teleop.mode_label : `${selectedMode?.label || "Teleop"} preview`;
  const clutch = $("#visual-clutch");
  clutch.textContent = teleop.arm_tracking ? "TRACKING" : "CLUTCH ENGAGED";
  clutch.classList.toggle("tracking", teleop.arm_tracking);
  const translation = teleop.motion_delta?.translation_m || [0, 0, 0];
  const rotation = teleop.motion_delta?.rotation_rad || [0, 0, 0];
  [
    ["x", translation[0], 1000], ["y", translation[1], 1000], ["z", translation[2], 1000],
    ["rx", rotation[0], 180 / Math.PI], ["ry", rotation[1], 180 / Math.PI], ["rz", rotation[2], 180 / Math.PI],
  ].forEach(([axis, raw, scale]) => {
    const scaled = Number(raw) * scale;
    const value = Number.isFinite(scaled) && Math.abs(scaled) >= .005 ? scaled : 0;
    $("#teleop-delta-" + axis).textContent = value.toFixed(2);
  });
  $("#vive-age").textContent = formatAge(teleop.vive_age);
  $("#manus-age").textContent = formatAge(teleop.manus_age);
  $("#teleop-toggle").textContent = teleop.arm_tracking ? "Engage clutch" : "Release clutch";
  $("#teleop-status").innerHTML = [
    statusCard("Controller", { state: teleop.state, detail: teleop.detail }, teleop.state),
    statusCard("Mode", { state: app.selectedMode === "simulation" ? "simulated" : (teleop.active ? "connected" : "preview"), detail: teleop.active ? teleop.mode_label : `${selectedMode?.label || "Teleop"} selected` }),
    statusCard("Vive input", { state: teleop.vive_age === null ? "unknown" : (teleop.vive_age < .25 ? "connected" : "offline"), detail: `Age ${formatAge(teleop.vive_age)}` }),
    statusCard("Manus input", { state: teleop.manus_age === null ? "unknown" : (teleop.manus_age < .25 ? "connected" : "offline"), detail: `Age ${formatAge(teleop.manus_age)}` }),
  ].join("");
  pollTeleopVisual();
}

function transformedManusPoint(position) {
  const source = [-Number(position[0]), Number(position[1]), Number(position[2])];
  const yaw = Number($("#manus-view-yaw").value) * Math.PI / 180 + Math.PI;
  const pitch = Number($("#manus-view-pitch").value) * Math.PI / 180;
  const x = source[0] * Math.cos(yaw) - source[2] * Math.sin(yaw);
  const z = source[0] * Math.sin(yaw) + source[2] * Math.cos(yaw);
  return [x, source[1] * Math.cos(pitch) - z * Math.sin(pitch), z];
}

function drawManusAxes(ctx, viewportX, viewportWidth, height) {
  if (!$("#manus-show-axes").checked) return;
  const origin = { x: viewportX + 34, y: height - 34 };
  [["X", [1, 0, 0], "#ff6969"], ["Y", [0, 1, 0], "#69e58d"], ["Z", [0, 0, 1], "#6ca8ff"]]
    .map(([label, axis, color]) => [label, transformedManusPoint(axis), color])
    .sort((a, b) => b[1][2] - a[1][2])
    .forEach(([label, axis, color]) => {
      const x = origin.x + axis[0] * 23, y = origin.y - axis[1] * 23;
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = color; ctx.font = "700 10px ui-monospace, monospace";
      ctx.fillText(label, x + 4, y);
    });
}

function drawManusHand(ctx, hand, item, viewportX, viewportWidth, height) {
  const bones = (item?.frame?.bones || []).filter(bone =>
    Array.isArray(bone.rawPos) && bone.rawPos.length === 3 && bone.rawPos.every(Number.isFinite)
  );
  if (!bones.length) return;
  const points = bones.map(bone => ({ bone, point: transformedManusPoint(bone.rawPos) }));
  const minX = Math.min(...points.map(item => item.point[0]));
  const maxX = Math.max(...points.map(item => item.point[0]));
  const minY = Math.min(...points.map(item => item.point[1]));
  const maxY = Math.max(...points.map(item => item.point[1]));
  const centerX = (minX + maxX) / 2, centerY = (minY + maxY) / 2;
  const zoom = Number($("#manus-view-zoom").value);
  const scale = Math.min(
    viewportWidth * .72 / Math.max(.03, maxX - minX),
    height * .72 / Math.max(.03, maxY - minY),
  ) * zoom;
  const projected = new Map();
  points.forEach(({ bone, point }) => projected.set(Number(bone.nodeId), {
    x: viewportX + viewportWidth / 2 + (point[0] - centerX) * scale,
    y: height / 2 - (point[1] - centerY) * scale,
    z: point[2],
    bone,
  }));
  const color = hand === "left" ? "#65d9e6" : "#f5d56d";
  const edges = [];
  projected.forEach(child => {
    const parent = projected.get(Number(child.bone.parentId));
    if (parent && parent !== child) edges.push({ parent, child, z: (parent.z + child.z) / 2 });
  });
  edges.sort((a, b) => b.z - a.z).forEach(({ parent, child }) => {
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(0,0,0,.82)"; ctx.lineWidth = 11;
    ctx.beginPath(); ctx.moveTo(parent.x, parent.y); ctx.lineTo(child.x, child.y); ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 6; ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,.48)"; ctx.lineWidth = 1.5; ctx.stroke();
  });
  [...projected.values()].sort((a, b) => b.z - a.z).forEach(point => {
    const gradient = ctx.createRadialGradient(point.x - 2, point.y - 2, 1, point.x, point.y, 6);
    gradient.addColorStop(0, "#fff"); gradient.addColorStop(.25, color); gradient.addColorStop(1, "#11110f");
    ctx.fillStyle = gradient; ctx.beginPath(); ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2); ctx.fill();
  });
}

function drawManus() {
  const canvas = $("#manus-canvas");
  if (!canvas || canvas.closest(".tab-page").hidden) return;
  const hands = app.state?.manus?.hands || {};
  const connected = ["left", "right"].filter(hand => hands[hand]?.connected);
  const single = connected.length === 1 ? connected[0] : null;
  const stage = canvas.closest(".manus-stage");
  stage.classList.toggle("single-hand", Boolean(single));
  stage.classList.toggle("single-left", single === "left");
  stage.classList.toggle("single-right", single === "right");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (single) {
    drawManusHand(ctx, single, hands[single], 0, rect.width, rect.height);
    drawManusAxes(ctx, 0, rect.width, rect.height);
  } else if (connected.length === 2) {
    ctx.strokeStyle = "rgba(237,224,201,.13)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(rect.width / 2, 0); ctx.lineTo(rect.width / 2, rect.height); ctx.stroke();
    drawManusHand(ctx, "left", hands.left, 0, rect.width / 2, rect.height);
    drawManusHand(ctx, "right", hands.right, rect.width / 2, rect.width / 2, rect.height);
    drawManusAxes(ctx, 0, rect.width / 2, rect.height);
    drawManusAxes(ctx, rect.width / 2, rect.width / 2, rect.height);
  }
}

async function pollManusCalibration() {
  if (app.manusCalibrationPolling) return;
  app.manusCalibrationPolling = true;
  try {
    await action("/api/manus/calibration/status");
  } finally {
    app.manusCalibrationPolling = false;
  }
}

async function pollManusVisual() {
  if (app.manusVisualPolling || !app.state?.manus?.active || $("#tab-manus").hidden) return;
  app.manusVisualPolling = true;
  try {
    app.state.manus = await api("/api/manus/visual");
    renderManus(app.state);
  } catch (error) {
    toast(`MANUS visual disconnected: ${error.message}`, true);
  } finally {
    app.manusVisualPolling = false;
    if (app.state?.manus?.active && !$("#tab-manus").hidden) setTimeout(pollManusVisual, 50);
  }
}

function renderLogs(logs) {
  const visible = logs.slice(app.clearedBefore);
  const html = visible.map(entry => `<div class="log-line ${entry.level}"><span class="time">${escapeHtml(entry.time)}</span><span class="source">${escapeHtml(entry.source)}</span><span class="message">${escapeHtml(entry.message)}</span></div>`).join("");
  $$('[data-terminal]').forEach(terminal => {
    const atBottom = terminal.scrollTop + terminal.clientHeight >= terminal.scrollHeight - 14;
    terminal.innerHTML = html || '<div class="log-line"><span class="time">--:--:--</span><span class="source">console</span><span class="message">No events in this view</span></div>';
    if (atBottom) terminal.scrollTop = terminal.scrollHeight;
  });
}

function render(state) {
  app.state = state;
  app.frameTick += 1;
  if (!state.manus && !app.backendMismatchWarned) {
    app.backendMismatchWarned = true;
    toast("Console backend is outdated; restart ssr-console", true);
  }
  renderSystem(state);
  renderManus(state);
  renderCalibration(state);
  renderTeleop(state);
  renderLogs(state.logs);
  const global = $("#global-state");
  global.className = "state-pill";
  if (state.teleop.arm_tracking) {
    global.classList.add("running");
    global.textContent = "tracking";
  } else if (state.teleop.state === "fault" || state.calibration.state === "fault" || state.manus?.state === "fault" || state.manus?.state === "error") {
    global.classList.add("error");
    global.textContent = "error";
  } else if (state.teleop.active || state.calibration.active || state.manus?.active) {
    global.classList.add("running");
    global.textContent = "running";
  } else {
    global.textContent = "idle";
  }
}

async function pollTeleopVisual() {
  if (app.teleopVisualPolling || !(app.state?.teleop?.active || app.state?.teleop?.preview_active) || $("#tab-teleop").hidden) return;
  app.teleopVisualPolling = true;
  try {
    app.state.teleop = await api("/api/teleop/visual");
    renderTeleop(app.state);
  } catch (error) {
    toast(`Teleop visual disconnected: ${error.message}`, true);
  } finally {
    app.teleopVisualPolling = false;
    if ((app.state?.teleop?.active || app.state?.teleop?.preview_active) && !$("#tab-teleop").hidden) setTimeout(pollTeleopVisual, 50);
  }
}

async function poll() {
  try {
    const state = await api("/api/state");
    render(state);
    const visibleTab = $(".tab-page.active")?.id.replace("tab-", "");
    if (visibleTab && state.active_tab !== visibleTab) {
      await action("/api/console/tab", { tab: visibleTab, ...teleopInputSelection() });
    }
  } catch (error) {
    toast(`Console disconnected: ${error.message}`, true);
  } finally {
    setTimeout(poll, 500);
  }
}

function bind() {
  $$(".tab").forEach(button => button.addEventListener("click", () => switchTab(button.dataset.tab)));
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("huawei-console-theme", root.dataset.theme);
    $("#theme-toggle").textContent = root.dataset.theme === "dark" ? "Light" : "Dark";
    $("#theme-toggle").setAttribute("aria-pressed", String(root.dataset.theme === "light"));
    drawManus();
  });
  document.documentElement.dataset.theme = localStorage.getItem("huawei-console-theme") || "dark";
  $("#theme-toggle").textContent = document.documentElement.dataset.theme === "dark" ? "Light" : "Dark";
  $("#theme-toggle").setAttribute("aria-pressed", String(document.documentElement.dataset.theme === "light"));
  $$(".clear-log").forEach(button => button.addEventListener("click", () => {
    app.clearedBefore = app.state?.logs.length || 0;
    renderLogs(app.state?.logs || []);
  }));
  $("#scan-system").addEventListener("click", async () => {
    const button = $("#scan-system"); button.disabled = true; button.textContent = "Checking…";
    try { render(await api("/api/system/scan")); toast("Read-only scan complete"); }
    catch (error) { toast(error.message, true); }
    finally { button.disabled = false; button.textContent = "Refresh status"; }
  });
  $("#camera-start").addEventListener("click", () => action("/api/camera/start", { serial: $("#camera-select").value }));
  $("#camera-stop").addEventListener("click", () => action("/api/camera/stop"));
  $("#ur-connect").addEventListener("click", async () => {
    const button = $("#ur-connect");
    button.disabled = true;
    button.textContent = "Connecting…";
    const succeeded = await action("/api/system/ur/connect", { ip: $("#ur-ip").value });
    if (succeeded) {
      const ur = app.state.system.ur;
      toast(ur.detail, ur.state !== "connected");
    }
    button.disabled = false;
    button.textContent = "Connect UR";
  });
  $("#ryhand-init").addEventListener("click", async () => {
    const button = $("#ryhand-init");
    button.disabled = true;
    button.textContent = "Initializing…";
    if (await action("/api/system/ryhand/init")) toast("RYHand CAN initialized");
    button.disabled = false;
    button.textContent = "Run RYHand Init";
  });
  $("#manus-start").addEventListener("click", async () => {
    app.manusSettingsInitialized = false;
    if (await action("/api/manus/start")) toast("Shared MANUS stream started");
  });
  $("#manus-stop").addEventListener("click", async () => {
    if (await action("/api/manus/stop")) toast("Shared MANUS stream stopped");
  });
  ["yaw", "pitch", "zoom"].forEach(name => {
    const input = $(`#manus-view-${name}`), output = $(`#manus-view-${name}-value`);
    input.addEventListener("input", () => {
      output.textContent = name === "zoom" ? `${Number(input.value).toFixed(2)}×` : `${input.value}°`;
      drawManus();
    });
  });
  $("#manus-show-axes").addEventListener("change", drawManus);
  $("#manus-casing").addEventListener("input", event => { $("#manus-casing-value").textContent = Number(event.target.value).toFixed(2); });
  $("#manus-settings-load").addEventListener("click", async () => {
    app.manusSettingsInitialized = false;
    if (await action("/api/manus/settings")) toast("MANUS SDK settings loaded");
  });
  $("#manus-settings-apply").addEventListener("click", async () => {
    app.manusSettingsInitialized = false;
    const values = { handMotion: Number($("#manus-motion").value), pinchCompensation: $("#manus-pinch").checked, casingCompensation: Number($("#manus-casing").value) };
    if (await action("/api/manus/settings/apply", values)) toast("MANUS SDK settings applied");
  });
  $("#manus-calibration-hand").addEventListener("change", () => { if (app.state) renderManus(app.state); });
  $("#manus-calibration-start").addEventListener("click", () => action("/api/manus/calibration/start", { hand: $("#manus-calibration-hand").value }));
  $("#manus-calibration-next").addEventListener("click", () => action("/api/manus/calibration/step"));
  $("#manus-calibration-save").addEventListener("click", async () => { if (await action("/api/manus/calibration/finish")) toast("Official MANUS calibration saved"); });
  $("#manus-calibration-cancel").addEventListener("click", () => action("/api/manus/calibration/cancel"));
  $("#calibration-start").addEventListener("click", () => action("/api/calibration/start", { use_right: $("#calibration-side").value === "right", live_output: $("#calibration-live").checked }));
  $("#calibration-stop").addEventListener("click", () => action("/api/calibration/stop"));
  $("#calibration-apply").addEventListener("click", async () => { if (await action("/api/calibration/update", calibrationValues())) toast("Calibration applied to simulation"); });
  $("#calibration-save").addEventListener("click", async () => {
    if (!await action("/api/calibration/update", calibrationValues())) return;
    if (await action("/api/calibration/save")) toast("Calibration saved atomically");
  });
  $("#teleop-mode").addEventListener("change", event => {
    app.selectedMode = event.target.value;
    app.teleopConfigMode = null;
    if (app.state) renderTeleop(app.state);
  });
  $("#teleop-config-save").addEventListener("click", async () => {
    if (await action("/api/teleop/config", selectedTeleopConfig())) toast("Teleoperation configuration saved");
  });
  ["#teleop-vive-side", "#teleop-manus-side"].forEach(selector => {
    $(selector).addEventListener("change", () => action("/api/teleop/preview", teleopInputSelection()));
  });
  $("#teleop-start").addEventListener("click", () => action("/api/teleop/start", { mode: app.selectedMode, ...teleopInputSelection() }));
  $("#teleop-toggle").addEventListener("click", () => action("/api/teleop/toggle"));
  $("#teleop-stop").addEventListener("click", () => action("/api/teleop/stop"));
  document.addEventListener("keydown", event => {
    if (event.code === "Space" && !event.repeat && $("#tab-teleop").classList.contains("active") && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
      event.preventDefault(); action("/api/teleop/toggle");
    }
  });
  window.addEventListener("resize", drawManus);
}

bind();
poll();
