/* ============================================================
   Handboard: webcam hand tracking whiteboard
   Hand tracking: MediaPipe Hand Landmarker (runs locally via WASM)
   ============================================================ */

const TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Part of the camera frame that maps onto the whole board. Fingertips near the
// frame edges track poorly (and the wrist leaves the frame at the bottom), so
// the reachable area is inset, mostly from the bottom.
const REACH = { x: 0.12, top: 0.10, bottom: 0.28 };

const DWELL_MS = 650;        // hold over a button to press it
const REPEAT_MS = 550;       // undo / redo repeat while held
const STAMP_MS = 900;        // hold a shape still to place it
const STAMP_TOL = 14;        // px of drift allowed while holding a shape
const LOST_GRACE_MS = 140;   // keep a hand alive through brief tracking dropouts
const HIGHLIGHT_ALPHA = 0.36;

const TAU = Math.PI * 2;
const $ = (s, r = document) => r.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const r1 = v => Math.round(v * 10) / 10;
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const TOOLS = [
  { id: "select",    label: "Select",    kind: "select", key: "s", widths: [3, 6, 11, 20] },
  { id: "brush",     label: "Brush",     kind: "stroke", key: "b", widths: [3, 6, 11, 20] },
  { id: "highlight", label: "Highlight", kind: "stroke", key: "h", widths: [12, 20, 30, 46] },
  { id: "eraser",    label: "Eraser",    kind: "erase",  key: "e", widths: [16, 30, 56, 96] },
  { id: "text",      label: "Text",      kind: "text",   key: "t", widths: [20, 28, 40, 60] },   // font sizes
  { id: "line",      label: "Line",      kind: "shape",  key: "l", widths: [3, 5, 9, 15] },
  { id: "arrow",     label: "Arrow",     kind: "shape",  key: "a", widths: [3, 5, 9, 15] },
  { id: "rect",      label: "Box",       kind: "shape",  key: "r", widths: [3, 5, 9, 15] },
  { id: "circle",    label: "Circle",    kind: "shape",  key: "c", widths: [3, 5, 9, 15] },
  { id: "triangle",  label: "Triangle",  kind: "shape",  key: "y", widths: [3, 5, 9, 15] },
];
const TOOL = Object.fromEntries(TOOLS.map(t => [t.id, t]));

// Each ink has a chalk tone for the blackboard and a marker tone for the whiteboard.
const COLORS = [
  { key: "ink",    black: "#f1eee4", white: "#1d2420" },
  { key: "yellow", black: "#f4d867", white: "#c99400" },
  { key: "pink",   black: "#f497b6", white: "#d6336c" },
  { key: "blue",   black: "#86c8f4", white: "#1f5fd1" },
  { key: "green",  black: "#a3e092", white: "#2b8a3e" },
  { key: "orange", black: "#f6a868", white: "#e8590c" },
];
const SIZE_DOTS = [5, 9, 14, 21];

const ICONS = {
  select: '<path d="M6 3.5v15.5l4-4 3 6.2 2.6-1.2-3-6.1H18z"/>',
  brush: '<path d="M4 20l1.1-4.4L16.2 4.5a2.1 2.1 0 013 0l.3.3a2.1 2.1 0 010 3L8.4 18.9z"/><path d="M14.5 6.2l3.3 3.3"/>',
  highlight: '<path d="M14 4l6 6-8.5 8.5H7.5v-4z"/><path d="M7.5 18.5L4.5 21.5"/><path d="M11 7l6 6"/>',
  eraser: '<path d="M4.6 17.4a2 2 0 010-2.8l9.2-9.2a2 2 0 012.8 0l3 3a2 2 0 010 2.8L11 19.8H8.6a2 2 0 01-1.4-.6z"/><path d="M9.5 9.5l5 5"/><path d="M14 20h6"/>',
  line: '<path d="M5 19L19 5"/>',
  arrow: '<path d="M5 19L19 5"/><path d="M10 5h9v9"/>',
  rect: '<rect x="4" y="6" width="16" height="12" rx="1.5"/>',
  circle: '<circle cx="12" cy="12" r="8"/>',
  triangle: '<path d="M12 4.5l8 14H4z"/>',
  text: '<path d="M5 7V5h14v2"/><path d="M12 5v14"/><path d="M9 19h6"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 010 11H11"/>',
  redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 000 11H13"/>',
  clear: '<path d="M4 7h16"/><path d="M10 11v6M14 11v6"/><path d="M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12"/><path d="M9 7V4.5h6V7"/>',
  board: '<rect x="3.5" y="4.5" width="17" height="13" rx="1.5"/><path d="M12 4.5h7a1.5 1.5 0 011.5 1.5v10a1.5 1.5 0 01-1.5 1.5h-7z" fill="currentColor"/><path d="M8.5 21h7M12 17.5V21"/>',
  save: '<path d="M12 4v11"/><path d="M7 10.5l5 5 5-5"/><path d="M5 20h14"/>',
  camera: '<rect x="3" y="6.5" width="13" height="11" rx="2"/><path d="M16 10.5l5-3v9l-5-3z"/>',
  cameraOff: '<path d="M8.5 6.5H14a2 2 0 012 2v3.5m0 4V15.5a2 2 0 01-2 2H5a2 2 0 01-2-2v-7a2 2 0 011.2-1.8"/><path d="M16 10.5l5-3v9l-3.3-2"/><path d="M3 3l18 18"/>',
  preview: '<rect x="3" y="5" width="18" height="14" rx="2"/><rect x="12" y="11" width="6.5" height="5.5" rx="1" fill="currentColor"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.4a2.5 2.5 0 114 2.1c-.9.6-1.6 1.1-1.6 2.3"/><circle cx="12" cy="17.2" r=".7" fill="currentColor"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
  practice: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
};
const svg = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const FINGERS = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

/* ---------- Persistence (per-browser convenience only) ---------- */
const STORE = "handboard.v1";
const store = {
  get(k, fallback) {
    try { const v = localStorage.getItem(`${STORE}.${k}`); return v == null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(k, v) { try { localStorage.setItem(`${STORE}.${k}`, JSON.stringify(v)); } catch { /* storage full or blocked */ } },
};

const saved = store.get("settings", {}) || {};
const state = {
  tool: TOOL[saved.tool] ? saved.tool : "brush",
  color: COLORS.some(c => c.key === saved.color) ? saved.color : "ink",
  sizeIdx: [0, 1, 2, 3].includes(saved.sizeIdx) ? saved.sizeIdx : 1,
  board: saved.board === "white" ? "white" : "black",
  drawWith: saved.drawWith === "pinch" ? "pinch" : "point",
  maxHands: saved.maxHands === 1 ? 1 : 2,
  ghost: !!saved.ghost,
  camVisible: saved.camVisible !== false,
  camBox: saved.camBox && ["x", "y", "w"].every(k => Number.isFinite(saved.camBox[k])) ? saved.camBox : null,   // preview position + width
  camOn: false,             // webcam is capturing
  mode: "intro",            // intro | camera | mouse
  view: "board",            // board | practice
  items: [],
  redo: [],
};
{
  const items = store.get("items", []);
  if (Array.isArray(items)) state.items = items.filter(it => it && ["stroke", "shape", "text", "clear", "move", "remove"].includes(it.type));
}

/* ---------- Elements ---------- */
const el = {
  ghost: $("#ghost"), ink: $("#ink"), live: $("#live"), cursors: $("#cursors"),
  video: $("#video"), camView: $("#cam-view"), camOverlay: $("#cam-overlay"), cam: $("#cam"),
  status: $("#status"), statusTitle: $("#status-title"), statusHint: $("#status-hint"),
  hint: $("#empty-hint"), toast: $("#toast"), intro: $("#intro"), help: $("#help"),
  textEdit: $("#text-edit"), textInput: $("#text-input"), selBar: $("#sel-bar"),
  tools: $("#tools"), swatches: $("#swatches"), sizes: $("#sizes"), actions: $("#actions"),
  practice: $("#practice"), pCanvas: $("#p-canvas"), pStage: $("#p-stage"), pHud: $("#p-hud"), pEyebrow: $("#p-eyebrow"),
  pTrack: $("#p-track"), pTrackText: $("#p-track-text"), pSkip: $("#p-skip"), pMenuBtn: $("#p-menu-btn"), pHint: $("#p-hint"),
  pMenu: $("#p-menu"), pResult: $("#p-result"), pDiff: $("#p-diff"), pCards: $("#p-cards"), pStats: $("#p-stats"),
  rEyebrow: $("#r-eyebrow"), rStars: $("#r-stars"), rHeadline: $("#r-headline"), rMessage: $("#r-message"),
  rBest: $("#r-best"), rDetails: $("#r-details"), rItems: $("#r-items"),
};
const ctx = {
  ink: el.ink.getContext("2d"), live: el.live.getContext("2d"), cursors: el.cursors.getContext("2d"),
  ghost: el.ghost.getContext("2d"), cam: el.camOverlay.getContext("2d"), p: el.pCanvas.getContext("2d"),
};

let W = innerWidth, H = innerHeight, DPR = 1;
let VW = 1280, VH = 720;     // camera frame size
let landmarker = null;
let theme = {};

/* ---------- Build the chrome ---------- */
function buildUI() {
  const groups = [["select"], ["brush", "highlight", "eraser", "text"], ["line", "arrow", "rect", "circle", "triangle"]];
  el.tools.innerHTML = groups.map(g => g.map(id => {
    const t = TOOL[id];
    return `<button class="btn" data-act="tool" data-arg="${id}" aria-pressed="false" aria-label="${t.label}" title="${t.label} (${t.key.toUpperCase()})">${svg(id)}<span class="lbl">${t.label}</span></button>`;
  }).join("")).join('<span class="sep" aria-hidden="true"></span>');

  el.swatches.innerHTML = COLORS.map(c =>
    `<button class="btn swatch" data-act="color" data-arg="${c.key}" aria-pressed="false"><span class="dot"></span></button>`).join("");

  el.sizes.innerHTML = SIZE_DOTS.map((d, i) =>
    `<button class="btn size" data-act="size" data-arg="${i}" aria-pressed="false" aria-label="Size ${i + 1}" title="Size ${i + 1} (${i + 1})"><span class="dot" style="width:${d}px;height:${d}px"></span></button>`).join("");

  const act = (a, icon, label, title, extra = "") =>
    `<button class="btn" data-act="${a}" title="${title}" ${extra}>${svg(icon)}<span class="lbl">${label}</span></button>`;
  el.actions.innerHTML = [
    act("undo", "undo", "Undo", "Undo (⌘Z)", "data-repeat"),
    act("redo", "redo", "Redo", "Redo (⇧⌘Z)", "data-repeat"),
    act("clear", "clear", "Clear", "Clear the board. Hand: hold a little longer.", 'data-dwell="1400"'),
    '<span class="sep" aria-hidden="true"></span>',
    act("board", "board", "Board", "Switch board (W)"),
    act("save", "save", "Save", "Save as PNG (⌘S)"),
    act("camera", "camera", "Camera", "Turn the camera on or off", 'aria-pressed="false"'),
    act("cam", "preview", "Preview", "Show or hide the camera preview (V)", 'aria-pressed="false"'),
    act("practice", "practice", "Practice", "Hand practice games (P)"),
    act("help", "help", "Help", "Gestures and settings (?)"),
  ].join("");

  $("#cam-actions").innerHTML =
    `<button class="chip" data-act="ghost" aria-pressed="false" title="Show the camera faintly behind the board (G)">${svg("eye")}On board</button>` +
    `<button class="chip icon-only" data-act="cam" aria-label="Hide camera preview" title="Hide camera preview (V)">${svg("close")}</button>`;

  $("[data-act=help-close]").innerHTML = svg("close");

  $("[data-act=sel-delete]").innerHTML = `${svg("clear")}Delete`;

  const keys = [["S", "Select and move"], ["⌫", "Delete selected"], ["B", "Brush"], ["H", "Highlight"], ["E", "Eraser"], ["L", "Line"], ["A", "Arrow"], ["R", "Box"],
    ["C", "Circle"], ["Y", "Triangle"], ["T", "Text"], ["1–4", "Size"], ["⌘Z", "Undo"], ["⇧⌘Z", "Redo"], ["W", "Switch board"],
    ["G", "Camera on board"], ["V", "Camera preview"], ["⌘S", "Save PNG"], ["P", "Practice games"], ["?", "This panel"]];
  $("#keys").innerHTML = keys.map(([k, d]) => `<div><dt><kbd>${k}</kbd></dt><dd>${d}</dd></div>`).join("");
}

/* Small schematic hand drawings for the gesture guide. */
function handGlyph({ fingers, thumb = "in", extra = "", faded = false }) {
  const xs = [21, 30.5, 40, 49], tops = [6, 2, 5, 12];
  let back = "";
  fingers.forEach((up, i) => {
    let top = up ? tops[i] : 22;
    if (thumb === "pinch" && i === 0) top = 14;
    back += `<rect x="${xs[i] - 4.5}" y="${top}" width="9" height="${38 - top}" rx="4.5" class="${up ? "g-up" : "g-base"}"/>`;
  });
  if (thumb === "out") back += `<rect x="13.5" y="26" width="9" height="24" rx="4.5" transform="rotate(-34 18 50)" class="g-up"/>`;
  if (thumb === "pinch") back += `<rect x="13.5" y="15" width="9" height="35" rx="4.5" transform="rotate(5 18 50)" class="g-up"/>`;
  if (thumb === "in") back += `<rect x="11" y="34" width="9" height="16" rx="4.5" transform="rotate(-18 15.5 50)" class="g-base"/>`;
  let front = `<rect x="16" y="30" width="38" height="26" rx="11" class="g-base"/>`;
  if (thumb === "pinch") front += `<circle cx="21" cy="16.5" r="3" class="g-dot"/>`;
  const body = back + front + extra;
  return `<svg class="glyph" viewBox="0 0 60 60" aria-hidden="true">${faded ? `<g opacity=".45">${body}</g>` : body}</svg>`;
}
const RING = '<circle cx="21" cy="9.5" r="9" class="g-ringbg"/><circle cx="21" cy="9.5" r="9" class="g-acc" stroke-dasharray="40 60" transform="rotate(-90 21 9.5)"/>';
const SPREAD = '<path d="M8.5 28.5L19 11" class="g-acc" stroke-dasharray="1 4.5"/><circle cx="6.5" cy="31" r="3" class="g-dot"/><circle cx="21" cy="8.5" r="3" class="g-dot"/>';

let guideMode = "";
function renderGuides() {
  if (guideMode === state.drawWith) return;
  guideMode = state.drawWith;
  const pinch = state.drawWith === "pinch";
  const cards = [
    pinch
      ? [handGlyph({ fingers: [1, 0, 0, 0], thumb: "pinch" }), "Draw", "Pinch your thumb and index finger together."]
      : [handGlyph({ fingers: [1, 0, 0, 0] }), "Draw", "Raise only your index finger."],
    pinch
      ? [handGlyph({ fingers: [1, 1, 0, 0], thumb: "out" }), "Move", "Open the pinch to move without drawing."]
      : [handGlyph({ fingers: [1, 1, 0, 0] }), "Move", "Raise two fingers to move without drawing."],
    [handGlyph({ fingers: [1, 0, 0, 0], extra: RING }), "Pick a tool",
      pinch ? "Hold the cursor on a button until the ring fills, or pinch on it." : "Hold the cursor on a button until the ring fills."],
    [handGlyph({ fingers: [1, 0, 0, 0], thumb: "out", extra: SPREAD }), "Shapes", "Spread thumb and index to size it. Hold still to place it."],
    pinch
      ? [handGlyph({ fingers: [1, 1, 1, 1], thumb: "out", faded: true }), "Rest", "Drop your hand out of view to pause."]
      : [handGlyph({ fingers: [0, 0, 0, 0] }), "Rest", "Make a fist to pause without drawing."],
  ];
  const html = cards.map(([g, t, d]) => `<figure class="gcard">${g}<figcaption><strong>${t}</strong><span>${d}</span></figcaption></figure>`).join("");
  document.querySelectorAll("[data-guide]").forEach(n => { n.innerHTML = html; });
}

/* ---------- Theme + chrome sync ---------- */
function colorOf(key) {
  const c = COLORS.find(c => c.key === key) || COLORS[0];
  return c[state.board];
}
function colorName(c) {
  if (c.key === "ink") return state.board === "black" ? "Chalk white" : "Marker black";
  return c.key[0].toUpperCase() + c.key.slice(1);
}
function withAlpha(hex, a) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
function refreshTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = name => cs.getPropertyValue(name).trim();
  theme = { accent: v("--accent"), fg: v("--fg"), board: v("--board"), live: v("--live"), danger: v("--danger") };
}
const toolWidth = (tool = TOOL[state.tool]) => tool.widths[state.sizeIdx];
const camReady = () => !!landmarker && state.camOn;
// Drawing with the mouse only: mouse mode, or the camera has been switched off.
const mouseOnly = () => state.mode === "mouse" || (state.mode === "camera" && !!landmarker && !state.camOn && !camStarting);

function syncUI() {
  document.documentElement.dataset.board = state.board;
  refreshTheme();
  for (const b of el.tools.querySelectorAll("[data-act=tool]")) b.setAttribute("aria-pressed", b.dataset.arg === state.tool);
  for (const b of el.swatches.querySelectorAll("[data-act=color]")) {
    const c = COLORS.find(c => c.key === b.dataset.arg);
    b.firstElementChild.style.background = c[state.board];
    b.setAttribute("aria-pressed", c.key === state.color);
    b.setAttribute("aria-label", colorName(c));
    b.title = colorName(c);
  }
  const erasing = state.tool === "eraser";
  for (const b of el.sizes.querySelectorAll("[data-act=size]")) {
    b.setAttribute("aria-pressed", Number(b.dataset.arg) === state.sizeIdx);
    b.firstElementChild.style.background = erasing ? "transparent" : colorOf(state.color);
    b.firstElementChild.classList.toggle("hollow", erasing);
  }
  $("[data-act=board]").title = state.board === "black" ? "Switch to whiteboard (W)" : "Switch to blackboard (W)";
  const camBtn = $("#actions [data-act=camera]"), previewBtn = $("#actions [data-act=cam]");
  camBtn.setAttribute("aria-pressed", state.camOn);
  camBtn.querySelector("svg").innerHTML = ICONS[state.camOn ? "camera" : "cameraOff"];
  camBtn.title = state.camOn ? "Turn the camera off (stops capturing)" : "Turn the camera on";
  previewBtn.setAttribute("aria-pressed", state.camVisible && camReady());
  previewBtn.disabled = !camReady();
  $("[data-act=ghost]").setAttribute("aria-pressed", state.ghost);
  for (const b of document.querySelectorAll("[data-act=draw-with]")) b.setAttribute("aria-pressed", b.dataset.arg === state.drawWith);
  for (const b of document.querySelectorAll("[data-act=hands]")) b.setAttribute("aria-pressed", Number(b.dataset.arg) === state.maxHands);
  el.cam.classList.toggle("collapsed", !(camReady() && state.camVisible));
  placeCam();
  el.ghost.hidden = !(state.ghost && camReady());
  renderGuides();
  if (textEdit.open) styleEditor();
  syncDoc();
  statusKey = "";
  saveSettings();
}
function saveSettings() {
  store.set("settings", {
    tool: state.tool, color: state.color, sizeIdx: state.sizeIdx, board: state.board, drawWith: state.drawWith,
    maxHands: state.maxHands, ghost: state.ghost, camVisible: state.camVisible, camBox: state.camBox,
  });
}

function syncDoc() {
  $("[data-act=undo]").disabled = !state.items.length;
  $("[data-act=redo]").disabled = !state.redo.length;
  el.hint.textContent = mouseOnly() ? "Drag anywhere to draw"
    : state.drawWith === "pinch" ? "Pinch to draw" : "Raise your index finger to draw";
  el.hint.hidden = hasInk();
}

/* ---------- Document model: strokes, shapes, clears ---------- */
function lastClear() {
  for (let i = state.items.length - 1; i >= 0; i--) if (state.items[i].type === "clear") return i;
  return -1;
}
const hasInk = () => state.items.length - 1 > lastClear();

function tracePath(c, p) {
  c.beginPath();
  c.moveTo(p[0][0], p[0][1]);
  if (p.length === 2) { c.lineTo(p[1][0], p[1][1]); return; }
  for (let i = 1; i < p.length - 1; i++) {
    c.quadraticCurveTo(p[i][0], p[i][1], (p[i][0] + p[i + 1][0]) / 2, (p[i][1] + p[i + 1][1]) / 2);
  }
  const l = p[p.length - 1];
  c.lineTo(l[0], l[1]);
}

function traceShape(c, shape, a, b, w) {
  const [ax, ay] = a, [bx, by] = b;
  c.beginPath();
  switch (shape) {
    case "line":
      c.moveTo(ax, ay); c.lineTo(bx, by); break;
    case "arrow": {
      const ang = Math.atan2(by - ay, bx - ax), len = Math.hypot(bx - ax, by - ay);
      const head = Math.min(len * 0.45, Math.max(14, w * 3.2));
      c.moveTo(ax, ay); c.lineTo(bx, by);
      c.moveTo(bx - head * Math.cos(ang - 0.5), by - head * Math.sin(ang - 0.5));
      c.lineTo(bx, by);
      c.lineTo(bx - head * Math.cos(ang + 0.5), by - head * Math.sin(ang + 0.5));
      break;
    }
    case "rect":
      c.rect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay)); break;
    case "circle":
      c.arc((ax + bx) / 2, (ay + by) / 2, Math.hypot(bx - ax, by - ay) / 2, 0, TAU); break;
    case "triangle": {
      const [p0, p1, p2] = trianglePts(a, b);
      c.moveTo(p0[0], p0[1]); c.lineTo(p1[0], p1[1]); c.lineTo(p2[0], p2[1]);
      c.closePath();
      break;
    }
  }
  c.stroke();
}
// An upright equilateral triangle inside the circle whose diameter is a–b.
function trianglePts([ax, ay], [bx, by]) {
  const cx = (ax + bx) / 2, cy = (ay + by) / 2, r = Math.hypot(bx - ax, by - ay) / 2;
  return [0, 1, 2].map(k => { const t = -Math.PI / 2 + k * TAU / 3; return [cx + r * Math.cos(t), cy + r * Math.sin(t)]; });
}

function drawItem(c, it) {
  if (it.type === "move" || it.type === "remove") return;   // history entries, nothing to draw
  if (it.type === "clear") {
    c.save(); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, c.canvas.width, c.canvas.height); c.restore();
    return;
  }
  const col = colorOf(it.c);
  c.save();
  c.lineCap = "round"; c.lineJoin = "round"; c.lineWidth = it.w;
  c.strokeStyle = col; c.fillStyle = col;
  if (it.tool === "eraser") {
    c.globalCompositeOperation = "destination-out";
    c.strokeStyle = c.fillStyle = "#000";
  } else if (it.tool === "highlight") {
    c.globalAlpha *= HIGHLIGHT_ALPHA;
    if (state.board === "white") c.globalCompositeOperation = "multiply";
  } else if (state.board === "black") {
    c.shadowColor = withAlpha(col, 0.4); c.shadowBlur = 5;   // a little chalk bloom
  }
  if (it.type === "stroke") {
    const p = it.p;
    if (p.length === 1) { c.beginPath(); c.arc(p[0][0], p[0][1], it.w / 2, 0, TAU); c.fill(); }
    else { tracePath(c, p); c.stroke(); }
  } else if (it.type === "text") {
    drawText(c, it);
  } else {
    traceShape(c, it.shape, it.a, it.b, it.w);
  }
  c.restore();
}

function renderAll() {
  const c = ctx.ink;
  c.clearRect(0, 0, W, H);
  const hidden = hiddenIds(), off = offsets();
  for (let i = lastClear() + 1; i < state.items.length; i++) {
    const it = state.items[i];
    if (it.id && hidden.has(it.id)) continue;
    const o = it.id && off.get(it.id);
    if (o) { c.save(); c.translate(o[0], o[1]); drawItem(c, it); c.restore(); }
    else drawItem(c, it);
  }
}

let saveTimer;
function afterChange() {
  syncDoc();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store.set("items", state.items.slice(lastClear() + 1)), 500);
}

function commit(item) {
  state.items.push(item);
  state.redo.length = 0;
  // Edits, moves and deletes change something already on the board, so redraw it all.
  if (item.edits || item.type === "move" || item.type === "remove") renderAll();
  else drawItem(ctx.ink, item);
  afterChange();
}

function undo() {
  textCommit();
  dropSelection();
  const it = state.items.pop();
  if (!it) return;
  state.redo.push(it);
  renderAll(); afterChange();
}
function redo() {
  textCommit();
  dropSelection();
  const it = state.redo.pop();
  if (!it) return;
  state.items.push(it);
  renderAll(); afterChange();
}
function clearBoard() {
  textCommit();
  dropSelection();
  if (!hasInk()) { toast("The board is already clear"); return; }
  commit({ type: "clear" });
  toast("Board cleared. Undo brings it back.");
}

/* ---------- Strokes ---------- */
function beginStroke(s, x, y) {
  const tool = TOOL[state.tool];
  s.stroke = { type: "stroke", id: newId(), tool: tool.id, c: state.color, w: toolWidth(tool), p: [[r1(x), r1(y)]] };
  if (tool.id === "eraser") {
    const c = ctx.ink;
    c.save(); c.globalCompositeOperation = "destination-out";
    c.beginPath(); c.arc(x, y, s.stroke.w / 2, 0, TAU); c.fill(); c.restore();
  }
}
function extendStroke(s, x, y) {
  const st = s.stroke, last = st.p[st.p.length - 1];
  if (Math.hypot(x - last[0], y - last[1]) < 1.5) return;
  st.p.push([r1(x), r1(y)]);
  if (st.tool === "eraser") {   // erase live, straight onto the ink layer
    const c = ctx.ink;
    c.save(); c.globalCompositeOperation = "destination-out";
    c.lineCap = "round"; c.lineWidth = st.w;
    c.beginPath(); c.moveTo(last[0], last[1]); c.lineTo(x, y); c.stroke(); c.restore();
  }
}
function endStroke(s, trim = 0) {
  const st = s.stroke;
  if (!st) return;
  s.stroke = null;
  // Curling the finger to stop drawing drags the tip; drop that last hook.
  if (trim && st.tool !== "eraser" && st.p.length > trim + 3) st.p.length -= trim;
  commit(st);
}
function makeShape(tool, a, b) {
  return { type: "shape", id: newId(), shape: tool.id, c: state.color, w: toolWidth(tool), a: [r1(a[0]), r1(a[1])], b: [r1(b[0]), r1(b[1])] };
}
// Also makes each hand lift its finger before it draws again, so switching
// tools while pointing doesn't start a mark straight away.
function endAllStrokes() {
  textCommit();
  dropSelection();
  for (const s of allSources()) {
    endStroke(s, 0);
    s.shape = null; s.stamp = null; s.anchor = null; s.drawing = false;
    if (s.kind === "hand" && s.visible) s.needsFresh = true;
  }
}

/* ---------- Text ---------- */
const TEXT_LINE = 1.25;                       // line height, as a multiple of the font size
const TEXT_PAD = [4, 6], TEXT_BORDER = 1.5;   // editor padding (top/bottom, sides) and border, px
const textFont = size => `700 ${size}px Kalam, "Segoe Print", "Bradley Hand", cursive`;
const textSize = () => TOOL.text.widths[state.sizeIdx];
const newId = () => Math.random().toString(36).slice(2, 10);
const measure = document.createElement("canvas").getContext("2d");
const textEdit = { open: false, x: 0, y: 0, edits: null };

// Where the first baseline sits in a line, worked out the way CSS lays out the
// editor, so text doesn't jump when it's committed to the board.
function baselineOffset(c, size) {
  const m = c.measureText("Hg"), asc = m.fontBoundingBoxAscent, desc = m.fontBoundingBoxDescent;
  const lh = size * TEXT_LINE;
  return asc != null && desc != null ? (lh - asc - desc) / 2 + asc : lh * 0.78;
}
function drawText(c, it) {
  if (!it.t) return;
  c.font = textFont(it.s);
  c.textBaseline = "alphabetic";
  const lh = it.s * TEXT_LINE, base = baselineOffset(c, it.s);
  it.t.split("\n").forEach((line, i) => c.fillText(line, it.x, it.y + i * lh + base));
}
function textBounds(it) {
  measure.font = textFont(it.s);
  const lines = it.t.split("\n");
  return { x: it.x, y: it.y, w: Math.max(...lines.map(l => measure.measureText(l).width)), h: lines.length * it.s * TEXT_LINE };
}
// A click (or a point from the hand) with the Text tool: finish the open text,
// edit the text under the cursor, or start a new one.
function textClick(x, y) {
  if (textEdit.open) { textCommit(); return; }
  const hit = itemAt(x, y, it => it.type === "text");
  textEdit.open = true;
  if (hit) {
    const { it } = hit;
    Object.assign(textEdit, { x: it.x + hit.dx, y: it.y + hit.dy, edits: it.id });   // where it is now, after any moves
    state.color = it.c;
    const i = TOOL.text.widths.indexOf(it.s);
    if (i >= 0) state.sizeIdx = i;
    el.textInput.value = it.t;
    renderAll();
  } else {
    Object.assign(textEdit, { x, y: y - textSize() * TEXT_LINE / 2, edits: null });
    el.textInput.value = "";
  }
  el.textEdit.hidden = false;
  syncUI();
  keepTyping();
}
function styleEditor() {
  const size = textSize(), ta = el.textInput;
  ta.style.font = textFont(size);
  ta.style.lineHeight = String(TEXT_LINE);
  ta.style.color = colorOf(state.color);
  measure.font = textFont(size);
  const lines = ta.value.split("\n");
  const w = Math.max(size * 2, ...lines.map(l => measure.measureText(l).width));
  ta.style.width = `${Math.ceil(w + size * 0.6 + 2 * (TEXT_PAD[1] + TEXT_BORDER))}px`;
  ta.style.height = `${Math.ceil(lines.length * size * TEXT_LINE + 2 * (TEXT_PAD[0] + TEXT_BORDER)) + 1}px`;
  el.textEdit.style.left = `${textEdit.x - TEXT_PAD[1] - TEXT_BORDER}px`;
  el.textEdit.style.top = `${textEdit.y - TEXT_PAD[0] - TEXT_BORDER}px`;
}
function keepTyping() {
  if (!textEdit.open) return;
  el.textInput.focus({ preventScroll: true });
  // A click on the board can take focus back after this runs.
  requestAnimationFrame(() => { if (textEdit.open) el.textInput.focus({ preventScroll: true }); });
}
function closeEditor() {
  textEdit.open = false;
  el.textEdit.hidden = true;
  el.textInput.blur();
}
function textCommit() {
  if (!textEdit.open) return;
  const t = el.textInput.value.replace(/\s+$/, ""), { x, y, edits } = textEdit;
  closeEditor();
  const old = edits && state.items.find(it => it.id === edits);
  const unchanged = old && old.t === t && old.c === state.color && old.s === textSize();
  if ((!t && !edits) || unchanged) { if (edits) renderAll(); return; }
  commit({ type: "text", id: newId(), x: r1(x), y: r1(y), t, c: state.color, s: textSize(), ...(edits && { edits }) });
}
function textCancel() {
  if (!textEdit.open) return;
  const edits = textEdit.edits;
  closeEditor();
  if (edits) renderAll();
}
el.textInput.addEventListener("input", styleEditor);

/* ---------- Select: move and delete text, shapes and drawings ---------- */
// Moves and deletes are history entries ({type:"move"|"remove", target}), so Undo reverses them.
const sel = { id: null, drag: null };   // drag: { by: "mouse" | hand, sx, sy, dx, dy }
const MOVABLE = new Set(["stroke", "shape", "text"]);
const findItem = id => state.items.find(it => it.id === id);

// Items not drawn: texts with a newer edited version, deleted items, the text open
// in the editor, and the item being dragged (the live layer draws that one).
function hiddenIds() {
  const ids = new Set();
  for (const it of state.items) {
    if (it.edits) ids.add(it.edits);
    if (it.type === "remove") ids.add(it.target);
  }
  if (textEdit.open && textEdit.edits) ids.add(textEdit.edits);
  if (sel.drag) ids.add(sel.id);
  return ids;
}
// How far each item has been moved in total.
function offsets() {
  const m = new Map();
  for (const it of state.items) {
    if (it.type !== "move") continue;
    const o = m.get(it.target) || [0, 0];
    m.set(it.target, [o[0] + it.dx, o[1] + it.dy]);
  }
  return m;
}

function segDist(px, py, [ax, ay], [bx, by]) {
  const vx = bx - ax, vy = by - ay, len2 = vx * vx + vy * vy;
  const t = len2 ? clamp(((px - ax) * vx + (py - ay) * vy) / len2, 0, 1) : 0;
  return Math.hypot(px - ax - vx * t, py - ay - vy * t);
}
function polyDist(pts, x, y) {
  let d = Infinity;
  for (let i = 1; i < pts.length; i++) d = Math.min(d, segDist(x, y, pts[i - 1], pts[i]));
  return d;
}
// Text is grabbed anywhere in its box; lines and outlines near the ink itself.
function hitItem(it, x, y) {
  if (it.type === "text") {
    const b = textBounds(it);
    return x >= b.x - 8 && x <= b.x + b.w + 8 && y >= b.y - 8 && y <= b.y + b.h + 8;
  }
  const tol = Math.max(14, it.w / 2 + 8);
  if (it.type === "stroke") {
    return it.p.length === 1 ? Math.hypot(x - it.p[0][0], y - it.p[0][1]) <= tol : polyDist(it.p, x, y) <= tol;
  }
  const [ax, ay] = it.a, [bx, by] = it.b;
  switch (it.shape) {
    case "circle": return Math.abs(Math.hypot(x - (ax + bx) / 2, y - (ay + by) / 2) - Math.hypot(bx - ax, by - ay) / 2) <= tol;
    case "rect": return polyDist([[ax, ay], [bx, ay], [bx, by], [ax, by], [ax, ay]], x, y) <= tol;
    case "triangle": { const t = trianglePts(it.a, it.b); return polyDist([...t, t[0]], x, y) <= tol; }
    default: return segDist(x, y, it.a, it.b) <= tol;   // line, arrow
  }
}
function itemBounds(it) {
  if (it.type === "text") return textBounds(it);
  const pad = it.w / 2 + (it.shape === "arrow" ? Math.max(14, it.w * 3.2) / 2 : 0) + 2;
  let pts = it.type === "stroke" ? it.p : [it.a, it.b];
  if (it.shape === "circle" || it.shape === "triangle") {
    const cx = (it.a[0] + it.b[0]) / 2, cy = (it.a[1] + it.b[1]) / 2, r = dist(it.a, it.b) / 2;
    pts = [[cx - r, cy - r], [cx + r, cy + r]];
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  return { x: x0 - pad, y: y0 - pad, w: x1 - x0 + 2 * pad, h: y1 - y0 + 2 * pad };
}
// The topmost item under (x, y), with how far it has been moved.
function itemAt(x, y, accept = () => true) {
  const hidden = hiddenIds(), off = offsets();
  for (let i = state.items.length - 1; i > lastClear(); i--) {
    const it = state.items[i];
    if (!MOVABLE.has(it.type) || !it.id || hidden.has(it.id) || it.tool === "eraser") continue;
    if ((it.type === "text" && !it.t) || !accept(it)) continue;
    const [dx, dy] = off.get(it.id) || [0, 0];
    if (hitItem(it, x - dx, y - dy)) return { it, dx, dy };
  }
  return null;
}

function beginMove(by, x, y, hit) {
  sel.id = hit.it.id;
  sel.drag = { by, sx: x, sy: y, dx: 0, dy: 0 };
  renderAll();   // lift it off the board while it moves
}
function updateMove(x, y) {
  sel.drag.dx = x - sel.drag.sx;
  sel.drag.dy = y - sel.drag.sy;
}
function endMove() {
  const d = sel.drag;
  if (!d) return;
  sel.drag = null;
  if (Math.hypot(d.dx, d.dy) >= 1) commit({ type: "move", target: sel.id, dx: r1(d.dx), dy: r1(d.dy) });
  else renderAll();
}
function dropSelection() { endMove(); sel.id = null; }
function deleteSelected() {
  if (!sel.id) return;
  endMove();
  commit({ type: "remove", target: sel.id });
  sel.id = null;
  toast("Deleted. Undo brings it back.");
}

function outlineBox(c, b, dx, dy, selected) {
  c.save();
  c.setLineDash(selected ? [6, 5] : [3, 5]);
  c.lineWidth = selected ? 1.5 : 1;
  c.strokeStyle = selected ? theme.accent : withAlpha(theme.fg, 0.45);
  c.strokeRect(b.x + dx - 6, b.y + dy - 6, b.w + 12, b.h + 12);
  c.restore();
}
// The Delete button floats just above the selected item.
let selBarAt = "";
function placeSelBar(it, off) {
  if (!it || sel.drag || state.tool !== "select") {
    if (!el.selBar.hidden) { el.selBar.hidden = true; selBarAt = ""; }
    return;
  }
  const [dx, dy] = off.get(it.id) || [0, 0], b = itemBounds(it);
  const x = clamp(b.x + dx + b.w + 6 - 96, 8, W - 104);
  let y = b.y + dy - 62;
  if (y < 8) y = b.y + dy + b.h + 14;
  const at = `${Math.round(x)},${Math.round(y)}`;
  if (at === selBarAt) return;
  if (el.selBar.hidden) el.selBar.dataset.armAt = performance.now() + 900;   // don't delete by landing on it
  selBarAt = at;
  el.selBar.style.left = `${Math.round(x)}px`;
  el.selBar.style.top = `${Math.round(y)}px`;
  el.selBar.hidden = false;
}

/* ---------- Input sources: mouse/touch + up to two hands ---------- */
class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.reset();
  }
  reset() { this.x = null; this.dx = 0; this.t = 0; }
  static alpha(cutoff, dt) { const tau = 1 / (TAU * cutoff); return 1 / (1 + tau / dt); }
  filter(x, t) {
    if (this.x === null) { this.x = x; this.t = t; return x; }
    const dt = Math.max(1e-3, (t - this.t) / 1000);
    this.t = t;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    this.dx = ad * ((x - this.x) / dt) + (1 - ad) * this.dx;
    const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt);
    this.x = a * x + (1 - a) * this.x;
    return this.x;
  }
}

function makeSource(id, kind) {
  return {
    id, kind, visible: false, x: 0, y: 0, ix: 0, iy: 0, tx: 0, ty: 0, rx: 0, ry: 0, lastSeen: 0,
    pose: "none", rawPose: "none", cand: null, candN: 0, ext: [false, false, false, false],
    pinched: false, pinchRaw: false, pinchN: 0, prevPinched: false,
    stroke: null, shape: null, stamp: null, stampLock: null, anchor: null, down: false,
    dwell: null, overUI: false, drawing: false, needsFresh: false,
    f: { ix: new OneEuro(), iy: new OneEuro(), tx: new OneEuro(), ty: new OneEuro() },
  };
}
const mouse = makeSource("mouse", "mouse");
const hands = [makeSource("hand-0", "hand"), makeSource("hand-1", "hand")];
const allSources = () => [mouse, ...hands];
const overlayOpen = () => !el.help.hidden || !el.intro.hidden;

el.live.addEventListener("pointerdown", e => {
  if (state.mode === "intro" || e.button > 0) return;
  const tool = TOOL[state.tool];
  if (tool.kind === "text") { e.preventDefault(); textClick(e.clientX, e.clientY); return; }
  if (tool.kind === "select") {
    const hit = itemAt(e.clientX, e.clientY);
    if (!hit) { sel.id = null; return; }
    el.live.setPointerCapture(e.pointerId);
    Object.assign(mouse, { down: true, visible: true, x: e.clientX, y: e.clientY });
    beginMove("mouse", e.clientX, e.clientY, hit);
    return;
  }
  el.live.setPointerCapture(e.pointerId);
  Object.assign(mouse, { down: true, visible: true, x: e.clientX, y: e.clientY });
  if (tool.kind === "shape") { mouse.anchor = [mouse.x, mouse.y]; mouse.shape = null; }
  else { beginStroke(mouse, mouse.x, mouse.y); mouse.drawing = true; }
});
el.live.addEventListener("pointermove", e => {
  Object.assign(mouse, { visible: true, x: e.clientX, y: e.clientY });
  if (!mouse.down) return;
  if (sel.drag?.by === "mouse") { updateMove(mouse.x, mouse.y); return; }
  if (mouse.anchor) {
    mouse.shape = makeShape(TOOL[state.tool], mouse.anchor, [mouse.x, mouse.y]);
  } else if (mouse.stroke) {
    const evs = e.getCoalescedEvents?.() || [];
    for (const ev of evs.length ? evs : [e]) extendStroke(mouse, ev.clientX, ev.clientY);
  }
});
function mouseUp() {
  if (!mouse.down) return;
  mouse.down = false;
  if (sel.drag?.by === "mouse") { endMove(); return; }
  if (mouse.anchor) {
    if (mouse.shape && dist(mouse.shape.a, mouse.shape.b) > 4) commit(mouse.shape);
    mouse.shape = null; mouse.anchor = null;
  } else endStroke(mouse, 0);
  mouse.drawing = false;
}
el.live.addEventListener("pointerup", mouseUp);
el.live.addEventListener("pointercancel", mouseUp);
el.live.addEventListener("pointerleave", () => { if (!mouse.down) mouse.visible = false; });

/* ---------- Hand geometry ---------- */
function toScreen(p) {
  const nx = 1 - p.x;   // mirror, so moving right moves the cursor right
  const x = (nx - REACH.x) / (1 - 2 * REACH.x);
  const y = (p.y - REACH.top) / (1 - REACH.top - REACH.bottom);
  return [clamp(x, 0, 1) * W, clamp(y, 0, 1) * H];
}
const vec = (a, b) => [(b.x - a.x) * VW, (b.y - a.y) * VH, (b.z - a.z) * VW];
const len3 = v => Math.hypot(v[0], v[1], v[2]);
const d3 = (a, b) => len3(vec(a, b));

function fingerStates(h, lm) {
  const w = lm[0];
  return FINGERS.map(([mcp, pip, , tip], i) => {
    const v1 = vec(lm[mcp], lm[pip]), v2 = vec(lm[pip], lm[tip]);
    const cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / ((len3(v1) * len3(v2)) || 1);
    const ratio = d3(w, lm[tip]) / (d3(w, lm[pip]) || 1);
    // Hysteresis: easier to stay extended than to become extended.
    const on = h.ext[i] ? cos > 0.3 && ratio > 1.05 : cos > 0.6 && ratio > 1.15;
    return (h.ext[i] = on);
  });
}
function classify([i, m, r, p]) {
  if (i && !m && !r && !p) return "point";
  if (i && m && !r && !p) return "two";
  if (!i && !m && !r && !p) return "fist";
  if (i && m && r) return "open";
  return "other";
}

function uiAt(x, y) {
  const node = document.elementFromPoint(clamp(x, 0, W - 1), clamp(y, 0, H - 1));
  if (!node) return { node, zone: null, target: null };
  return { node, zone: node.closest("[data-ui]"), target: node.closest("[data-act]") };
}

/* Pair this frame's detections with the hands we were already tracking. */
function assignHands(list) {
  const anchors = list.map(lm => toScreen(lm[9]));
  const pairs = [];
  list.forEach((_, i) => hands.forEach(h => {
    if (h.visible) pairs.push([Math.hypot(h.rx - anchors[i][0], h.ry - anchors[i][1]), i, h]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const taken = new Map(), free = new Set(hands);
  for (const [d, i, h] of pairs) {
    if (taken.has(i) || !free.has(h) || d > Math.max(W, H) * 0.2) continue;
    taken.set(i, h); free.delete(h);
  }
  return list.map((lm, i) => {
    let h = taken.get(i);
    if (!h) {
      h = [...free].find(x => !x.visible) || [...free][0];
      free.delete(h);
      if (h.visible) dropHand(h);   // a different hand: don't continue its stroke
    }
    h.rx = anchors[i][0]; h.ry = anchors[i][1];
    return [h, lm];
  });
}

function dropHand(h) {
  endStroke(h, 0);
  if (camDrag?.by === h) endCamDrag();
  if (sel.drag?.by === h) endMove();
  Object.assign(h, {
    shape: null, stamp: null, stampLock: null, dwell: null, visible: false, drawing: false, overUI: false, pen: false, textDown: false,
    pose: "none", rawPose: "none", cand: null, candN: 0, ext: [false, false, false, false],
    pinched: false, pinchRaw: false, pinchN: 0, prevPinched: false, needsFresh: false,
  });
  for (const f of Object.values(h.f)) f.reset();
}

function processHands(res, now) {
  const list = (res.landmarks || []).slice(0, state.maxHands);
  const seen = new Set();
  for (const [h, lm] of assignHands(list)) { seen.add(h); updateHand(h, lm, now); }
  for (const h of hands) if (!seen.has(h) && h.visible && now - h.lastSeen > LOST_GRACE_MS) dropHand(h);
}

function updateHand(h, lm, now) {
  h.visible = true; h.lastSeen = now;

  // Pose, debounced so a single noisy frame can't start or stop a stroke.
  const raw = classify(fingerStates(h, lm));
  h.rawPose = raw;
  if (raw === h.cand) h.candN++; else { h.cand = raw; h.candN = 1; }
  const need = raw === "point" ? 2 : h.pose === "point" ? 3 : 2;
  if (h.pose !== raw && h.candN >= need) h.pose = raw;

  // Pinch, relative to hand size so it works near or far from the camera.
  const span = d3(lm[4], lm[8]) / (d3(lm[0], lm[9]) || 1);
  h.pinchRaw = h.pinchRaw ? span < 0.42 : span < 0.27;
  if (h.pinchRaw === h.pinched) h.pinchN = 0;
  else if (++h.pinchN >= 2) { h.pinched = h.pinchRaw; h.pinchN = 0; }

  const [ix, iy] = toScreen(lm[8]), [tx, ty] = toScreen(lm[4]);
  h.ix = h.f.ix.filter(ix, now); h.iy = h.f.iy.filter(iy, now);
  h.tx = h.f.tx.filter(tx, now); h.ty = h.f.ty.filter(ty, now);
  if (state.drawWith === "pinch") { h.x = (h.ix + h.tx) / 2; h.y = (h.iy + h.ty) / 2; }
  else { h.x = h.ix; h.y = h.iy; }

  driveHand(h, now);
}

function driveHand(h, now) {
  const tool = TOOL[state.tool];
  const pinchMode = state.drawWith === "pinch";
  const pinchEdge = h.pinched && !h.prevPinched;
  h.prevPinched = h.pinched;

  // Holding the camera preview: it follows the hand until the pinch opens.
  if (camDrag?.by === h) {
    if (h.pinched) { moveCamDrag(h.x, h.y); h.overUI = true; h.dwell = null; return; }
    endCamDrag();
  }

  const ui = uiAt(h.x, h.y);
  h.overUI = !!ui.zone;
  if (h.overUI) {
    endStroke(h, 0);
    if (sel.drag?.by === h) endMove();
    h.shape = null; h.stamp = null; h.drawing = false; h.pen = false;
    h.needsFresh = true;   // leaving a toolbar mid-gesture shouldn't start a line
    if (pinchEdge && ui.zone === el.cam && !ui.target && !camDrag) {   // pinch the preview to grab it
      beginCamDrag(ui.node.closest(".cam-grip") ? "resize" : "move", h.x, h.y, h);
      h.dwell = null;
      return;
    }
    if (pinchMode && pinchEdge && ui.target && !ui.target.disabled && now >= armedAt(ui.target)) {
      activate(ui.target, "hand");
      pulse(h.x, h.y, 14);
      h.dwell = { el: ui.target, since: now, fired: true, progress: 0 };
      return;
    }
    dwellUpdate(h, !pinchMode && h.pose === "fist" ? null : ui.target, now);
    return;
  }
  h.dwell = null;

  const practicing = state.view === "practice";
  const engaged = !practicing && tool.kind === "shape" ? h.pose === "point" : pinchMode ? h.pinched : h.pose === "point";
  if (!engaged) h.needsFresh = false;
  const go = engaged && !h.needsFresh;

  if (practicing) {   // the practice games read h.pen every frame
    h.pen = go; h.drawing = go;
    return;
  }
  if (tool.kind === "text") {   // pointing (or pinching) works like a click
    endStroke(h, 0);
    h.shape = null; h.stamp = null; h.stampLock = null; h.drawing = false;
    if (go && !h.textDown) textClick(h.x, h.y);
    h.textDown = go;
    return;
  }
  if (tool.kind === "select") {   // point (or pinch) on something to grab it, lift to drop
    endStroke(h, 0);
    h.shape = null; h.stamp = null; h.stampLock = null;
    const mine = sel.drag?.by === h;
    if (go && !h.selDown && !sel.drag) {
      const hit = itemAt(h.x, h.y);
      if (hit) beginMove(h, h.x, h.y, hit); else sel.id = null;
    } else if (go && mine) {
      if (pinchMode ? h.pinchRaw : h.rawPose === "point") updateMove(h.x, h.y);   // ignore the finger curling away
    } else if (!go && mine) {
      endMove();
    }
    h.selDown = go;
    h.drawing = sel.drag?.by === h;
    return;
  }
  if (tool.kind === "shape") {
    endStroke(h, 0);
    h.drawing = false;
    shapeUpdate(h, now, tool, go);
    return;
  }
  h.shape = null; h.stamp = null; h.stampLock = null;
  if (go) {
    if (!h.stroke) beginStroke(h, h.x, h.y);
    else if (pinchMode ? h.pinchRaw : h.rawPose === "point") extendStroke(h, h.x, h.y);
  } else if (h.stroke) {
    endStroke(h, pinchMode ? 0 : 1);
  }
  h.drawing = !!h.stroke;
}

/* Shapes: the thumb tip and index tip are the two handles. */
function shapeUpdate(h, now, tool, go) {
  if (!go) { h.shape = null; h.stamp = null; h.stampLock = null; return; }
  const a = [h.tx, h.ty], b = [h.ix, h.iy];
  if (dist(a, b) < 24) { h.shape = null; h.stamp = null; return; }
  if (h.stampLock) {
    // Just placed one: wait for a clear move before offering the next.
    if (dist(a, h.stampLock.a) < 40 && dist(b, h.stampLock.b) < 40) { h.shape = null; h.stamp = null; return; }
    h.stampLock = null;
  }
  h.shape = makeShape(tool, a, b);
  const st = h.stamp;
  if (!st || dist(a, st.a) > STAMP_TOL || dist(b, st.b) > STAMP_TOL) {
    h.stamp = { a, b, since: now, progress: 0 };
    return;
  }
  st.progress = (now - st.since) / STAMP_MS;
  if (st.progress >= 1) {
    commit(h.shape);
    pulse((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, Math.max(16, dist(a, b) / 2));
    h.stampLock = { a, b };
    h.shape = null; h.stamp = null;
  }
}

// Screens that just appeared ignore a hand that happens to rest on a button.
const armedAt = target => Number(target.closest("[data-arm-at]")?.dataset.armAt) || 0;

function dwellUpdate(h, target, now) {
  if (!target) { h.dwell = null; return; }
  let d = h.dwell;
  if (!d || d.el !== target) d = h.dwell = { el: target, since: now, fired: false, progress: 0 };
  if (target.disabled || now < armedAt(target)) { d.since = now; d.progress = 0; return; }
  if (d.fired) {
    d.progress = 0;
    if ("repeat" in target.dataset && now - d.since > REPEAT_MS) {
      activate(target, "hand"); d.since = now; pulse(h.x, h.y, 12);
    }
    return;
  }
  d.progress = Math.min(1, (now - d.since) / (Number(target.dataset.dwell) || DWELL_MS));
  if (d.progress >= 1) {
    d.fired = true; d.since = now; d.progress = 0;
    activate(target, "hand");
    pulse(h.x, h.y, 14);
  }
}

let hovered = new Set();
function syncHover() {
  const next = new Set();
  for (const h of hands) if (h.visible && h.dwell?.el) next.add(h.dwell.el);
  for (const e of hovered) if (!next.has(e)) e.classList.remove("hand-hover");
  for (const e of next) e.classList.add("hand-hover");
  hovered = next;
}

/* ---------- Actions (shared by click, hand and keyboard) ---------- */
const ACTIONS = {
  tool: a => { endAllStrokes(); state.tool = a; syncUI(); },
  color: a => { state.color = a; syncUI(); keepTyping(); },
  size: a => { state.sizeIdx = Number(a); syncUI(); keepTyping(); },
  "text-done": textCommit,
  "text-cancel": textCancel,
  "sel-delete": deleteSelected,
  undo, redo,
  clear: clearBoard,
  board: () => { textCommit(); state.board = state.board === "black" ? "white" : "black"; syncUI(); renderAll(); },
  save: savePng,
  cam: () => { state.camVisible = !state.camVisible; syncUI(); },
  camera: () => {
    if (camStarting) return;
    if (state.camOn) stopCamera(); else startCamera();
  },
  ghost: () => {
    state.ghost = !state.ghost;
    if (!state.ghost) ctx.ghost.clearRect(0, 0, el.ghost.width, el.ghost.height);
    if (state.ghost && !camReady()) toast("Start the camera to see yourself on the board");
    syncUI();
  },
  help: () => { el.help.hidden = false; endAllStrokes(); },
  "help-close": () => { el.help.hidden = true; },
  "draw-with": a => { state.drawWith = a; endAllStrokes(); syncUI(); },
  hands: a => {
    state.maxHands = Number(a);
    landmarker?.setOptions({ numHands: state.maxHands }).catch(() => {});
    syncUI();
  },
  "start-camera": startCamera,
  "start-mouse": startMouse,
  practice: practiceOpen,
  "practice-exit": practiceExit,
  "practice-menu": showMenu,
  "practice-start": a => startActivity(a),
  "practice-again": () => startActivity(P.lastId),
  "practice-skip": () => P.active?.skip?.(performance.now()),
  "practice-diff": a => {
    P.difficulty = a; practiceLog.difficulty = a;
    store.set("practice", practiceLog);
    renderMenu();
  },
};
function activate(btn, via) {
  if (btn.disabled) return;
  ACTIONS[btn.dataset.act]?.(btn.dataset.arg, via);
}
document.addEventListener("click", e => {
  const b = e.target.closest("[data-act]");
  if (b) activate(b, "mouse");
});

addEventListener("keydown", e => {
  // Typing in a text box: Enter finishes, Shift+Enter adds a line, Esc cancels.
  if (e.target.closest?.("#text-edit")) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); textCommit(); }
    else if (e.key === "Escape") { e.preventDefault(); textCancel(); }
    return;
  }
  const k = e.key.toLowerCase(), mod = e.metaKey || e.ctrlKey;
  const practicing = state.view === "practice";
  if (mod && !practicing) {
    if (k === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (k === "y") { e.preventDefault(); redo(); return; }
    if (k === "s") { e.preventDefault(); savePng(); return; }
  }
  if (mod || e.altKey || e.repeat) return;
  if (e.key === "Escape") {
    if (!el.help.hidden) el.help.hidden = true;
    else if (practicing && (P.active || !el.pResult.hidden)) showMenu();
    else if (practicing) practiceExit();
    else if (sel.id) dropSelection();
    return;
  }
  if (!el.intro.hidden) return;
  if ((e.key === "Delete" || e.key === "Backspace") && sel.id && !practicing) { e.preventDefault(); deleteSelected(); return; }
  if (e.key === "?") { el.help.hidden ? ACTIONS.help() : ACTIONS["help-close"](); return; }
  if (practicing) return;
  if (k === "p") return ACTIONS.practice();
  const t = TOOLS.find(t => t.key === k);
  if (t) return ACTIONS.tool(t.id);
  if (["1", "2", "3", "4"].includes(k)) return ACTIONS.size(Number(k) - 1);
  if (k === "w") return ACTIONS.board();
  if (k === "g") return ACTIONS.ghost();
  if (k === "v") return ACTIONS.cam();
});

/* ---------- Save ---------- */
function savePng() {
  if (!hasInk()) { toast("Nothing to save yet"); return; }
  const out = document.createElement("canvas");
  out.width = el.ink.width; out.height = el.ink.height;
  const o = out.getContext("2d");
  o.fillStyle = theme.board; o.fillRect(0, 0, out.width, out.height);
  o.drawImage(el.ink, 0, 0);
  out.toBlob(blob => {
    if (!blob) { toast("Couldn't create the image"); return; }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const d = new Date(), pad = n => String(n).padStart(2, "0");
    a.href = url;
    a.download = `handboard-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.png`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast("Saved to your downloads");
  }, "image/png");
}

/* ---------- Status + toast ---------- */
let statusKey = "", statusLocked = false;
function setStatus(kind, title, hint = "") {
  el.status.hidden = false;
  el.status.dataset.state = kind;
  el.statusTitle.textContent = title;
  el.statusHint.textContent = hint;
}
function toolHint() {
  const t = TOOL[state.tool];
  if (t.kind === "select") {
    if (mouseOnly()) return "Drag text, shapes or drawings to move them. Press Delete to remove the selected one.";
    return state.drawWith === "pinch"
      ? "Pinch something to grab it, move, then open your fingers to drop it."
      : "Point at something to grab it, move, then raise two fingers to drop it.";
  }
  if (t.kind === "text") {
    const how = mouseOnly() ? "Click" : state.drawWith === "pinch" ? "Pinch" : "Point";
    return `${how} where the text goes, type, then press Enter. ${how} on text to edit it.`;
  }
  const verb = t.id === "eraser" ? "erase" : "draw";
  if (mouseOnly()) return t.kind === "shape" ? `Drag to draw a ${t.label.toLowerCase()}.` : `Drag to ${verb}.`;
  if (t.kind === "shape") return "Point, then spread thumb and index to size. Hold still to place.";
  return state.drawWith === "pinch"
    ? `Pinch to ${verb}. Open your fingers to move.`
    : `Index finger up to ${verb}. Two fingers up to move.`;
}
function updateStatus() {
  if (statusLocked) return;
  if (state.mode === "camera" && camReady()) {
    const n = hands.filter(h => h.visible).length;
    const key = `c${n}${state.tool}${state.drawWith}`;
    if (key === statusKey) return;
    statusKey = key;
    setStatus(n ? "tracking" : "waiting", n ? `Tracking ${n} hand${n > 1 ? "s" : ""}` : "Show your hand to the camera", toolHint());
  } else if (mouseOnly()) {
    const off = state.mode === "camera";
    const key = `${off ? "off" : "m"}${state.tool}`;
    if (key === statusKey) return;
    statusKey = key;
    setStatus("idle", off ? "Camera off" : "Mouse mode", off ? `Hand tracking is paused. ${toolHint()}` : toolHint());
  }
}
let toastTimer;
function toast(msg) {
  el.toast.textContent = msg;
  el.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove("show"), 2400);
}

/* ---------- Camera + model ---------- */
async function createLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(`${TASKS_URL}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TASKS_URL}/wasm`);
  const options = delegate => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: state.maxHands,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU hand tracking unavailable, using CPU", err);
    return await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}

function fallBackToMouse(title, hint) {
  state.mode = "mouse";
  statusLocked = true;
  setStatus("error", title, `${hint} You can still draw with the mouse.`);
  syncUI();
}

let camStream = null, camStarting = false;

async function startCamera() {
  if (camStarting || state.camOn) return;
  camStarting = true;
  try { await openCamera(); }
  finally { camStarting = false; statusKey = ""; syncUI(); }
}

async function openCamera() {
  el.intro.hidden = true;
  state.mode = "camera";
  statusLocked = false;
  syncUI();
  if (!navigator.mediaDevices?.getUserMedia) {
    fallBackToMouse("Camera unavailable", "Open this page through a local server (http://localhost) or over https.");
    return;
  }
  setStatus("loading", "Starting camera", "Allow camera access when your browser asks.");
  const model = landmarker ? Promise.resolve(landmarker) : createLandmarker();
  model.catch(() => {});   // handled below; avoids an unhandled-rejection warning meanwhile

  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
  } catch (err) {
    const why = err.name === "NotAllowedError" ? "Camera access was blocked. Allow it in the browser's site settings, then reload."
      : err.name === "NotFoundError" ? "No camera was found. Connect a webcam and reload."
      : err.name === "NotReadableError" ? "Another app is using the camera. Close it and reload."
      : `The camera failed to start (${err.name}).`;
    fallBackToMouse("Camera unavailable", why);
    return;
  }
  el.video.srcObject = camStream;
  try { await el.video.play(); } catch { /* autoplay+muted normally succeeds */ }
  if (!el.video.videoWidth) await new Promise(r => el.video.addEventListener("loadedmetadata", r, { once: true }));
  VW = el.video.videoWidth || 1280; VH = el.video.videoHeight || 720;
  el.camView.style.aspectRatio = `${VW} / ${VH}`;

  if (!landmarker) setStatus("loading", "Loading hand tracking", "Downloading the hand model. This takes a few seconds the first time.");
  try {
    landmarker = await model;
  } catch (err) {
    console.error(err);
    stopTracks();
    fallBackToMouse("Hand tracking didn't load", "Check your internet connection and reload.");
    return;
  }
  state.camOn = true;
}

function stopTracks() {
  camStream?.getTracks().forEach(t => t.stop());
  camStream = null;
  el.video.srcObject = null;
}

// Stops the webcam itself (its light goes off), not just the preview.
function stopCamera() {
  if (!state.camOn) return;
  stopTracks();
  state.camOn = false;
  lastVideoTime = -1;
  for (const h of hands) dropHand(h);
  ctx.ghost.clearRect(0, 0, el.ghost.width, el.ghost.height);
  el.camOverlay.width = el.camOverlay.width;   // clears the hand overlay
  statusKey = "";
  syncUI();
  toast("Camera off. Turn it back on with the Camera button.");
}

/* ---------- Camera preview: drag to move, drag the corner to resize ---------- */
const CAM_MIN_W = 160;
let camDrag = null;   // { kind: "move" | "resize", by: "mouse" | hand, sx, sy, start }

function camBox() {
  const w = clamp(state.camBox?.w ?? 232, CAM_MIN_W, Math.max(CAM_MIN_W, Math.min(860, W - 32)));
  return { x: state.camBox?.x ?? W - 16 - w, y: state.camBox?.y ?? 104, w };
}
// Size and position the preview, keeping it on screen.
function placeCam() {
  const b = camBox();
  el.cam.style.width = `${b.w}px`;
  b.x = clamp(b.x, 8, Math.max(8, W - b.w - 8));
  b.y = clamp(b.y, 8, Math.max(8, H - el.cam.offsetHeight - 8));
  el.cam.style.left = `${b.x}px`;
  el.cam.style.top = `${b.y}px`;
  return b;
}
function beginCamDrag(kind, x, y, by) {
  camDrag = { kind, by, sx: x, sy: y, start: placeCam() };
  el.cam.classList.add("dragging");
}
function moveCamDrag(x, y) {
  const { kind, sx, sy, start } = camDrag, dx = x - sx, dy = y - sy;
  state.camBox = kind === "move"
    ? { ...start, x: start.x + dx, y: start.y + dy }
    : { ...start, w: start.w + Math.max(dx, dy * VW / VH) };   // the corner follows whichever way you pull further
  placeCam();
}
function endCamDrag() {
  if (!camDrag) return;
  camDrag = null;
  el.cam.classList.remove("dragging");
  state.camBox = placeCam();
  saveSettings();
}
el.cam.addEventListener("pointerdown", e => {
  if (e.button > 0 || e.target.closest("[data-act]")) return;
  e.preventDefault();
  el.cam.setPointerCapture(e.pointerId);
  beginCamDrag(e.target.closest(".cam-grip") ? "resize" : "move", e.clientX, e.clientY, "mouse");
});
el.cam.addEventListener("pointermove", e => { if (camDrag?.by === "mouse") moveCamDrag(e.clientX, e.clientY); });
for (const type of ["pointerup", "pointercancel"]) {
  el.cam.addEventListener(type, () => { if (camDrag?.by === "mouse") endCamDrag(); });
}

function startMouse() {
  el.intro.hidden = true;
  state.mode = "mouse";
  statusKey = "";
  syncUI();
}

/* ---------- Rendering ---------- */
function resize() {
  DPR = Math.min(devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  for (const [cv, c] of [[el.ink, ctx.ink], [el.live, ctx.live], [el.cursors, ctx.cursors], [el.pCanvas, ctx.p]]) {
    cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
    c.setTransform(DPR, 0, 0, DPR, 0, 0);
  }
  el.ghost.width = Math.round(W / 2); el.ghost.height = Math.round(H / 2);
  placeCam();
  renderAll();
  if (state.view === "practice" && P.active && !P.finished) P.active.resize?.();
}

function renderLive() {
  const c = ctx.live;
  c.clearRect(0, 0, W, H);
  for (const s of allSources()) {
    if (s.stroke && s.stroke.tool !== "eraser") drawItem(c, s.stroke);
    if (s.shape) { c.save(); c.globalAlpha = 0.7; drawItem(c, s.shape); c.restore(); }
  }
  // Selection: the item being moved, its box, and a faint box on whatever the cursor is over.
  const selected = sel.id ? findItem(sel.id) : null;
  if (sel.id && !selected) sel.id = null;
  const off = selected || state.tool === "select" ? offsets() : null;
  if (selected) {
    const [ox, oy] = off.get(selected.id) || [0, 0];
    const dx = ox + (sel.drag?.dx || 0), dy = oy + (sel.drag?.dy || 0);
    if (sel.drag) { c.save(); c.translate(dx, dy); drawItem(c, selected); c.restore(); }
    outlineBox(c, itemBounds(selected), dx, dy, true);
  }
  for (const s of allSources()) {
    s.hover = null;
    if (state.tool !== "select" || sel.drag || !s.visible || s.overUI) continue;
    const hit = itemAt(s.x, s.y);
    if (!hit) continue;
    s.hover = hit.it.id;
    if (hit.it.id !== sel.id) outlineBox(c, itemBounds(hit.it), hit.dx, hit.dy, false);
  }
  placeSelBar(selected, off);
}

const pulses = [];
function pulse(x, y, r) { if (!reduceMotion) pulses.push({ x, y, r, t: performance.now() }); }

function renderCursors(now) {
  const c = ctx.cursors;
  c.clearRect(0, 0, W, H);
  const tool = TOOL[state.tool];
  for (const s of allSources()) {
    if (!s.visible) continue;
    if (state.view === "board" && tool.kind === "select" && !s.overUI) drawPointerCursor(c, s);
    else if (state.view === "board" && s.kind === "hand" && tool.kind === "shape" && !s.overUI && s.pose === "point" && !s.needsFresh) drawShapeCursor(c, s);
    else drawRingCursor(c, s, tool);
  }
  for (let i = pulses.length - 1; i >= 0; i--) {
    const p = pulses[i], k = (now - p.t) / 420;
    if (k >= 1) { pulses.splice(i, 1); continue; }
    c.beginPath(); c.arc(p.x, p.y, p.r + 18 * k, 0, TAU);
    c.lineWidth = 2.5 * (1 - k) + 0.5; c.strokeStyle = withAlpha(theme.accent, 1 - k); c.stroke();
  }
}

// Select tool: an arrow pointer whose tip is the fingertip, with a glow when it's over something it can grab.
function drawPointerCursor(c, s) {
  const paused = s.kind === "hand" && state.drawWith === "point" && s.pose === "fist";
  const grabbing = !!sel.drag && sel.drag.by === (s.kind === "mouse" ? "mouse" : s);
  c.save();
  c.translate(s.x, s.y);
  c.globalAlpha = paused ? 0.45 : 1;
  if (s.hover || grabbing) {
    c.beginPath(); c.arc(0, 0, 13, 0, TAU);
    c.fillStyle = withAlpha(theme.accent, grabbing ? 0.32 : 0.18); c.fill();
  }
  c.beginPath();
  c.moveTo(0, 0); c.lineTo(0, 17); c.lineTo(4.5, 13); c.lineTo(7.5, 19.5);
  c.lineTo(10, 18.4); c.lineTo(7, 12); c.lineTo(12.5, 12); c.closePath();
  c.fillStyle = theme.fg; c.fill();
  c.lineJoin = "round"; c.lineWidth = 1.5; c.strokeStyle = theme.board; c.stroke();
  c.restore();
}

// The cursor: a soft, nearly transparent disc sized to the tool.
function drawRingCursor(c, s, tool) {
  const practicing = state.view === "practice";
  const erasing = tool.id === "eraser" && !practicing;
  const texting = tool.kind === "text" && !practicing && !s.overUI;
  const paused = s.kind === "hand" && state.drawWith === "point" && s.pose === "fist" && !s.overUI;
  let r = s.overUI ? 13 : practicing ? 12 : texting ? 14 : erasing ? Math.max(10, toolWidth(tool) / 2) : Math.max(11, toolWidth(tool) / 2 + 5);
  if (paused) r = 8;
  const tint = s.overUI || erasing || practicing ? theme.fg : colorOf(state.color);
  c.save();
  c.globalAlpha = paused ? 0.45 : 1;
  c.beginPath(); c.arc(s.x, s.y, r, 0, TAU);
  c.fillStyle = withAlpha(tint, s.drawing ? 0.3 : 0.13); c.fill();
  c.lineWidth = 1.5;
  c.strokeStyle = s.overUI ? theme.accent : withAlpha(theme.fg, s.drawing ? 0.85 : 0.5);
  if (erasing && !s.overUI) c.setLineDash([4, 4]);
  c.stroke(); c.setLineDash([]);
  if (texting && !paused) {   // a text caret
    c.beginPath();
    c.moveTo(s.x, s.y - 7); c.lineTo(s.x, s.y + 7);
    c.moveTo(s.x - 3, s.y - 7); c.lineTo(s.x + 3, s.y - 7);
    c.moveTo(s.x - 3, s.y + 7); c.lineTo(s.x + 3, s.y + 7);
    c.lineWidth = 1.5; c.strokeStyle = withAlpha(theme.fg, 0.9); c.stroke();
  } else if (s.drawing || s.kind === "mouse") {
    c.beginPath(); c.arc(s.x, s.y, 1.8, 0, TAU); c.fillStyle = withAlpha(theme.fg, 0.9); c.fill();
  }
  const p = s.dwell?.progress || 0;
  if (p > 0.04) {
    c.beginPath(); c.arc(s.x, s.y, r + 7, -Math.PI / 2, -Math.PI / 2 + TAU * p);
    c.lineWidth = 3.5; c.lineCap = "round"; c.strokeStyle = theme.accent; c.stroke();
  }
  c.restore();
}

// Shape mode: a handle on the thumb tip and one on the index tip.
function drawShapeCursor(c, s) {
  const a = [s.tx, s.ty], b = [s.ix, s.iy];
  c.save();
  c.setLineDash([3, 5]); c.lineWidth = 1.25; c.strokeStyle = withAlpha(theme.fg, 0.4);
  c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); c.setLineDash([]);
  for (const [x, y] of [a, b]) {
    c.beginPath(); c.arc(x, y, 9, 0, TAU);
    c.fillStyle = withAlpha(colorOf(state.color), 0.16); c.fill();
    c.lineWidth = 1.5; c.strokeStyle = withAlpha(theme.fg, 0.6); c.stroke();
  }
  const p = s.stamp?.progress || 0;
  if (p > 0.15) {
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, k = clamp((p - 0.15) / 0.85, 0, 1);
    c.beginPath(); c.arc(mx, my, 16, 0, TAU); c.lineWidth = 3; c.strokeStyle = withAlpha(theme.fg, 0.15); c.stroke();
    c.beginPath(); c.arc(mx, my, 16, -Math.PI / 2, -Math.PI / 2 + TAU * k);
    c.lineCap = "round"; c.strokeStyle = theme.accent; c.stroke();
  }
  c.restore();
}

function drawPreview(list) {
  if (el.cam.classList.contains("collapsed")) return;
  const c = ctx.cam, cw = el.camOverlay.clientWidth, ch = el.camOverlay.clientHeight;
  if (!cw || !ch) return;
  if (el.camOverlay.width !== Math.round(cw * DPR)) {
    el.camOverlay.width = Math.round(cw * DPR); el.camOverlay.height = Math.round(ch * DPR);
  }
  c.setTransform(DPR, 0, 0, DPR, 0, 0);
  c.clearRect(0, 0, cw, ch);
  // The dashed box is the part of the frame that covers the board.
  c.setLineDash([4, 4]); c.lineWidth = 1; c.strokeStyle = "rgba(255, 255, 255, 0.55)";
  c.strokeRect(REACH.x * cw, REACH.top * ch, (1 - 2 * REACH.x) * cw, (1 - REACH.top - REACH.bottom) * ch);
  c.setLineDash([]);
  for (const lm of list) {
    const P = i => [(1 - lm[i].x) * cw, lm[i].y * ch];
    c.beginPath();
    for (const [a, b] of BONES) { const p = P(a), q = P(b); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]); }
    c.lineWidth = 1.5; c.strokeStyle = "rgba(255, 255, 255, 0.8)"; c.stroke();
    for (let i = 0; i < 21; i++) {
      const [x, y] = P(i), tip = i === 8 || i === 4;
      c.beginPath(); c.arc(x, y, tip ? 3.4 : 1.8, 0, TAU);
      c.fillStyle = tip ? theme.accent : "#ffffff"; c.fill();
    }
  }
}

function drawGhost() {
  const g = ctx.ghost, gw = el.ghost.width, gh = el.ghost.height;
  g.save();
  g.setTransform(-1, 0, 0, 1, gw, 0);
  g.drawImage(el.video, REACH.x * VW, REACH.top * VH, (1 - 2 * REACH.x) * VW, (1 - REACH.top - REACH.bottom) * VH, 0, 0, gw, gh);
  g.restore();
}

let lastVideoTime = -1;
function frame(now) {
  if (landmarker && el.video.readyState >= 2 && el.video.currentTime !== lastVideoTime) {
    lastVideoTime = el.video.currentTime;
    let res = null;
    try { res = landmarker.detectForVideo(el.video, now); } catch (err) { console.error(err); }
    if (res) {
      processHands(res, now);
      drawPreview((res.landmarks || []).slice(0, state.maxHands));
    }
    if (state.ghost && state.view === "board") drawGhost();
  }
  syncHover();
  if (state.view === "practice") { practiceTick(now); renderPractice(now); }
  else renderLive();
  renderCursors(now);
  updateStatus();
  requestAnimationFrame(frame);
}

/* ============================================================
   Practice: short games for hand control
   ============================================================ */
const DIFF = {
  //        trace lane  bubble   hold to pop  steady lane  pose hold  bubble drift  path bends
  easy:   { tol: 30, bubble: 58, hold: 500, lane: 48, pose: 500, drift: 0,  bends: 4 },
  medium: { tol: 20, bubble: 42, hold: 700, lane: 34, pose: 700, drift: 0,  bends: 6 },
  hard:   { tol: 12, bubble: 28, hold: 900, lane: 22, pose: 900, drift: 40, bends: 8 },
};
const DIFF_LABEL = { easy: "Easy", medium: "Medium", hard: "Hard" };
const TRACE_SETS = {
  easy:   ["line", "wave", "circle", "square", "triangle"],
  medium: ["wave", "circle", "zigzag", "star", "loop"],
  hard:   ["zigzag", "star", "spiral", "loop", "heart"],
};
const MESSAGES = [
  "Every round builds control. Try an easier level if this one felt hard.",
  "Good start. Keep practising to earn more stars.",
  "Nice work. One more star to go.",
  "Excellent control. Try the next difficulty when you're ready.",
];

const practiceLog = Object.assign({ best: {}, days: [], rounds: 0, stars: 0, difficulty: "easy" }, store.get("practice", {}) || {});
const P = {
  difficulty: DIFF[practiceLog.difficulty] ? practiceLog.difficulty : "easy",
  active: null, activeId: null, lastId: "trace", finished: false, resultTimer: 0,
  particles: [], popups: [], last: 0, hud: "", hint: "", track: "",
};

const steps = (n, f) => Array.from({ length: n + 1 }, (_, i) => f(i / n));
const fmtTime = ms => { const s = Math.round(ms / 1000); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const fmtSec = ms => `${(ms / 1000).toFixed(1)} s`;
const starsFor = (v, [one, two, three]) => v >= three ? 3 : v >= two ? 2 : v >= one ? 1 : 0;
const dayKey = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const starRow = n => `<span class="stars" aria-label="${n} of 3 stars">${[0, 1, 2].map(i => `<i class="${i < n ? "on" : ""}">★</i>`).join("")}</span>`;

function playArea() {
  const top = 108, bottom = 76, side = 40;
  const w = Math.max(240, W - side * 2), h = Math.max(240, H - top - bottom);
  return { x: side, y: top, w, h, cx: side + w / 2, cy: top + h / 2 };
}

// Evenly spaced points along a polyline, so distances and coverage are fair everywhere.
function resample(pts, step) {
  const out = [[pts[0][0], pts[0][1]]];
  let acc = 0;
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i];
    const seg = Math.hypot(bx - ax, by - ay);
    if (!seg) continue;
    let t = step - acc;
    while (t <= seg) { out.push([ax + (bx - ax) * t / seg, ay + (by - ay) * t / seg]); t += step; }
    acc = seg - (t - step);
  }
  const last = pts[pts.length - 1], tail = out[out.length - 1];
  if (Math.hypot(last[0] - tail[0], last[1] - tail[1]) > step * 0.3) out.push([last[0], last[1]]);
  return out;
}

function catmull(p, seg) {
  const out = [];
  for (let i = 0; i < p.length - 1; i++) {
    const p0 = p[Math.max(0, i - 1)], p1 = p[i], p2 = p[i + 1], p3 = p[Math.min(p.length - 1, i + 2)];
    for (let k = 0; k < seg; k++) {
      const t = k / seg, t2 = t * t, t3 = t2 * t;
      out.push([0, 1].map(j => 0.5 * (2 * p1[j] + (p2[j] - p0[j]) * t
        + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 + (3 * p1[j] - p0[j] - 3 * p2[j] + p3[j]) * t3)));
    }
  }
  out.push(p[p.length - 1].slice());
  return out;
}

function tracePoly(c, pts, from = 0, to = pts.length - 1, close = false) {
  c.beginPath();
  c.moveTo(pts[from][0], pts[from][1]);
  for (let i = from + 1; i <= to; i++) c.lineTo(pts[i][0], pts[i][1]);
  if (close) c.closePath();
}

function label(c, text, x, y, color = theme.fg) {
  c.save();
  c.font = "600 13px Figtree, system-ui, sans-serif";
  c.textAlign = "center"; c.textBaseline = "middle";
  c.fillStyle = withAlpha(color, 0.85);
  c.fillText(text, x, y);
  c.restore();
}

function setHud(t) { if (t !== P.hud) { P.hud = t; el.pHud.textContent = t; } }
function setHint(t) { if (t !== P.hint) { P.hint = t; el.pHint.textContent = t; el.pHint.hidden = !t; } }
function penHint(verb) {
  if (mouseOnly()) return `Hold the mouse button to ${verb}.`;
  return state.drawWith === "pinch" ? `Pinch to ${verb}.` : `Point your index finger to ${verb}, two fingers up to stop.`;
}

function burst(x, y, n = 24, color) {
  if (reduceMotion) return;
  const cols = COLORS.slice(1).map(c => c[state.board]);
  for (let i = 0; i < n; i++) {
    const a = Math.random() * TAU, sp = 140 + Math.random() * 320;
    P.particles.push({ x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 120, life: 0, max: 700 + Math.random() * 500, r: 2.5 + Math.random() * 3.5, col: color || cols[i % cols.length] });
  }
}
function popupText(text, x, y, color) { P.popups.push({ text, x, y, color, t: performance.now() }); }

/* ----- Trace the shape: precision ----- */
const SHAPES = {
  line: s => ({ name: "Line", pts: [[-s, 0], [s, 0]] }),
  wave: s => ({ name: "Wave", pts: steps(160, t => [-s + 2 * s * t, -0.3 * s * Math.sin(t * TAU * 1.5)]) }),
  circle: s => ({ name: "Circle", closed: true, pts: steps(200, t => [0.8 * s * Math.cos(t * TAU - Math.PI / 2), 0.8 * s * Math.sin(t * TAU - Math.PI / 2)]) }),
  square: s => { const a = 0.72 * s; return { name: "Square", closed: true, pts: [[-a, -a], [a, -a], [a, a], [-a, a], [-a, -a]] }; },
  triangle: s => ({ name: "Triangle", closed: true, pts: steps(3, t => [0.9 * s * Math.cos(t * TAU - Math.PI / 2), 0.9 * s * Math.sin(t * TAU - Math.PI / 2) + 0.22 * s]) }),
  zigzag: s => ({ name: "Zigzag", pts: steps(6, t => [-s + 2 * s * t, (Math.round(t * 6) % 2 ? 0.32 : -0.32) * s]) }),
  star: s => ({ name: "Star", closed: true, pts: steps(10, t => {
    const k = Math.round(t * 10), r = (k % 2 ? 0.4 : 0.92) * s, a = k * Math.PI / 5 - Math.PI / 2;
    return [r * Math.cos(a), r * Math.sin(a) + 0.09 * s];
  }) }),
  spiral: s => ({ name: "Spiral", pts: steps(320, t => {
    const a = t * TAU * 2.25, r = s * (0.1 + 0.8 * t);
    return [r * Math.cos(a - Math.PI / 2), r * Math.sin(a - Math.PI / 2)];
  }) }),
  loop: s => ({ name: "Figure eight", closed: true, pts: steps(260, t => [0.95 * s * Math.sin(t * TAU), 0.45 * s * Math.sin(2 * t * TAU)]) }),
  heart: s => {
    const k = s / 17 * 0.95;
    return { name: "Heart", closed: true, pts: steps(260, t => {
      const a = t * TAU;
      return [16 * Math.sin(a) ** 3 * k, (-(13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) - 2.5) * k];
    }) };
  },
};
const praise = s => s >= 95 ? "Perfect!" : s >= 85 ? "Great!" : s >= 70 ? "Nice!" : "Done";

function traceActivity() {
  const tol = DIFF[P.difficulty].tol, tol2 = tol * tol, list = TRACE_SETS[P.difficulty];
  const results = [];
  let idx = 0, cur = null, nextAt = 0;

  function load() {
    const A = playArea(), s = Math.min(A.w, A.h) * 0.4;
    const g = SHAPES[list[idx]](s);
    const pts = resample(g.pts.map(([x, y]) => [A.cx + x, A.cy + y]), 4);
    cur = {
      name: g.name, closed: !!g.closed, pts, covered: new Uint8Array(pts.length), nCov: 0,
      trail: [], prev: new Map(), onLen: 0, totalLen: 0, devSum: 0, startAt: 0, done: false,
    };
  }
  // Score one sample of the pen: is it on the line, and which part of the line does it cover?
  function sample(x, y, step) {
    const pts = cur.pts;
    let best = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i][0] - x, dy = pts[i][1] - y, q = dx * dx + dy * dy;
      if (q < best) best = q;
      if (q <= tol2 && !cur.covered[i]) { cur.covered[i] = 1; cur.nCov++; }
    }
    const dd = Math.sqrt(best);
    cur.totalLen += step;
    if (dd <= tol) cur.onLen += step;
    cur.devSum += Math.min(dd, tol * 3) * step;
    return dd <= tol;
  }
  function complete(now, skipped) {
    const coverage = cur.nCov / cur.pts.length;
    const accuracy = cur.totalLen ? cur.onLen / cur.totalLen : 0;
    const score = Math.round(100 * accuracy * (skipped ? coverage : 1));
    results.push({ name: cur.name, score, skipped, dev: cur.totalLen ? cur.devSum / cur.totalLen : null, ms: cur.startAt ? now - cur.startAt : 0 });
    cur.done = true;
    nextAt = now + 1300;
    const A = playArea();
    if (!skipped && score >= 60) burst(A.cx, A.cy, score >= 90 ? 44 : 26);
    popupText(skipped ? `Skipped · ${score}%` : `${praise(score)} ${score}%`, A.cx, A.y + 22, skipped ? theme.fg : theme.accent);
  }
  function finish() {
    const avg = Math.round(results.reduce((s, r) => s + r.score, 0) / results.length);
    const devs = results.filter(r => r.dev != null);
    const dev = devs.length ? devs.reduce((s, r) => s + r.dev, 0) / devs.length : 0;
    finishActivity({
      activity: "trace", title: "Trace the shape", score: avg, stars: starsFor(avg, [50, 75, 90]),
      headline: `${avg}% on the line`, label: `${avg}%`,
      details: [
        ["Shapes finished", `${results.filter(r => !r.skipped).length} of ${results.length}`],
        ["Average distance from the line", devs.length ? `${dev.toFixed(1)} px` : "–"],
        ["Tracing time", fmtTime(results.reduce((s, r) => s + r.ms, 0))],
      ],
      items: results.map(r => [r.name, r.skipped ? `${r.score}% skipped` : `${r.score}%`]),
    });
  }
  return {
    title: "Trace the shape",
    start() { load(); setHint(`${penHint("trace")} Green is on the line, red is off it.`); },
    skip(now) { if (!cur.done) complete(now, true); },
    resize() { if (!cur.done) load(); },
    tick(now, ptrs) {
      if (cur.done) {
        if (now >= nextAt) { idx++; if (idx >= list.length) finish(); else load(); }
        return;
      }
      const seen = new Set();
      for (const p of ptrs) {
        if (!p.pen) continue;
        seen.add(p.id);
        const prev = cur.prev.get(p.id);
        if (!prev) { cur.prev.set(p.id, [p.x, p.y]); cur.startAt ||= now; continue; }
        const len = Math.hypot(p.x - prev[0], p.y - prev[1]);
        if (len < 1.5) continue;
        const n = Math.ceil(len / 4);
        let a = prev;
        for (let k = 1; k <= n; k++) {
          const b = [prev[0] + (p.x - prev[0]) * k / n, prev[1] + (p.y - prev[1]) * k / n];
          cur.trail.push([a[0], a[1], b[0], b[1], sample(b[0], b[1], len / n)]);
          a = b;
        }
        cur.prev.set(p.id, [p.x, p.y]);
      }
      for (const id of [...cur.prev.keys()]) if (!seen.has(id)) cur.prev.delete(id);
      const cov = cur.nCov / cur.pts.length;
      setHud(`${cur.name} · shape ${idx + 1} of ${list.length} · ${Math.round(cov * 100)}% traced`);
      if (cov >= 0.96) complete(now, false);
    },
    render(c, now) {
      const pts = cur.pts, last = pts.length - 1;
      c.lineCap = "round"; c.lineJoin = "round";
      // The lane, which fills in as it's covered
      tracePoly(c, pts, 0, last, cur.closed);
      c.lineWidth = tol * 2; c.strokeStyle = withAlpha(theme.fg, 0.07); c.stroke();
      c.beginPath();
      for (let i = 1; i <= last; i++) {
        if (cur.covered[i] && cur.covered[i - 1]) { c.moveTo(pts[i - 1][0], pts[i - 1][1]); c.lineTo(pts[i][0], pts[i][1]); }
      }
      c.strokeStyle = withAlpha(theme.accent, cur.done ? 0.42 : 0.26); c.stroke();
      tracePoly(c, pts, 0, last, cur.closed);
      c.setLineDash([2, 9]); c.lineWidth = 2; c.strokeStyle = withAlpha(theme.fg, 0.5); c.stroke(); c.setLineDash([]);
      // The player's trace: green on the line, red off it
      c.lineWidth = 4;
      for (const on of [true, false]) {
        c.beginPath();
        for (const s of cur.trail) if (s[4] === on) { c.moveTo(s[0], s[1]); c.lineTo(s[2], s[3]); }
        c.strokeStyle = on ? theme.live : theme.danger; c.stroke();
      }
      if (!cur.startAt) {
        const [sx, sy] = pts[0], j = Math.min(16, last), [ax, ay] = pts[Math.max(0, j - 4)], [tx, ty] = pts[j];
        const ang = Math.atan2(ty - ay, tx - ax);
        c.beginPath(); c.arc(sx, sy, 8 + 2 * Math.sin(now / 260), 0, TAU); c.fillStyle = theme.live; c.fill();
        c.beginPath();
        c.moveTo(tx - 10 * Math.cos(ang - 0.6), ty - 10 * Math.sin(ang - 0.6)); c.lineTo(tx, ty);
        c.lineTo(tx - 10 * Math.cos(ang + 0.6), ty - 10 * Math.sin(ang + 0.6));
        c.lineWidth = 2.5; c.strokeStyle = theme.live; c.stroke();
        label(c, "Start", sx, sy - tol - 14);
      }
      if (!cur.closed) {
        c.beginPath(); c.arc(pts[last][0], pts[last][1], 7, 0, TAU);
        c.lineWidth = 2; c.strokeStyle = withAlpha(theme.fg, 0.6); c.stroke();
      }
    },
  };
}

/* ----- Bubble pop: reach and hold steady ----- */
function popActivity() {
  const d = DIFF[P.difficulty], ROUND = 45000;
  const goals = { easy: [8, 14, 20], medium: [6, 11, 16], hard: [4, 8, 12] }[P.difficulty];
  const reach = [];
  let A = playArea(), b = null, started = 0, popped = 0;

  function spawn(now, first) {
    const r = first ? Math.max(d.bubble, 60) : d.bubble;
    let x = A.cx, y = A.cy;
    if (!first) {
      for (let tries = 0; tries < 30; tries++) {
        x = A.x + r + Math.random() * (A.w - 2 * r);
        y = A.y + r + Math.random() * (A.h - 2 * r);
        if (!b || Math.hypot(x - b.x, y - b.y) > Math.min(A.w, A.h) * 0.3) break;
      }
    }
    const ang = Math.random() * TAU, col = COLORS[1 + Math.floor(Math.random() * (COLORS.length - 1))].key;
    b = { x, y, r, hold: 0, born: now, first, col, vx: first ? 0 : Math.cos(ang) * d.drift, vy: first ? 0 : Math.sin(ang) * d.drift };
  }
  function finish() {
    const avg = reach.length ? reach.reduce((s, v) => s + v, 0) / reach.length : 0;
    finishActivity({
      activity: "pop", title: "Bubble pop", score: popped, stars: starsFor(popped, goals),
      headline: `${popped} bubble${popped === 1 ? "" : "s"} popped`, label: `${popped} popped`,
      details: [
        ["Average time per bubble", reach.length ? fmtSec(avg) : "–"],
        ["Bubble size", `${d.bubble * 2} px`],
        ["Hold to pop", fmtSec(d.hold)],
      ],
      items: [["1 star", goals[0]], ["2 stars", goals[1]], ["3 stars", goals[2]]],
    });
  }
  return {
    title: "Bubble pop",
    start(now) {
      spawn(now, true);
      setHint(`Hold the cursor inside each bubble until it pops.${camReady() ? " Make a fist to rest." : ""}`);
    },
    resize() {
      A = playArea();
      b.x = clamp(b.x, A.x + b.r, A.x + A.w - b.r);
      b.y = clamp(b.y, A.y + b.r, A.y + A.h - b.r);
    },
    tick(now, ptrs, dt) {
      if (started && now - started >= ROUND) { finish(); return; }
      if (b.vx || b.vy) {
        b.x += b.vx * dt / 1000; b.y += b.vy * dt / 1000;
        if (b.x < A.x + b.r || b.x > A.x + A.w - b.r) { b.vx *= -1; b.x = clamp(b.x, A.x + b.r, A.x + A.w - b.r); }
        if (b.y < A.y + b.r || b.y > A.y + A.h - b.r) { b.vy *= -1; b.y = clamp(b.y, A.y + b.r, A.y + A.h - b.r); }
      }
      const inside = ptrs.some(p => !p.rest && Math.hypot(p.x - b.x, p.y - b.y) <= b.r);
      b.hold = inside ? b.hold + dt : Math.max(0, b.hold - dt * 1.5);
      if (b.hold >= d.hold) {
        burst(b.x, b.y, 22, colorOf(b.col));
        if (b.first) { started = now; popupText("Go!", b.x, b.y - b.r - 16, theme.accent); }
        else { popped++; reach.push(now - b.born); popupText("+1", b.x, b.y - b.r - 16, colorOf(b.col)); }
        spawn(now, false);
      }
      setHud(started ? `${popped} popped · ${Math.ceil((ROUND - (now - started)) / 1000)} s left` : "Pop the start bubble when you're ready");
    },
    render(c, now) {
      if (started) {
        const k = clamp(1 - (now - started) / ROUND, 0, 1);
        c.fillStyle = withAlpha(theme.fg, 0.1); c.fillRect(A.x, A.y - 18, A.w, 4);
        c.fillStyle = theme.accent; c.fillRect(A.x, A.y - 18, A.w * k, 4);
      }
      const col = colorOf(b.col), r = b.r * (1 + 0.03 * Math.sin(now / 240 + b.born));
      const g = c.createRadialGradient(b.x - r * 0.35, b.y - r * 0.4, r * 0.1, b.x, b.y, r);
      g.addColorStop(0, withAlpha(col, 0.55)); g.addColorStop(1, withAlpha(col, 0.16));
      c.beginPath(); c.arc(b.x, b.y, r, 0, TAU); c.fillStyle = g; c.fill();
      c.lineWidth = 2; c.strokeStyle = withAlpha(col, 0.9); c.stroke();
      c.beginPath(); c.arc(b.x - r * 0.38, b.y - r * 0.4, r * 0.14, 0, TAU); c.fillStyle = "rgba(255, 255, 255, 0.55)"; c.fill();
      if (b.hold > 0) {
        c.beginPath(); c.arc(b.x, b.y, r + 9, -Math.PI / 2, -Math.PI / 2 + TAU * Math.min(1, b.hold / d.hold));
        c.lineWidth = 4; c.lineCap = "round"; c.strokeStyle = theme.accent; c.stroke();
      }
      if (b.first) label(c, "Start", b.x, b.y);
    },
  };
}

/* ----- Steady path: keep inside a winding lane ----- */
function steadyActivity() {
  const d = DIFF[P.difficulty], hw = d.lane;
  let path = [], N = 0, phase = "ready", prog = 0, bumps = 0, startAt = 0, readyHold = 0, flashAt = -1e9, lostAt = 0;

  function build() {
    const A = playArea(), m = hw + 28, ctrl = [];
    for (let i = 0; i <= d.bends; i++) {
      const x = A.x + m + (A.w - 2 * m) * i / d.bends;
      const y = i === 0 || i === d.bends ? A.cy : A.y + m + Math.random() * (A.h - 2 * m);
      ctrl.push([x, y]);
    }
    path = resample(catmull(ctrl, 30), 4);
    N = path.length;
    phase = "ready"; prog = 0; bumps = 0; startAt = 0; readyHold = 0; flashAt = -1e9; lostAt = 0;
  }
  function finish(now) {
    const ms = now - startAt, par = N * 4 / 160 * 1000;   // par: 160 px a second
    const score = clamp(Math.round(100 - bumps * 10 - Math.max(0, ms - par) / 500), 0, 100);
    const [fx, fy] = path[N - 1];
    burst(fx, fy, 36);
    finishActivity({
      activity: "steady", title: "Steady path", score, stars: starsFor(score, [40, 70, 90]),
      headline: bumps ? `${bumps} bump${bumps === 1 ? "" : "s"} in ${fmtTime(ms)}` : `No bumps in ${fmtTime(ms)}`,
      label: `${score} points`,
      details: [["Score", `${score} / 100`], ["Bumps", String(bumps)], ["Time", fmtTime(ms)], ["Lane width", `${hw * 2} px`]],
    });
  }
  return {
    title: "Steady path",
    start() { build(); setHint("Hold the start circle, then guide the dot to the finish without touching the walls."); },
    resize() { build(); },
    tick(now, ptrs, dt) {
      const p = ptrs.find(q => !q.rest);
      if (!p) {
        // Lost the hand mid-run: wait for it back at the dot, without counting a bump.
        if (phase === "run") { lostAt ||= now; if (now - lostAt > 300) phase = "off"; }
        return;
      }
      lostAt = 0;
      if (phase === "ready") {
        readyHold = Math.hypot(p.x - path[0][0], p.y - path[0][1]) <= hw + 10 ? readyHold + dt : 0;
        if (readyHold >= 600) { phase = "run"; startAt = now; popupText("Go!", path[0][0], path[0][1] - hw - 22, theme.accent); }
        setHud("Hold the start circle to begin");
        return;
      }
      let best = prog, bd = Infinity;
      for (let i = Math.max(0, prog - 40), hi = Math.min(N - 1, prog + 60); i <= hi; i++) {
        const dx = path[i][0] - p.x, dy = path[i][1] - p.y, q = dx * dx + dy * dy;
        if (q < bd) { bd = q; best = i; }
      }
      const dd = Math.sqrt(bd);
      if (phase === "run") {
        if (dd <= hw) prog = Math.max(prog, best);
        else { phase = "off"; bumps++; flashAt = now; popupText("Bump!", p.x, p.y - 26, theme.danger); }
      } else if (dd <= hw * 0.85 && Math.abs(best - prog) <= 20) {
        phase = "run";
      }
      if (prog >= N - 4) { finish(now); return; }
      setHud(phase === "off" ? `Bumps ${bumps} · go back to the glowing dot` : `Bumps ${bumps} · ${fmtTime(now - startAt)}`);
    },
    render(c, now) {
      c.lineCap = "round"; c.lineJoin = "round";
      const flash = now - flashAt < 400;
      tracePoly(c, path);
      c.lineWidth = hw * 2 + 8; c.strokeStyle = flash ? withAlpha(theme.danger, 0.85) : withAlpha(theme.fg, 0.32); c.stroke();
      c.lineWidth = hw * 2; c.strokeStyle = theme.board; c.stroke();
      if (prog > 0) {
        tracePoly(c, path, 0, prog);
        c.strokeStyle = withAlpha(theme.accent, 0.2); c.stroke();
      }
      tracePoly(c, path, prog, N - 1);
      c.setLineDash([2, 10]); c.lineWidth = 2; c.strokeStyle = withAlpha(theme.fg, 0.3); c.stroke(); c.setLineDash([]);
      const [fx, fy] = path[N - 1];
      c.beginPath(); c.arc(fx, fy, Math.max(12, hw * 0.75), 0, TAU);
      c.fillStyle = withAlpha(theme.live, 0.22); c.fill(); c.lineWidth = 2; c.strokeStyle = theme.live; c.stroke();
      label(c, "Finish", fx, fy - hw - 18);
      if (phase === "ready") {
        const [sx, sy] = path[0];
        c.beginPath(); c.arc(sx, sy, hw + 10, 0, TAU); c.fillStyle = withAlpha(theme.live, 0.18); c.fill();
        c.beginPath(); c.arc(sx, sy, hw + 10, -Math.PI / 2, -Math.PI / 2 + TAU * Math.min(1, readyHold / 600));
        c.lineWidth = 4; c.strokeStyle = theme.accent; c.stroke();
        label(c, "Start", sx, sy - hw - 24);
      }
      const [bx, by] = path[prog], grow = phase === "off" ? 1 + 0.3 * Math.sin(now / 110) : 1;
      c.save();
      c.shadowColor = theme.accent; c.shadowBlur = phase === "off" ? 22 : 12;
      c.beginPath(); c.arc(bx, by, 10 * grow, 0, TAU); c.fillStyle = theme.accent; c.fill();
      c.restore();
    },
  };
}

/* ----- Finger moves: finger isolation and pinch ----- */
const same = (h, want) => want.every((v, i) => !!v === h.ext[i]);
const MOVES = {
  fist:  { label: "Make a fist",         glyph: () => handGlyph({ fingers: [0, 0, 0, 0] }),                  test: h => same(h, [0, 0, 0, 0]) },
  one:   { label: "Point one finger",    glyph: () => handGlyph({ fingers: [1, 0, 0, 0] }),                  test: h => same(h, [1, 0, 0, 0]) },
  two:   { label: "Show two fingers",    glyph: () => handGlyph({ fingers: [1, 1, 0, 0] }),                  test: h => same(h, [1, 1, 0, 0]) },
  three: { label: "Show three fingers",  glyph: () => handGlyph({ fingers: [1, 1, 1, 0] }),                  test: h => same(h, [1, 1, 1, 0]) },
  open:  { label: "Open your hand wide", glyph: () => handGlyph({ fingers: [1, 1, 1, 1], thumb: "out" }),    test: h => same(h, [1, 1, 1, 1]) },
  pinch: { label: "Make an OK sign",     glyph: () => handGlyph({ fingers: [1, 1, 1, 1], thumb: "pinch" }),  test: h => h.pinched && h.ext[1] && h.ext[2] },
};
const MOVE_SETS = {
  easy: ["fist", "one", "open"],
  medium: ["fist", "one", "two", "open", "pinch"],
  hard: ["fist", "one", "two", "three", "open", "pinch"],
};
function describeHand(h) {
  if (h.pinched && h.ext[1] && h.ext[2]) return "an OK sign";
  const n = h.ext.filter(Boolean).length;
  if (n === 0) return "a fist";
  if (n === 4) return "an open hand";
  if (same(h, [1, 0, 0, 0])) return "one finger";
  if (same(h, [1, 1, 0, 0])) return "two fingers";
  if (same(h, [1, 1, 1, 0])) return "three fingers";
  return `${n} finger${n === 1 ? "" : "s"} up`;
}

function fingersActivity() {
  const d = DIFF[P.difficulty], set = MOVE_SETS[P.difficulty], total = 10, RING = 2 * Math.PI * 96;
  const seq = [];
  while (seq.length < total) {
    const m = set[Math.floor(Math.random() * set.length)];
    if (m !== seq[seq.length - 1]) seq.push(m);
  }
  const times = [];
  let i = 0, hold = 0, shownAt = 0, waitUntil = 0, ring = null, seen = null, seenText = "";

  function show(now) {
    const m = MOVES[seq[i]];
    el.pStage.hidden = false;
    el.pStage.innerHTML = `<div class="f-prompt"><span class="eyebrow">Move ${i + 1} of ${total}</span>`
      + `<div class="f-glyph"><svg class="f-ring" viewBox="0 0 220 220" aria-hidden="true"><circle cx="110" cy="110" r="96" class="f-track"/>`
      + `<circle cx="110" cy="110" r="96" class="f-fill" stroke-dasharray="${RING}" stroke-dashoffset="${RING}"/></svg>${m.glyph()}</div>`
      + `<strong class="f-label">${m.label}</strong><span class="f-seen"></span></div>`;
    ring = el.pStage.querySelector(".f-fill");
    seen = el.pStage.querySelector(".f-seen");
    seenText = ""; shownAt = now; hold = 0;
    setHud(`Move ${i + 1} of ${total} · ${m.label}`);
  }
  function finish() {
    const avg = times.reduce((s, v) => s + v, 0) / times.length;
    const score = clamp(Math.round(100 * (6000 - avg) / 5000), 0, 100);
    finishActivity({
      activity: "fingers", title: "Finger moves", score, stars: starsFor(score, [35, 60, 80]),
      headline: `${fmtSec(avg)} per move`, label: `${fmtSec(avg)} per move`,
      details: [["Moves", String(total)], ["Fastest", fmtSec(Math.min(...times))], ["Slowest", fmtSec(Math.max(...times))], ["Hold time", fmtSec(d.pose)]],
    });
  }
  return {
    title: "Finger moves",
    // The first prompt appears on the first tick, so every move is timed on the frame clock.
    start() { setHint("Copy the hand shape and hold it until the ring fills."); },
    tick(now, ptrs, dt) {
      if (!ring) { if (now >= waitUntil) show(now); return; }
      const vis = hands.filter(h => h.visible);
      hold = vis.some(h => MOVES[seq[i]].test(h)) ? hold + dt : Math.max(0, hold - dt * 2);
      ring.setAttribute("stroke-dashoffset", RING * (1 - Math.min(1, hold / d.pose)));
      const t = vis.length ? `I can see ${describeHand(vis[0])}` : "Show your hand to the camera";
      if (t !== seenText) { seenText = t; seen.textContent = t; }
      if (hold >= d.pose) {
        times.push(now - shownAt);
        el.pStage.querySelector(".f-prompt").classList.add("done");
        burst(W / 2, H / 2, 24);
        ring = null; i++;
        if (i >= total) finish(); else waitUntil = now + 650;
      }
    },
    render() {},
  };
}

const ACTIVITY_MAKERS = { trace: traceActivity, pop: popActivity, steady: steadyActivity, fingers: fingersActivity };
const ART = {
  trace: '<svg viewBox="0 0 64 64" aria-hidden="true"><path class="a-guide" d="M32 8l5.9 16.9 17.9.4-14.3 10.8 5.2 17.1L32 43l-14.7 10.2 5.2-17.1L8.2 25.3l17.9-.4z"/><path class="a-ink" d="M32 8l5.9 16.9 17.9.4-14.3 10.8"/></svg>',
  pop: () => {
    const b = (cx, cy, r, key) => `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${withAlpha(colorOf(key), 0.3)}" stroke="${colorOf(key)}" stroke-width="2"/>`;
    return `<svg viewBox="0 0 64 64" aria-hidden="true">${b(24, 38, 15, "blue")}${b(45, 20, 10, "pink")}${b(49, 47, 6, "green")}<path class="a-ink" d="M24 18.5a19.5 19.5 0 0 1 19.5 19.5"/></svg>`;
  },
  steady: '<svg viewBox="0 0 64 64" aria-hidden="true"><path class="a-wall" d="M10 42c8 0 8-20 16-20s8 20 16 20 8-20 12-20"/><path class="a-lane" d="M10 42c8 0 8-20 16-20s8 20 16 20 8-20 12-20"/><circle class="a-dot" cx="26" cy="22" r="4.5"/></svg>',
};
const PRACTICE = [
  { id: "trace", title: "Trace the shape", skill: "Precision", desc: "Follow each outline inside the lane. Five shapes a round.", art: () => ART.trace },
  { id: "pop", title: "Bubble pop", skill: "Reach and hold", desc: "Hold the cursor inside each bubble until it pops. 45 seconds.", art: ART.pop },
  { id: "steady", title: "Steady path", skill: "Steadiness", desc: "Guide the dot along a winding lane without touching the walls.", art: () => ART.steady },
  { id: "fingers", title: "Finger moves", skill: "Finger control", desc: "Copy the hand shape on screen and hold it. Ten moves a round.", art: () => handGlyph({ fingers: [1, 1, 0, 0] }), camera: true },
];

function streakDays() {
  const days = new Set(practiceLog.days), d = new Date();
  if (!days.has(dayKey(d))) d.setDate(d.getDate() - 1);   // a streak survives until the end of today
  let n = 0;
  while (days.has(dayKey(d))) { n++; d.setDate(d.getDate() - 1); }
  return n;
}

function recordRound(res) {
  const key = `${res.activity}:${P.difficulty}`, prev = practiceLog.best[key] || null;
  const isBest = !prev || res.score > prev.score;
  if (isBest) practiceLog.best[key] = { score: res.score, stars: res.stars, label: res.label };
  practiceLog.rounds++;
  practiceLog.stars += res.stars;
  const today = dayKey(new Date());
  if (practiceLog.days[practiceLog.days.length - 1] !== today) practiceLog.days = [...practiceLog.days, today].slice(-120);
  store.set("practice", practiceLog);
  return { isBest, prev };
}

function renderMenu() {
  for (const b of el.pDiff.querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.arg === P.difficulty);
  el.pCards.innerHTML = PRACTICE.map(a => {
    const best = practiceLog.best[`${a.id}:${P.difficulty}`];
    const off = a.camera && !camReady();
    const foot = off ? "Needs the camera" : best ? `${starRow(best.stars)}<span>Best: ${best.label}</span>` : "Not played yet";
    return `<button class="p-card" data-act="practice-start" data-arg="${a.id}"${off ? " disabled" : ""}>`
      + `<span class="p-art">${a.art()}</span><span class="eyebrow">${a.skill}</span><strong>${a.title}</strong>`
      + `<span class="p-desc">${a.desc}</span><span class="p-best">${foot}</span></button>`;
  }).join("");
  const streak = streakDays(), n = practiceLog.rounds;
  el.pStats.innerHTML = n
    ? `<b>${n}</b> round${n === 1 ? "" : "s"} played · <b>${practiceLog.stars}</b> stars earned · <b>${streak}</b>-day streak`
    : "Your rounds, stars and daily streak will show here.";
}

function practiceOpen() {
  if (state.view === "practice") return;
  endAllStrokes();
  state.view = "practice";
  el.help.hidden = true;
  el.practice.hidden = false;
  mouse.visible = false;
  showMenu();
}
function practiceExit() {
  stopActivity();
  state.view = "board";
  el.practice.hidden = true;
  Object.assign(mouse, { down: false, drawing: false, visible: false });
  endAllStrokes();
  statusKey = "";
}
function stopActivity() {
  clearTimeout(P.resultTimer);
  P.active = null; P.activeId = null; P.finished = false;
  P.particles.length = 0; P.popups.length = 0;
  el.pStage.innerHTML = ""; el.pStage.hidden = true;
  el.pSkip.hidden = true;
  setHint("");
}
function showMenu() {
  stopActivity();
  renderMenu();
  el.pResult.hidden = true;
  el.pMenu.hidden = false;
  el.pMenu.dataset.armAt = performance.now() + 600;
  el.pMenuBtn.hidden = true;
  el.pEyebrow.textContent = "Practice";
  setHud("Choose an activity");
}
function startActivity(id) {
  if (!ACTIVITY_MAKERS[id]) return;
  if (id === "fingers" && !camReady()) { toast("Finger moves needs the camera"); return; }
  stopActivity();
  P.activeId = P.lastId = id;
  P.active = ACTIVITY_MAKERS[id]();
  el.pMenu.hidden = true;
  el.pResult.hidden = true;
  el.pMenuBtn.hidden = false;
  el.pSkip.hidden = id !== "trace";
  el.pEyebrow.textContent = `${P.active.title} · ${DIFF_LABEL[P.difficulty]}`;
  P.active.start(performance.now());
}
function finishActivity(res) {
  P.finished = true;
  el.pSkip.hidden = true;
  const rec = recordRound(res), id = P.activeId;
  P.resultTimer = setTimeout(() => { if (P.activeId === id && P.finished) showResult(res, rec); }, 800);
}
function showResult(res, rec) {
  el.pStage.hidden = true;
  el.rEyebrow.textContent = `${res.title} · ${DIFF_LABEL[P.difficulty]}`;
  el.rStars.innerHTML = [0, 1, 2].map(i => `<i class="${i < res.stars ? "on" : ""}" style="--d:${i * 140}ms">★</i>`).join("");
  el.rStars.setAttribute("aria-label", `${res.stars} of 3 stars`);
  el.rHeadline.textContent = res.headline;
  el.rMessage.textContent = res.stars === 3 && P.difficulty === "hard" ? "Excellent control. That's top marks on Hard." : MESSAGES[res.stars];
  el.rBest.textContent = !rec.prev ? "First round at this level. That's the score to beat."
    : rec.isBest ? `New personal best! Previous: ${rec.prev.label}` : `Your best: ${rec.prev.label}`;
  el.rBest.classList.toggle("is-best", rec.isBest && !!rec.prev);
  el.rDetails.innerHTML = res.details.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  el.rItems.innerHTML = (res.items || []).map(([k, v]) => `<span class="r-item">${k} <b>${v}</b></span>`).join("");
  el.rItems.hidden = !res.items?.length;
  el.pResult.dataset.armAt = performance.now() + 900;
  el.pResult.hidden = false;
  el.pMenuBtn.hidden = true;
  setHud("Round complete");
  setHint("");
  if (res.stars === 3) burst(W / 2, H * 0.3, 60);
}

// Mouse and touch play too.
el.pCanvas.addEventListener("pointerdown", e => {
  if (e.button > 0) return;
  el.pCanvas.setPointerCapture(e.pointerId);
  Object.assign(mouse, { down: true, drawing: true, visible: true, x: e.clientX, y: e.clientY });
});
el.pCanvas.addEventListener("pointermove", e => Object.assign(mouse, { visible: true, x: e.clientX, y: e.clientY }));
for (const type of ["pointerup", "pointercancel"]) el.pCanvas.addEventListener(type, () => Object.assign(mouse, { down: false, drawing: false }));
el.pCanvas.addEventListener("pointerleave", () => { if (!mouse.down) mouse.visible = false; });

function practicePointers() {
  const out = [];
  if (mouse.visible) out.push({ id: "mouse", x: mouse.x, y: mouse.y, pen: mouse.down, rest: false });
  for (const h of hands) {
    if (!h.visible || h.overUI) continue;
    out.push({ id: h.id, x: h.x, y: h.y, pen: !!h.pen, rest: state.drawWith === "point" && h.pose === "fist" });
  }
  return out;
}

function practiceTick(now) {
  const dt = clamp(now - (P.last || now), 0, 50);
  P.last = now;
  const track = camReady() ? (hands.some(h => h.visible) ? "Hand found" : "Show your hand")
    : camStarting ? "Camera starting"
    : state.mode === "camera" && landmarker ? "Camera off" : "Mouse mode";
  if (track !== P.track) {
    P.track = track;
    el.pTrackText.textContent = track;
    el.pTrack.dataset.state = track === "Hand found" ? "on" : "off";
  }
  if (P.active && !P.finished) P.active.tick(now, practicePointers(), dt);
  for (let i = P.particles.length - 1; i >= 0; i--) {
    const p = P.particles[i];
    p.life += dt;
    if (p.life >= p.max) { P.particles.splice(i, 1); continue; }
    p.x += p.vx * dt / 1000; p.y += p.vy * dt / 1000;
    p.vy += 520 * dt / 1000; p.vx *= 0.99;
  }
  P.popups = P.popups.filter(p => performance.now() - p.t < 1100);
}

function renderPractice(now) {
  const c = ctx.p;
  c.clearRect(0, 0, W, H);
  if (P.active) { c.save(); P.active.render(c, now); c.restore(); }
  for (const p of P.particles) {
    const k = p.life / p.max;
    c.globalAlpha = 1 - k;
    c.beginPath(); c.arc(p.x, p.y, p.r * (1 - k * 0.5), 0, TAU);
    c.fillStyle = p.col; c.fill();
  }
  c.globalAlpha = 1;
  const t = performance.now();
  for (const p of P.popups) {
    const k = (t - p.t) / 1100;
    c.save();
    c.globalAlpha = clamp(1 - k, 0, 1);
    c.font = "700 24px Figtree, system-ui, sans-serif";
    c.textAlign = "center"; c.textBaseline = "middle";
    c.fillStyle = p.color || theme.fg;
    c.fillText(p.text, p.x, p.y - 36 * k);
    c.restore();
  }
}

/* ---------- Boot ---------- */
function makeGrain() {
  const c = document.createElement("canvas");
  c.width = c.height = 140;
  const g = c.getContext("2d"), img = g.createImageData(140, 140);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = Math.random() * 14;
  }
  g.putImageData(img, 0, 0);
  document.documentElement.style.setProperty("--grain", `url(${c.toDataURL()})`);
}

// Drawings saved before the Select tool existed need ids so they can be moved.
for (const it of state.items) if (!it.id && (it.type === "stroke" || it.type === "shape")) it.id = newId();

buildUI();
makeGrain();
syncUI();
resize();
// Canvas text only uses Kalam once it has loaded, so redraw saved text when it arrives.
document.fonts?.load(textFont(28)).then(() => { if (state.items.some(it => it.type === "text")) renderAll(); }).catch(() => {});
addEventListener("resize", resize);
requestAnimationFrame(frame);
