/**
 * Sim tab — canvas renderer + WebSocket client for /sim/events.
 *
 * Coordinate system: world cm (x right, y up). The canvas has y flipped
 * (pixel y down). We compute a viewport transform once per resize that
 * maps world coordinates to canvas pixels with a fixed margin.
 */
'use strict';

(function () {

// --- State -----------------------------------------------------------------
let ws = null;
let walls = [];          // [[ax,ay],[bx,by]] in world cm
let lastMsg = null;      // most recent sim frame
let vp = null;           // viewport transform {ox, oy, scale}

const canvas = document.getElementById('sim-canvas');
const ctx    = canvas.getContext('2d');

// Keyboard drive state
const keys = {};
window.addEventListener('keydown', e => { keys[e.code] = true; });
window.addEventListener('keyup',   e => { keys[e.code] = false; });
let driveTick = null;

// JWT — stored in sessionStorage so the page survives soft refreshes.
function getJWT() { return sessionStorage.getItem('pysaac_jwt') || ''; }

// --- WebSocket -------------------------------------------------------------
function connectSimWS() {
  const jwt = getJWT();
  if (!jwt) {
    pysaac.log('No JWT in sessionStorage["pysaac_jwt"] — sim WS skipped', 'err');
    return;
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/sim/events?token=${encodeURIComponent(jwt)}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    pysaac.setConnected(true);
    pysaac.log('Sim WS connected', 'ok');
    fetchWalls();
    startDriveLoop();
  };

  ws.onmessage = e => {
    try { handleSimMsg(JSON.parse(e.data)); }
    catch (_) {}
  };

  ws.onclose = (ev) => {
    pysaac.setConnected(false);
    pysaac.log(`Sim WS closed (${ev.code})`, ev.wasClean ? '' : 'err');
    stopDriveLoop();
    // Reconnect after 3 s.
    setTimeout(connectSimWS, 3000);
  };

  ws.onerror = () => pysaac.log('Sim WS error', 'err');
}

async function fetchWalls() {
  const jwt = getJWT();
  try {
    const r = await fetch('/sim/state', { headers: { Authorization: 'Bearer ' + jwt } });
    if (!r.ok) { pysaac.log('fetchWalls: ' + r.status, 'err'); return; }
    const data = await r.json();
    walls = data.walls || [];
    vp = null;   // force recompute
    pysaac.log(`Loaded ${walls.length} wall segments`);
  } catch (e) {
    pysaac.log('fetchWalls failed: ' + e, 'err');
  }
}

// --- Keyboard drive --------------------------------------------------------
function startDriveLoop() {
  if (driveTick) return;
  driveTick = setInterval(sendDriveCmd, 80);
}
function stopDriveLoop() {
  if (driveTick) { clearInterval(driveTick); driveTick = null; }
}

async function sendDriveCmd() {
  // Check if any drive key is held; skip the POST if idle (saves RTTs).
  const fwd = keys['KeyW'] || keys['ArrowUp'];
  const rev = keys['KeyS'] || keys['ArrowDown'];
  const lft = keys['KeyA'] || keys['ArrowLeft'];
  const rgt = keys['KeyD'] || keys['ArrowRight'];
  if (!fwd && !rev && !lft && !rgt) return;

  const throttle = fwd ? 0.8 : rev ? -0.5 : 0.0;
  const steer    = lft ? -0.6 : rgt ? 0.6 : 0.0;
  const jwt = getJWT();
  try {
    await fetch('/sim/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + jwt },
      body: JSON.stringify({ throttle, steer }),
    });
  } catch (_) {}
}

// --- Render ----------------------------------------------------------------
function handleSimMsg(msg) {
  if (msg.kind !== 'sim') return;
  lastMsg = msg;

  // Update HUD stats.
  document.getElementById('ctrl-v').textContent     = (msg.v || 0).toFixed(1) + ' cm/s';
  document.getElementById('ctrl-steer').textContent = ((msg.steer || 0) * 180 / Math.PI).toFixed(1) + '°';
  const colEl = document.getElementById('sim-collided');
  colEl.style.display = msg.collided ? 'inline' : 'none';

  draw(msg);
}

function ensureViewport() {
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  if (!walls.length) {
    // Default oval bounds (cm): 0..~490 x 0..~240
    vp = { ox: 20, oy: 20, scale: Math.min((W - 40) / 490, (H - 40) / 240) };
    return;
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const w of walls) {
    for (const p of w) {
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    }
  }
  const ww = maxX - minX || 1, wh = maxY - minY || 1;
  const margin = 20;
  const scale = Math.min((W - 2*margin) / ww, (H - 2*margin) / wh);
  vp = { ox: margin - minX * scale, oy: margin - minY * scale, scale };
}

/** World cm → canvas px */
function wx(x) { return vp.ox + x * vp.scale; }
function wy(y) { return canvas.height - (vp.oy + y * vp.scale); }  // flip y

function draw(msg) {
  ensureViewport();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Walls
  ctx.strokeStyle = '#555';
  ctx.lineWidth = Math.max(1, vp.scale * 2);
  for (const w of walls) {
    ctx.beginPath();
    ctx.moveTo(wx(w[0][0]), wy(w[0][1]));
    ctx.lineTo(wx(w[1][0]), wy(w[1][1]));
    ctx.stroke();
  }

  const pose = msg.pose;
  if (!pose) return;

  // Sensor rays
  const SENSORS = [
    { key: 'center', color: '#4a9eff' },
    { key: 'left',   color: '#3ecf8e' },
    { key: 'right',  color: '#3ecf8e' },
  ];
  for (const { key, color } of SENSORS) {
    const s = msg.lidar?.[key];
    if (!s) continue;
    const o = s.origin, h = s.hit;
    if (!o) continue;
    const endX = h ? h[0] : (o[0] + s.distance_cm * Math.cos(pose.theta));
    const endY = h ? h[1] : (o[1] + s.distance_cm * Math.sin(pose.theta));
    const frac = Math.min(1, s.distance_cm / 800);
    ctx.strokeStyle = `rgba(${hexToRgb(color)}, ${0.3 + 0.7 * (1 - frac)})`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(wx(o[0]), wy(o[1])); ctx.lineTo(wx(endX), wy(endY)); ctx.stroke();
    if (h) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(wx(h[0]), wy(h[1]), 2, 0, 2*Math.PI); ctx.fill(); }
  }
  for (const { key, color } of [{key:'left',color:'#f0c040'},{key:'right',color:'#f0c040'}]) {
    const s = msg.ir?.[key];
    if (!s || !s.valid) continue;
    const o = s.origin, h = s.hit;
    if (!o) continue;
    ctx.strokeStyle = 'rgba(240,192,64,0.35)';
    ctx.lineWidth = 1;
    const endX = h ? h[0] : o[0]; const endY = h ? h[1] : o[1];
    ctx.beginPath(); ctx.moveTo(wx(o[0]), wy(o[1])); ctx.lineTo(wx(endX), wy(endY)); ctx.stroke();
  }

  // Robot rectangle (chassis 29.5 × 19 cm)
  const L = 29.5, W = 19, hl = L/2, hw = W/2;
  const corners = [
    rotPt(+hl, +hw, pose.theta), rotPt(+hl, -hw, pose.theta),
    rotPt(-hl, -hw, pose.theta), rotPt(-hl, +hw, pose.theta),
  ].map(([bx, by]) => [pose.x + bx, pose.y + by]);
  ctx.fillStyle   = 'rgba(74,158,255,0.25)';
  ctx.strokeStyle = msg.collided ? '#ff4d4d' : '#4a9eff';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(wx(corners[0][0]), wy(corners[0][1]));
  for (let i = 1; i < 4; i++) ctx.lineTo(wx(corners[i][0]), wy(corners[i][1]));
  ctx.closePath(); ctx.fill(); ctx.stroke();

  // Heading arrow
  const hlen = 18;
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(wx(pose.x), wy(pose.y));
  ctx.lineTo(wx(pose.x + hlen * Math.cos(pose.theta)),
             wy(pose.y + hlen * Math.sin(pose.theta)));
  ctx.stroke();
}

function rotPt(bx, by, theta) {
  return [bx * Math.cos(theta) - by * Math.sin(theta),
          bx * Math.sin(theta) + by * Math.cos(theta)];
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `${r},${g},${b}`;
}

// --- Init ------------------------------------------------------------------
window.loadPolicyIntoSim = async function loadPolicyIntoSim() {
  const jobId = window._lastJobId;
  if (!jobId) { pysaac.log('No completed job to load', 'err'); return; }
  const jwt = getJWT();
  const r = await fetch('/sim/policy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + jwt },
    body: JSON.stringify({ job_id: jobId }),
  });
  if (r.ok) { pysaac.log('Policy loaded into sim ✓', 'ok'); fetchWalls(); }
  else       { pysaac.log('Load policy failed: ' + r.status, 'err'); }
};

// Kick off WS connection on page load.
window.addEventListener('load', () => {
  // Give the user a chance to paste a JWT; auto-connect if it's already there.
  if (getJWT()) {
    connectSimWS();
  } else {
    // Prompt for credentials and fetch token
    const username = prompt('Enter username:');
    if (username) {
      const password = prompt('Enter password:');
      
      fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username, password })
      })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(data => {
        sessionStorage.setItem('pysaac_jwt', data.access_token);
        connectSimWS();
      })
      .catch(e => {
        pysaac.log('Login failed: ' + e.message, 'err');
        alert('Login failed. Please refresh the page to try again.');
      });
    }
  }
});

})();
