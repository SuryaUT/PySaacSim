/**
 * Training tab — submits /jobs/train, subscribes to the job WS, charts
 * the reward curve with Chart.js, handles cancel + policy load.
 */
'use strict';

(function () {

// Chart instance
let chart = null;
let jobWs = null;
window._lastJobId = null;

function getJWT() { return sessionStorage.getItem('pysaac_jwt') || ''; }

// --- Chart setup -----------------------------------------------------------
function initChart() {
  if (chart) return;
  const ctx = document.getElementById('reward-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'mean reward',
        data: [],
        borderColor: '#4a9eff',
        backgroundColor: 'rgba(74,158,255,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#777', font: { size: 10 } },
          grid:  { color: '#222' },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function pushRewardPoint(step, reward) {
  if (!chart) initChart();
  chart.data.labels.push(step);
  chart.data.datasets[0].data.push(reward);
  // Keep last 200 points to avoid chart blowup.
  if (chart.data.labels.length > 200) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

function resetChart() {
  if (chart) {
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update('none');
  }
}

// --- Stats panel -----------------------------------------------------------
function setStats({ step, reward, fps, state }) {
  if (step   != null) document.getElementById('ts-step').textContent   = step.toLocaleString();
  if (reward != null) document.getElementById('ts-reward').textContent = reward.toFixed(3);
  if (fps    != null) document.getElementById('ts-fps').textContent    = fps.toFixed(0);
  if (state  != null) document.getElementById('ts-state').textContent  = state;
}

// --- Job WS ---------------------------------------------------------------
function openJobWS(jobId) {
  if (jobWs) { try { jobWs.close(); } catch (_) {} }
  const jwt = getJWT();
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/jobs/${jobId}/events?token=${encodeURIComponent(jwt)}`;
  jobWs = new WebSocket(url);

  jobWs.onopen  = () => pysaac.log(`Job WS ${jobId.slice(0,8)}… connected`, 'ok');
  jobWs.onclose = (ev) => pysaac.log(`Job WS closed (${ev.code})`);
  jobWs.onmessage = (e) => {
    try { handleJobMsg(jobId, JSON.parse(e.data)); }
    catch (_) {}
  };
}

function handleJobMsg(jobId, msg) {
  switch (msg.kind) {
    case 'progress':
      pushRewardPoint(msg.step, msg.mean_reward);
      setStats({ step: msg.step, reward: msg.mean_reward, fps: msg.fps });
      break;

    case 'state':
      setStats({ state: msg.state });
      pysaac.log(`Job state → ${msg.state}`, msg.state === 'failed' ? 'err' : 'info');
      if (msg.state !== 'running') setJobInFlight(false);
      break;

    case 'done': {
      setStats({ state: 'done' });
      setJobInFlight(false);
      window._lastJobId = jobId;
      document.getElementById('btn-load-policy').disabled = false;
      const evalCard = document.getElementById('eval-card');
      evalCard.classList.add('visible');
      document.getElementById('eval-pre').textContent = JSON.stringify(msg.eval, null, 2);
      pysaac.log('Training done! Eval: ' + JSON.stringify(msg.eval), 'ok');
      break;
    }

    case 'error':
      pysaac.log(`Job error: ${msg.code}: ${msg.message}`, 'err');
      setStats({ state: 'failed' });
      setJobInFlight(false);
      break;

    default: break;
  }
}

// --- UI helpers ------------------------------------------------------------
function setJobInFlight(inFlight) {
  document.getElementById('btn-start').disabled = inFlight;
  document.getElementById('btn-stop').disabled  = !inFlight;
}

// --- Public API (called from index.html onclick) ---------------------------
window.startJob = async function startJob() {
  const trackId = document.getElementById('f-track-id').value.trim();
  if (!trackId) { pysaac.log('Enter a Track ID first', 'err'); return; }

  const minutes = parseInt(document.getElementById('f-minutes').value, 10);
  const nEnvs   = parseInt(document.getElementById('f-n-envs').value, 10);
  const lr      = parseFloat(document.getElementById('f-lr').value);
  const jwt     = getJWT();

  if (!jwt) { pysaac.log('No JWT — reload page and enter token', 'err'); return; }

  resetChart();
  document.getElementById('eval-card').classList.remove('visible');
  setStats({ step: 0, reward: 0, fps: 0, state: 'submitting…' });
  setJobInFlight(true);
  pysaac.log(`Submitting job: track=${trackId}, minutes=${minutes}, n_envs=${nEnvs}`);

  try {
    const r = await fetch('/jobs/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + jwt },
      body: JSON.stringify({ track_id: trackId, minutes, n_envs: nEnvs, learning_rate: lr }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      pysaac.log('Submit failed: ' + (err.detail || r.status), 'err');
      setJobInFlight(false);
      return;
    }
    const job = await r.json();
    window._lastJobId = job.job_id;
    setStats({ state: job.state });
    pysaac.log(`Job queued: ${job.job_id}`, 'info');
    openJobWS(job.job_id);
  } catch (e) {
    pysaac.log('Network error: ' + e, 'err');
    setJobInFlight(false);
  }
};

window.cancelJob = async function cancelJob() {
  const jobId = window._lastJobId;
  if (!jobId) return;
  const jwt = getJWT();
  try {
    const r = await fetch(`/jobs/${jobId}`, {
      method: 'DELETE', headers: { Authorization: 'Bearer ' + jwt },
    });
    const body = await r.json().catch(() => ({}));
    pysaac.log(`Cancel → ${body.state || r.status}`, 'info');
  } catch (e) {
    pysaac.log('Cancel error: ' + e, 'err');
  }
};

// Init chart eagerly so it renders in the correct panel size.
window.addEventListener('load', () => setTimeout(initChart, 100));

})();
