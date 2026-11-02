import streamlit as st
import streamlit.components.v1 as components
import textwrap

st.set_page_config(
    page_title="파동 반사: 물리 시뮬레이터",
    page_icon="〰️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px; }
      iframe { border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("〰️ 파동 반사: 물리 시뮬레이터")
st.write(
    "고정단과 자유단에서 파동이 어떻게 반사되는지 관찰해 보세요. "
    "움직임은 브라우저 안에서 직접 그리므로, 매 프레임마다 Streamlit이나 그래프를 다시 불러오지 않습니다."
)

html = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light; --ink:#202833; --muted:#617080; --line:#d9e0e7; --panel:#f6f8fa; }
  * { box-sizing:border-box; }
  html, body { margin:0; padding:0; background:transparent; color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { width:100%; padding:2px 2px 8px; }
  .controls { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:0 0 10px; }
  .control { border:1px solid var(--line); border-radius:12px; background:#fff; padding:10px 12px; min-width:0; }
  label, .control-title { display:block; font-size:13px; color:var(--muted); margin-bottom:7px; font-weight:600; }
  select, input[type=range] { width:100%; }
  select { border:1px solid #cbd4de; border-radius:8px; background:#fff; min-height:36px; padding:4px 7px; color:var(--ink); font-size:14px; }
  input[type=range] { accent-color:#2587cc; cursor:pointer; }
  .number { font-size:13px; font-weight:700; color:var(--ink); float:right; }
  .buttons { display:flex; flex-wrap:wrap; gap:7px; }
  button { appearance:none; border:1px solid #cbd4de; background:#fff; color:#243343; border-radius:9px; padding:9px 12px; min-height:38px; font-size:14px; font-weight:650; cursor:pointer; }
  button:hover { background:#f0f5f9; }
  button.primary { color:#fff; background:#2379b8; border-color:#2379b8; }
  button.primary:hover { background:#1b679e; }
  button.selected { box-shadow:inset 0 0 0 2px #2587cc; border-color:#2587cc; }
  .speed-row { display:flex; gap:6px; flex-wrap:wrap; }
  .speed-row button { flex:1; padding:7px 6px; min-width:62px; }
  .chart-box { border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fff; }
  canvas { display:block; width:100%; height:390px; }
  .timebar { padding:10px 14px 12px; border-top:1px solid var(--line); }
  .time-label { display:flex; justify-content:space-between; align-items:center; font-size:13px; color:var(--muted); gap:10px; }
  .readouts { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:10px 0; }
  .metric { border:1px solid var(--line); border-radius:10px; padding:10px 12px; min-width:0; }
  .metric .name { color:var(--muted); font-size:12px; margin-bottom:5px; }
  .metric .value { font-size:18px; font-weight:750; overflow-wrap:anywhere; }
  .note { border-radius:10px; background:#f3f7fb; color:#3a4c5e; padding:10px 12px; font-size:13px; line-height:1.55; }
  .model { margin-top:12px; padding:12px 14px; border:1px solid var(--line); border-radius:12px; font-size:13px; line-height:1.65; color:#465564; background:#fff; }
  .model strong { color:#263746; }
  @media (max-width:700px) {
    .controls { grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    canvas { height:320px; }
    .readouts { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .metric .value { font-size:16px; }
    button { padding:8px 10px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="controls">
    <div class="control">
      <label for="boundary">줄 끝 조건</label>
      <select id="boundary"><option value="fixed">고정단</option><option value="free">자유단</option></select>
    </div>
    <div class="control">
      <label for="tension">장력 <span class="number" id="tensionValue">40 N</span></label>
      <input id="tension" type="range" min="10" max="80" value="40" step="1">
    </div>
    <div class="control">
      <label for="density">선밀도 <span class="number" id="densityValue">25 g/m</span></label>
      <input id="density" type="range" min="10" max="80" value="25" step="1">
    </div>
    <div class="control">
      <div class="control-title">재생 속도</div>
      <div class="speed-row">
        <button type="button" data-speed="0.25">0.25×</button>
        <button type="button" data-speed="0.5">0.5×</button>
        <button type="button" data-speed="1" class="selected">1×</button>
        <button type="button" data-speed="2">2×</button>
      </div>
    </div>
  </div>

  <div class="buttons" style="margin:0 0 10px;">
    <button id="play" class="primary" type="button">▶ 재생</button>
    <button id="pause" type="button">Ⅱ 일시 정지</button>
    <button id="reset" type="button">↺ 초기화</button>
    <span id="status" style="align-self:center;color:var(--muted);font-size:13px;margin-left:3px;">일시 정지</span>
  </div>

  <div class="chart-box">
    <canvas id="waveCanvas" aria-label="입사파, 반사파, 합성파 그래프"></canvas>
    <div class="timebar">
      <div class="time-label"><label for="timeSlider" style="margin:0;">시간</label><span id="timeValue">0.0 ms</span></div>
      <input id="timeSlider" type="range" min="0" max="100" value="0" step="0.1" aria-label="시뮬레이션 시간">
    </div>
  </div>

  <div class="readouts">
    <div class="metric"><div class="name">이상적인 파동 속력</div><div class="value" id="waveSpeed">40.00 m/s</div></div>
    <div class="metric"><div class="name">입사파 중심 위치</div><div class="value" id="incidentPosition">0.50 m</div></div>
    <div class="metric"><div class="name">반사파 중심 위치*</div><div class="value" id="reflectedPosition">3.50 m</div></div>
    <div class="metric"><div class="name">끝점 변위</div><div class="value" id="endpointDisplacement">0.000 cm</div></div>
  </div>

  <div class="note" id="physicsNote">고정단에서는 끝점의 변위가 0이 되도록 반사파의 부호가 바뀝니다. (반사 계수 r = −1)</div>
  <div class="model">
    <strong>사용한 물리 모델</strong><br>
    줄의 길이 L = 2 m, 초기 입사 펄스 중심 x₀ = 0.5 m, 진폭 A = 0.03 m, 폭 매개변수 σ = 0.13 m입니다.<br>
    파동 속력은 <strong>v = √(T/μ)</strong>로 계산하며, 화면의 선밀도 g/m를 kg/m로 변환합니다.<br>
    입사파와 끝점에서 반사된 파동을 중첩한 이상적 모델입니다. 감쇠, 비선형 효과, 왼쪽 끝에서의 추가 반사는 고려하지 않습니다.<br>
    * 반사파 중심 위치는 줄 바깥까지 수학적으로 연장한 값일 수 있습니다.
  </div>
</div>

<script>
(() => {
  'use strict';
  const L = 2.0, x0 = 0.5, A = 0.03, sigma = 0.13;
  const canvas = document.getElementById('waveCanvas');
  const ctx = canvas.getContext('2d', { alpha: false });
  const boundaryInput = document.getElementById('boundary');
  const tensionInput = document.getElementById('tension');
  const densityInput = document.getElementById('density');
  const timeSlider = document.getElementById('timeSlider');
  const playBtn = document.getElementById('play');
  const pauseBtn = document.getElementById('pause');
  const resetBtn = document.getElementById('reset');
  const status = document.getElementById('status');
  const speedButtons = [...document.querySelectorAll('[data-speed]')];
  let playing = false, speedFactor = 1, timeMs = 0, lastStamp = null;
  let waveSpeed = 40, maxTimeMs = 97.5, dirty = true;
  let cssWidth = 900, cssHeight = 390, dpr = 1;

  const fmt = (v, n=2) => Number(v).toFixed(n);
  function recalc() {
    const tension = Number(tensionInput.value);
    const densityGm = Number(densityInput.value);
    const mu = densityGm / 1000;
    waveSpeed = Math.sqrt(tension / mu);
    maxTimeMs = Math.max(20, Math.min(350, ((2 * L - x0) / waveSpeed) * 1000 + 10));
    timeMs = Math.min(timeMs, maxTimeMs);
    document.getElementById('tensionValue').textContent = `${tension} N`;
    document.getElementById('densityValue').textContent = `${densityGm} g/m`;
    document.getElementById('waveSpeed').textContent = `${fmt(waveSpeed)} m/s`;
    timeSlider.max = String(maxTimeMs);
    timeSlider.value = String(timeMs);
    dirty = true;
  }

  function resizeCanvas() {
    const box = canvas.getBoundingClientRect();
    cssWidth = Math.max(300, box.width);
    cssHeight = Math.max(280, box.height);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    dirty = true;
  }

  function draw() {
    const W = cssWidth, H = cssHeight;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
    const left = 48, right = 18, top = 26, bottom = 42;
    const pw = W - left - right, ph = H - top - bottom;
    const yMin = -0.065, yMax = 0.065;
    const xPx = x => left + (x / L) * pw;
    const yPx = y => top + ((yMax - y) / (yMax - yMin)) * ph;
    const t = timeMs / 1000;
    const r = boundaryInput.value === 'fixed' ? -1 : 1;
    const inc = x => A * Math.exp(-Math.pow(x - x0 - waveSpeed*t, 2) / (2*sigma*sigma));
    const ref = x => r * A * Math.exp(-Math.pow(2*L - x - x0 - waveSpeed*t, 2) / (2*sigma*sigma));

    ctx.lineWidth = 1;
    ctx.strokeStyle = '#e6ebf0';
    ctx.fillStyle = '#617080'; ctx.font = '12px system-ui, sans-serif';
    for (let xv=0; xv<=L+1e-6; xv+=0.5) {
      const xx = xPx(xv); ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, H-bottom); ctx.stroke();
      ctx.textAlign = 'center'; ctx.fillText(xv.toFixed(1), xx, H-19);
    }
    for (let yv=-0.06; yv<=0.0601; yv+=0.03) {
      const yy = yPx(yv); ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(W-right, yy); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText((yv*100).toFixed(0), left-9, yy+4);
    }
    ctx.fillStyle = '#4e5d6b'; ctx.textAlign = 'center'; ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('줄의 위치 (m)', left + pw/2, H-3);
    ctx.save(); ctx.translate(13, top+ph/2); ctx.rotate(-Math.PI/2); ctx.fillText('변위 (cm)', 0, 0); ctx.restore();
    // Endpoint marker: fixed end is a solid wall; free end is shown with a dashed marker.
    ctx.beginPath(); ctx.moveTo(xPx(L), top); ctx.lineTo(xPx(L), H-bottom);
    ctx.strokeStyle = '#d98225'; ctx.lineWidth = boundaryInput.value === 'fixed' ? 4 : 2;
    ctx.setLineDash(boundaryInput.value === 'fixed' ? [] : [6,5]); ctx.stroke(); ctx.setLineDash([]);
    function drawLine(fn, color, width, dash=[]) {
      ctx.beginPath();
      const count = Math.max(220, Math.min(420, Math.round(pw * 0.48)));
      for (let i=0; i<count; i++) {
        const x = (i/(count-1))*L;
        const y = fn(x)*100; // convert metres to centimetres
        if (i===0) ctx.moveTo(xPx(x), yPx(y/100)); else ctx.lineTo(xPx(x), yPx(y/100));
      }
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
    }
    drawLine(x => inc(x)+ref(x), '#2587cc', 3);
    drawLine(inc, '#149688', 1.8);
    drawLine(ref, '#d98225', 1.8, [6,4]);
    // Compact legend.
    const items = [ ['합성파','#2587cc',[]], ['입사파','#149688',[]], ['반사파','#d98225',[5,4]] ];
    ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'left';
    let lx = left, ly = 15;
    for (const [label,color,dash] of items) {
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx+18, ly); ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle='#3f4d5a'; ctx.fillText(label,lx+23,ly+4); lx += 23 + ctx.measureText(label).width + 14;
    }
    // Readouts. End-point displacement is evaluated from the two overlapping pulses.
    const incidentCenter = x0 + waveSpeed*t;
    const reflectedCenter = 2*L - x0 - waveSpeed*t;
    const endValue = (1+r)*A*Math.exp(-Math.pow(L-x0-waveSpeed*t,2)/(2*sigma*sigma));
    document.getElementById('incidentPosition').textContent = `${fmt(incidentCenter)} m`;
    document.getElementById('reflectedPosition').textContent = `${fmt(reflectedCenter)} m`;
    document.getElementById('endpointDisplacement').textContent = `${fmt(endValue*100,3)} cm`;
    document.getElementById('timeValue').textContent = `${fmt(timeMs,1)} ms`;
    document.getElementById('physicsNote').textContent = r < 0
      ? '고정단: 끝점의 변위가 0이 되도록 반사파의 부호가 바뀝니다. (반사 계수 r = −1)'
      : '자유단: 끝점의 기울기가 0이 되도록 반사파가 같은 부호로 진행합니다. (반사 계수 r = +1)';
    dirty = false;
  }

  function setPlaying(value) {
    playing = value; lastStamp = null;
    status.textContent = playing ? `재생 중 · ${speedFactor}×` : '일시 정지';
    playBtn.disabled = playing; pauseBtn.disabled = !playing;
    playBtn.style.opacity = playing ? '.55' : '1'; pauseBtn.style.opacity = playing ? '1' : '.55';
  }
  function tick(stamp) {
    if (playing) {
      if (lastStamp !== null) {
        // Limit huge time jumps after tab switching, preventing sudden skipping.
        const delta = Math.min((stamp-lastStamp)/1000, 0.05);
        timeMs += delta*1000*speedFactor;
        if (timeMs >= maxTimeMs) timeMs = 0;
        timeSlider.value = String(timeMs);
        dirty = true;
      }
      lastStamp = stamp;
    } else {
      lastStamp = null;
    }
    if (dirty) draw();
    requestAnimationFrame(tick);
  }

  boundaryInput.addEventListener('change', () => { dirty = true; });
  tensionInput.addEventListener('input', recalc);
  densityInput.addEventListener('input', recalc);
  timeSlider.addEventListener('input', () => {
    timeMs = Number(timeSlider.value); dirty = true;
    // Keep playback continuous after scrubbing.
    lastStamp = null;
  });
  playBtn.addEventListener('click', () => setPlaying(true));
  pauseBtn.addEventListener('click', () => setPlaying(false));
  resetBtn.addEventListener('click', () => {
    setPlaying(false); timeMs = 0; timeSlider.value = '0'; dirty = true;
  });
  speedButtons.forEach(btn => btn.addEventListener('click', () => {
    speedFactor = Number(btn.dataset.speed);
    speedButtons.forEach(b => b.classList.toggle('selected', b===btn));
    status.textContent = playing ? `재생 중 · ${speedFactor}×` : '일시 정지';
  }));
  if ('ResizeObserver' in window) new ResizeObserver(resizeCanvas).observe(canvas);
  else window.addEventListener('resize', resizeCanvas);
  recalc(); resizeCanvas(); setPlaying(false); requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""

components.html(textwrap.dedent(html), height=850, scrolling=False)

st.caption(
    "부드러운 재생을 위해 그래프를 브라우저 Canvas로 직접 그리며, "
    "각 프레임마다 Streamlit 전체를 다시 실행하거나 외부 그래프 라이브러리를 호출하지 않습니다."
)
