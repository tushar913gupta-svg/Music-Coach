const back = document.getElementById('backing');
const guide = document.getElementById('guide');
const useguide = document.getElementById('use-vocals');
const backVol = document.getElementById('backing-vol');
const guideVol = document.getElementById('guide-vol');
const loopLast = document.getElementById('loop-last');

const recbtn = document.getElementById('btn-record');
const stopbtn = document.getElementById('btn-stop');
const playbtn = document.getElementById('btn-play-rec');
const discardbtn = document.getElementById('btn-discard');
const scorebtn = document.getElementById('btn-score');

const stat = document.getElementById('rec-status');
const take = document.getElementById('take-audio');
const canvas = document.getElementById('pitch-canvas');
const live = document.getElementById('live-stats');
const liveNote = document.getElementById('live-note');
const prog = document.getElementById('score-progress');
const meterFill = document.getElementById('mic-meter-fill');

const gslider = document.getElementById('mic-gain');
const gval = document.getElementById('mic-gain-val');

const ctx = canvas.getContext('2d');
const wrap = canvas.closest('.canvas-wrap');

const fmin = 32.7;
const fmax = 1016.6;
const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

let ref = null;
let buf = [];
let len = 0;
let actx = null;
let proc = null;
let stream = null;
let src = null;
let rec = false;
let blob = null;
let takeUrl = null;
let last = null;
let timer = null;


function hzToNote(hz) {
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  const name = noteNames[((midi % 12) + 12) % 12];
  const oct = Math.floor(midi / 12) - 1;
  return `${name}${oct}`;
}


function getWidth(n) {
  const w = wrap ? wrap.clientWidth : canvas.clientWidth;
  const need = n ? Math.round(n * 1.1) : 0;
  return Math.min(8000, Math.max(w || 700, need, 700));
}


function resize() {
  try {
    const d = window.devicePixelRatio || 1;
    const n = ref && ref.hz ? ref.hz.length : 500;
    const tw = getWidth(n);

    canvas.style.width = tw + 'px';
    canvas.width = tw * d;

    let H = 320;

    if (wrap) {
      const avail = wrap.clientHeight - 2;

      if (avail > H) {
        H = Math.floor(avail);
      }
    }

    canvas.style.height = H + 'px';
    canvas.height = H * d;
    ctx.setTransform(d, 0, 0, d, 0, 0);
  } catch (e) {}
}


function hzToY(hz, H) {
  return H - ((Math.log2(hz / fmin) / Math.log2(fmax / fmin)) * H);
}


function draw(p) {
  if (p) {
    last = p;
  }

  const cur = p || last;

  try {
    if (ref && ref.hz) {
      const tw = getWidth(ref.hz.length);

      if (parseInt(canvas.style.width) !== tw) {
        resize();
      }
    }

    const W = parseInt(canvas.style.width) || 700;
    let H = parseInt(canvas.style.height) || 320;

    if (wrap && wrap.clientHeight - 2 > H + 30) {
      resize();
      H = parseInt(canvas.style.height) || 320;
    }

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#13101a';
    ctx.fillRect(0, 0, W, H);

    for (let m = 24; m <= 84; m++) {
      const hz = 440 * Math.pow(2, (m - 69) / 12);

      if (hz < fmin || hz > fmax) {
        continue;
      }

      const y = hzToY(hz, H);
      const isC = m % 12 === 0;

      ctx.strokeStyle = isC ? '#2e2738' : '#241e2c';
      ctx.lineWidth = isC ? 1 : 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(W, y + 0.5);
      ctx.stroke();

      if (isC) {
        ctx.fillStyle = '#a79daf';
        ctx.font = '10px system-ui';
        ctx.fillText(`${noteNames[m % 12]}${Math.floor(m / 12) - 1}`, 6, y - 4);
      }
    }

    if (ref && ref.hop_time) {
      const hop = ref.hop_time || 0.02;
      const n = ref.hz.length;

      ctx.fillStyle = '#a79daf';
      ctx.font = '10px system-ui';

      for (let s = 0; s < n * hop; s += 5) {
        const x = (s / (n * hop)) * W;
        ctx.fillText(`${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`, x + 4, 12);
      }
    }

    if (ref && ref.hz) {
      ctx.strokeStyle = '#e8a13c';
      ctx.lineWidth = 1.6;
      ctx.beginPath();

      let pen = false;
      const n = ref.hz.length;

      for (let i = 0; i < n; i++) {
        const hz = ref.hz[i];
        const c = ref.conf[i];

        if (!hz || c < 0.05) {
          pen = false;
          continue;
        }

        const x = (i / n) * W;
        const y = hzToY(Math.max(fmin, Math.min(fmax, hz)), H);

        if (!pen) {
          ctx.moveTo(x, y);
          pen = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }

    if (cur && cur.hz) {
      ctx.strokeStyle = '#4fc3b6';
      ctx.lineWidth = 1.7;
      ctx.beginPath();

      let pen = false;
      const n = cur.hz.length;
      const rn = ref && ref.hz ? ref.hz.length : n;

      for (let i = 0; i < n; i++) {
        const hz = cur.hz[i];
        const c = cur.conf ? cur.conf[i] : 1;

        if (!hz || c < 0.05) {
          pen = false;
          continue;
        }

        const x = Math.max(0, (i / 2) / rn) * W;
        const y = hzToY(Math.max(fmin, Math.min(fmax, hz)), H);

        if (!pen) {
          ctx.moveTo(x, y);
          pen = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }

    if (back && back.duration && !isNaN(back.duration) && back.duration > 0) {
      const x = (back.currentTime / back.duration) * W;

      ctx.strokeStyle = '#f1ebe2';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
  } catch (e) {}
}


function updateLiveNote(hzList, confList) {
  if (!liveNote || !hzList) {
    return;
  }

  for (let i = hzList.length - 1; i >= 0; i--) {
    if (hzList[i] && (!confList || confList[i] >= 0.05)) {
      liveNote.textContent = hzToNote(hzList[i]);
      return;
    }
  }
}


async function loadRef() {
  try {
    const r = await fetch(`/api/reference_pitch/${SONG_ID}`);

    if (!r.ok) {
      throw new Error();
    }

    const j = await r.json();

    if (j.error) {
      throw new Error(j.error);
    }

    ref = j;

    if (live) {
      live.textContent = 'Ready';
    }

    draw();
  } catch (e) {
    if (live) {
      live.textContent = 'Could not load reference.';
    }
  }
}


async function poll() {
  if (!rec || len < 1600) {
    return;
  }

  const a = new Float32Array(len);
  let o = 0;

  for (const b of buf) {
    a.set(b, o);
    o += b.length;
  }

  try {
    const r = await fetch('/api/pitch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: Array.from(a), sampleRate: 16000 }),
    });

    const j = await r.json();

    if (j.hz) {
      draw({ hz: j.hz, conf: j.conf });
      updateLiveNote(j.hz, j.conf);

      if (live) {
        live.textContent = 'Recording';
      }
    }
  } catch (e) {}
}


function setMeter(peak) {
  if (!meterFill) {
    return;
  }

  const pct = Math.min(100, Math.round(peak * 100));
  meterFill.style.width = pct + '%';
}


async function start() {
  buf = [];
  len = 0;
  playbtn.disabled = true;
  discardbtn.disabled = true;
  scorebtn.disabled = true;

  if (stat) {
    stat.textContent = 'Mic access';
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch (e) {
    if (stat) {
      stat.textContent = 'Mic denied';
    }
    alert('Microphone access denied');
    return;
  }

  actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  src = actx.createMediaStreamSource(stream);

  const gain = actx.createGain();
  gain.gain.value = gslider ? parseFloat(gslider.value) : 2.8;

  const comp = actx.createDynamicsCompressor();
  comp.threshold.value = -28;
  comp.knee.value = 12;
  comp.ratio.value = 3;
  comp.attack.value = 0.005;
  comp.release.value = 0.25;
  window._gain = gain;

  proc = actx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (e) => {
    if (!rec) {
      return;
    }

    const inp = e.inputBuffer.getChannelData(0);
    const ch = new Float32Array(inp.length);
    ch.set(inp);

    buf.push(ch);
    len += ch.length;

    let peak = 0;

    for (let i = 0; i < inp.length; i += 4) {
      peak = Math.max(peak, Math.abs(inp[i]));
    }

    setMeter(peak);

    if (stat) {
      stat.textContent = `Recording ${(len / 16000).toFixed(1)}s`;
    }
  };

  const silent = actx.createGain();
  silent.gain.value = 0;

  src.connect(gain);
  gain.connect(comp);
  comp.connect(proc);
  proc.connect(silent);
  silent.connect(actx.destination);

  rec = true;
  recbtn.disabled = true;
  stopbtn.disabled = false;

  if (stat) {
    stat.textContent = 'Recording';
  }

  if (back.paused) {
    if (loopLast && loopLast.checked && back.duration && !isNaN(back.duration) && back.duration > 10) {
      back.currentTime = back.duration - 10;
    } else {
      back.currentTime = 0;
    }

    if (guide) {
      guide.currentTime = 0;
    }

    back.play().catch(() => {});

    if (useguide && useguide.checked && guide) {
      guide.play().catch(() => {});
    }
  }

  timer = setInterval(poll, 1200);

  const loop = setInterval(() => {
    if (!rec) {
      clearInterval(loop);
    }
    draw();
  }, 100);
}


function stop() {
  rec = false;

  if (timer) {
    clearInterval(timer);
  }
  if (proc) {
    try {
      proc.disconnect();
    } catch (e) {}
  }
  if (src) {
    try {
      src.disconnect();
    } catch (e) {}
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
  }
  if (actx) {
    try {
      actx.close();
    } catch (e) {}
  }
  if (back) {
    back.pause();
  }
  if (guide) {
    guide.pause();
  }

  setMeter(0);

  recbtn.disabled = false;
  stopbtn.disabled = true;

  if (len === 0) {
    if (stat) {
      stat.textContent = 'No audio recorded';
    }
    return;
  }

  const flat = new Float32Array(len);
  let o = 0;

  for (const b of buf) {
    flat.set(b, o);
    o += b.length;
  }

  let peak = 0;

  for (let i = 0; i < flat.length; i++) {
    peak = Math.max(peak, Math.abs(flat[i]));
  }

  if (peak > 0 && peak < 0.85) {
    const g = Math.min(4.0, 0.88 / peak);

    for (let i = 0; i < flat.length; i++) {
      flat[i] *= g;
    }
  }

  const wav = encodeWav(flat, 16000);
  blob = new Blob([wav], { type: 'audio/wav' });

  if (takeUrl) {
    URL.revokeObjectURL(takeUrl);
  }

  takeUrl = URL.createObjectURL(blob);
  take.src = takeUrl;
  take.style.display = 'block';
  take.load();

  playbtn.disabled = false;
  discardbtn.disabled = false;
  scorebtn.disabled = false;

  if (stat) {
    stat.textContent = `Recorded ${(len / 16000).toFixed(1)}s`;
  }

  poll();
}


function discard() {
  buf = [];
  len = 0;
  blob = null;
  last = null;

  if (takeUrl) {
    URL.revokeObjectURL(takeUrl);
    takeUrl = null;
  }

  take.removeAttribute('src');
  take.style.display = 'none';

  playbtn.disabled = true;
  discardbtn.disabled = true;
  scorebtn.disabled = true;

  if (liveNote) {
    liveNote.textContent = '···';
  }

  if (stat) {
    stat.textContent = 'Take discarded';
  }

  setMeter(0);
  draw();
}


function encodeWav(a, sampleRate) {
  const b = new ArrayBuffer(44 + a.length * 2);
  const v = new DataView(b);

  function writeStr(offset, s) {
    for (let i = 0; i < s.length; i++) {
      v.setUint8(offset + i, s.charCodeAt(i));
    }
  }

  writeStr(0, 'RIFF');
  v.setUint32(4, 36 + a.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  writeStr(36, 'data');
  v.setUint32(40, a.length * 2, true);

  let o = 44;

  for (let i = 0; i < a.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, a[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return v;
}


async function score() {
  if (!blob) {
    alert('No recording yet');
    return;
  }

  scorebtn.disabled = true;
  prog.style.display = 'block';

  if (stat) {
    stat.textContent = 'Scoring';
  }

  const fd = new FormData();
  fd.append('audio', blob, 'take.wav');
  fd.append('song_id', SONG_ID);

  try {
    const r = await fetch('/api/score', { method: 'POST', body: fd });
    const j = await r.json();

    if (j.error) {
      throw new Error(j.error);
    }

    window.location.href = `/results/${j.attempt_id}`;
  } catch (e) {
    alert('Scoring failed: ' + e.message);

    if (stat) {
      stat.textContent = 'Scoring failed';
    }

    scorebtn.disabled = false;
    prog.style.display = 'none';
  }
}


async function savePreview() {
  if (len === 0) {
    return;
  }

  const a = new Float32Array(len);
  let o = 0;

  for (const b of buf) {
    a.set(b, o);
    o += b.length;
  }

  const s = a.slice(0, Math.min(a.length, 16000));

  try {
    await fetch('/api/save_pcm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: Array.from(s), sampleRate: 16000 }),
    });
  } catch (e) {}
}


let lyrWords = [];
let lyrSegs = [];
let lyrWI = -2;
let lyrSI = -2;


function loadLyrics() {
  const box = document.getElementById('lyrics');
  if (!box) return;

  fetch('/api/lyrics/' + SONG_ID)
    .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
    .then((j) => {
      if (!j || !j.segments || !j.segments.length) {
        box.innerHTML = j && j.pending
          ? '<p class="text-secondary small mb-0">Transcribing lyrics…</p>'
          : '<p class="text-secondary small mb-0">No lyrics found in this track.</p>';
        return;
      }

      box.innerHTML = '';
      lyrWords = [];
      lyrSegs = [];
      lyrWI = -2;
      lyrSI = -2;

      j.segments.forEach((sg) => {
        const p = document.createElement('p');
        p.className = 'lyr-line';

        (sg.words || []).forEach((wd) => {
          const sp = document.createElement('span');
          sp.className = 'w';
          sp.textContent = wd.w;
          p.appendChild(sp);
          p.appendChild(document.createTextNode(' '));
          lyrWords.push({ el: sp, s: wd.start, e: wd.end });
        });

        box.appendChild(p);
        lyrSegs.push({ el: p, s: sg.start, e: sg.end });
      });
    })
    .catch(() => {
      box.innerHTML = '<p class="text-secondary small mb-0">Lyrics unavailable.</p>';
    });
}


function refVoicedAt(t) {
  if (!ref || !ref.hz || !ref.hop_time) return true;
  const hop = ref.hop_time;
  const lo = Math.floor((t - 0.2) / hop);
  const hi = Math.floor((t + 0.1) / hop);
  const n = ref.hz.length;
  let voiced = 0;
  let total = 0;
  for (let i = Math.max(0, lo); i <= Math.min(n - 1, hi); i++) {
    total++;
    const c = ref.conf ? ref.conf[i] : 1;
    if (ref.hz[i] && c >= 0.05) voiced++;
  }
  if (!total) return false;
  return voiced / total >= 0.4;
}


function updateLyrics(t) {
  if (!lyrWords.length || t === null || t === undefined || isNaN(t)) return;

  const voiced = refVoicedAt(t);

  let wi = -1;
  let si = -1;

  for (let i = 0; i < lyrWords.length; i++) {
    const wd = lyrWords[i];
    if (t < wd.s) break;
    if (voiced && t <= wd.e + 0.15) wi = i;
  }

  for (let k = 0; k < lyrSegs.length; k++) {
    const sg = lyrSegs[k];
    if (t < sg.s) break;
    if (t <= sg.e) si = k;
  }

  if (wi === lyrWI && si === lyrSI) return;

  if (lyrWI >= 0 && lyrWords[lyrWI]) lyrWords[lyrWI].el.classList.remove('on');
  if (lyrSI >= 0 && lyrSegs[lyrSI]) lyrSegs[lyrSI].el.classList.remove('active');

  lyrWI = wi;
  lyrSI = si;

  if (wi >= 0) lyrWords[wi].el.classList.add('on');

  if (si >= 0) {
    lyrSegs[si].el.classList.add('active');
    try { lyrSegs[si].el.scrollIntoView({ block: 'nearest' }); } catch (e) {}
  }
}


resize();
loadRef();
loadLyrics();

window.addEventListener('resize', () => {
  try {
    resize();
    draw();
  } catch (e) {}
});


window.addEventListener('load', () => {
  try {
    resize();
    draw();
  } catch (e) {}
});


if (backVol && back) {
  backVol.addEventListener('input', () => {
    back.volume = parseFloat(backVol.value);
  });
}


if (guideVol && guide) {
  guideVol.addEventListener('input', () => {
    guide.volume = parseFloat(guideVol.value);
  });
}


if (gslider) {
  gslider.addEventListener('input', () => {
    const v = parseFloat(gslider.value);

    if (window._gain) {
      window._gain.gain.value = v;
    }
    if (gval) {
      gval.textContent = v.toFixed(1) + 'x';
    }
  });
}


if (useguide && back && guide) {
  useguide.addEventListener('change', () => {
    if (useguide.checked) {
      guide.volume = guideVol ? parseFloat(guideVol.value) : back.volume * 0.9;

      if (!back.paused) {
        guide.currentTime = back.currentTime;
        guide.play().catch(() => {});
      }
    } else {
      guide.pause();
    }
  });

  back.addEventListener('play', () => {
    if (useguide.checked) {
      guide.currentTime = back.currentTime;
      guide.play().catch(() => {});
    }
  });

  back.addEventListener('pause', () => guide.pause());

  back.addEventListener('seeked', () => {
    if (useguide.checked && !back.paused) {
      guide.currentTime = back.currentTime;
    }
  });

  back.addEventListener('ended', () => {
    if (loopLast && loopLast.checked && back.duration && !isNaN(back.duration) && back.duration > 10) {
      back.currentTime = back.duration - 10;
      back.play().catch(() => {});
    }
  });

  function updateScroll() {
    if (!wrap || !back.duration || isNaN(back.duration)) {
      return;
    }

    const w = parseInt(canvas.style.width) || wrap.clientWidth || 700;
    const x = (back.currentTime / back.duration) * w;

    try {
      wrap.scrollLeft = Math.max(0, x - wrap.clientWidth / 2);
    } catch (e) {}
  }

  back.addEventListener('timeupdate', () => {
    if (useguide.checked && guide && !guide.paused) {
      guide.currentTime = back.currentTime;
    }

    if (loopLast && loopLast.checked && back.duration && !isNaN(back.duration) && back.duration > 10) {
      if (back.duration - back.currentTime < 0.3) {
        back.currentTime = back.duration - 10;
      }
    }

    updateScroll();

    try {
      draw();
      updateLyrics(back.currentTime);
    } catch (e) {}
  });

  (function loop() {
    try {
      updateScroll();

      if (back && !back.paused) {
        draw();
        updateLyrics(back.currentTime);
      }
    } catch (e) {}

    requestAnimationFrame(loop);
  })();
}


if (recbtn) {
  recbtn.addEventListener('click', start);
}

if (stopbtn) {
  stopbtn.addEventListener('click', () => {
    stop();
    savePreview();
  });
}

if (playbtn) {
  playbtn.addEventListener('click', () => {
    if (take.src) {
      take.play();
    }
  });
}

if (discardbtn) {
  discardbtn.addEventListener('click', discard);
}

if (scorebtn) {
  scorebtn.addEventListener('click', score);
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
    e.preventDefault();

    if (rec) {
      stop();
    } else {
      start();
    }
  }
});
