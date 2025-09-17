// Minimal ROI drawer for two canvases (clinical & lab) with two slots each: tooth, shade

const state = {
  clinical: { img: null, rois: { tooth: null, shade: null }, drawing: null, active: 'tooth' },
  lab:      { img: null, rois: { tooth: null, shade: null }, drawing: null, active: 'tooth' }
};

function $(sel) { return document.querySelector(sel); }
function canvas(id){ return document.getElementById(id); }

function loadImageToCanvas(inputId, canvasId, key) {
  const input = document.getElementById(inputId);
  const cnv = canvas(canvasId);
  const ctx = cnv.getContext('2d');

  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      // fit image into canvas (preserve aspect)
      const ratio = Math.min(cnv.width / img.width, cnv.height / img.height);
      const w = img.width * ratio, h = img.height * ratio;
      const x = (cnv.width - w)/2, y = (cnv.height - h)/2;
      state[key].img = { img, x, y, w, h, ratio };
      drawCanvas(key);
    };
    img.src = URL.createObjectURL(file);
  });

  // tool radio
  document.querySelectorAll(`input[name="tool_${key}"]`).forEach(r => {
    r.addEventListener('change', () => {
      state[key].active = r.value;
      drawCanvas(key);
    });
  });

  // draw ROI
  cnv.addEventListener('mousedown', (ev) => {
    if (!state[key].img) return;
    const p = relPoint(cnv, ev);
    if (!insideImg(key, p)) return;
    state[key].drawing = { start: p, current: p };
  });
  cnv.addEventListener('mousemove', (ev) => {
    if (!state[key].img) return;
    const p = relPoint(cnv, ev);
    if (state[key].drawing) {
      state[key].drawing.current = p;
      drawCanvas(key);
    } else {
      drawCanvas(key); // redraw hover crosshair
      drawCursor(cnv, p);
    }
  });
  cnv.addEventListener('mouseup', (ev) => {
    if (!state[key].img || !state[key].drawing) return;
    const p = relPoint(cnv, ev);
    const { start } = state[key].drawing;
    const rect = clampToImage(key, rectFrom(start, p));
    if (rect.w > 4 && rect.h > 4) {
      state[key].rois[state[key].active] = toNormalized(key, rect);
    }
    state[key].drawing = null;
    drawCanvas(key);
  });
}

function relPoint(cnv, ev) {
  const r = cnv.getBoundingClientRect();
  return { x: ev.clientX - r.left, y: ev.clientY - r.top };
}

function rectFrom(a, b) {
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  const w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
  return { x, y, w, h };
}

function insideImg(key, p) {
  const I = state[key].img;
  return p.x >= I.x && p.x <= I.x + I.w && p.y >= I.y && p.y <= I.y + I.h;
}

function clampToImage(key, rect) {
  const I = state[key].img;
  const x = Math.max(I.x, Math.min(rect.x, I.x + I.w));
  const y = Math.max(I.y, Math.min(rect.y, I.y + I.h));
  const xe = Math.max(I.x, Math.min(rect.x + rect.w, I.x + I.w));
  const ye = Math.max(I.y, Math.min(rect.y + rect.h, I.y + I.h));
  return { x, y, w: xe - x, h: ye - y };
}

function toNormalized(key, rect) {
  const I = state[key].img;
  return {
    x: (rect.x - I.x) / I.w,
    y: (rect.y - I.y) / I.h,
    w: rect.w / I.w,
    h: rect.h / I.h
  };
}

function fromNormalized(key, nrect) {
  const I = state[key].img;
  return {
    x: I.x + nrect.x * I.w,
    y: I.y + nrect.y * I.h,
    w: nrect.w * I.w,
    h: nrect.h * I.h
  };
}

function drawCanvas(key) {
  const cnv = canvas(key === 'clinical' ? 'canvas_clinical' : 'canvas_lab');
  const ctx = cnv.getContext('2d');
  ctx.clearRect(0,0,cnv.width, cnv.height);
  ctx.fillStyle = '#0b0b0d';
  ctx.fillRect(0,0,cnv.width, cnv.height);

  if (state[key].img) {
    const I = state[key].img;
    ctx.drawImage(I.img, I.x, I.y, I.w, I.h);

    // Existing ROIs
    ['tooth','shade'].forEach(name => {
      const nrect = state[key].rois[name];
      if (nrect) {
        const r = fromNormalized(key, nrect);
        ctx.lineWidth = 2;
        ctx.strokeStyle = name === 'tooth' ? '#22d3ee' : '#a78bfa';
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        // label
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(r.x, r.y - 16, 54, 16);
        ctx.fillStyle = '#fff';
        ctx.font = '12px sans-serif';
        ctx.fillText(name.toUpperCase(), r.x + 3, r.y - 4);
      }
    });

    // Drawing rect preview
    if (state[key].drawing) {
      const r = clampToImage(key, rectFrom(state[key].drawing.start, state[key].drawing.current));
      ctx.setLineDash([6,4]);
      ctx.strokeStyle = state[key].active === 'tooth' ? '#22d3ee' : '#a78bfa';
      ctx.lineWidth = 2;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.setLineDash([]);
    }
  } else {
    ctx.fillStyle = '#7a7b86';
    ctx.font = '14px sans-serif';
    ctx.fillText('No image loaded', 16, 24);
  }
}

function drawCursor(cnv, p){
  const ctx = cnv.getContext('2d');
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.beginPath();
  ctx.moveTo(p.x-8, p.y); ctx.lineTo(p.x+8, p.y);
  ctx.moveTo(p.x, p.y-8); ctx.lineTo(p.x, p.y+8);
  ctx.stroke();
}

function clearROIs(key){
  state[key].rois = { tooth:null, shade:null };
  drawCanvas(key);
}

function prepareROI(){
  // validate
  const need = ['clinical','lab'];
  for (const k of need){
    if (!state[k].img) { alert(`Please load the ${k} image.`); return false; }
    const r = state[k].rois;
    if (!r.tooth || !r.shade) { alert(`Draw both Tooth and Shade ROIs on the ${k} image.`); return false; }
  }
  const roi = {
    clinical: { tooth: toArray(state.clinical.rois.tooth), shade: toArray(state.clinical.rois.shade) },
    lab:      { tooth: toArray(state.lab.rois.tooth),      shade: toArray(state.lab.rois.shade) }
  };
  document.getElementById('roi_json').value = JSON.stringify(roi);
  return true;
}

function toArray(r){ return [r.x, r.y, r.w, r.h]; }

window.addEventListener('DOMContentLoaded', () => {
  loadImageToCanvas('clinical_input','canvas_clinical','clinical');
  loadImageToCanvas('lab_input','canvas_lab','lab');
});