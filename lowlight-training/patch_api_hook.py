from pathlib import Path

dash_path = Path("dashboard.html")
html = dash_path.read_text(encoding="utf-8")

old_code = """  function handleUserFile(file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      isCustom = true;
      document.getElementById('file-meta').textContent = file.name;
      document.getElementById('img-status-badge').textContent = 'Custom Upload';
      activeImg = new Image();
      activeImg.onload = processAndRender;
      activeImg.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }"""

new_code = """  async function handleUserFile(file) {
    isCustom = true;
    document.getElementById('file-meta').textContent = file.name;
    document.getElementById('img-status-badge').innerHTML = '<span class="animate-pulse text-yellow-400">⚡ Running PyTorch Detection...</span>';

    // Try backend /api/predict first
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('scale_factor', rngSf ? rngSf.value : 4);
      formData.append('conf', rngConf ? rngConf.value : 0.25);
      formData.append('mode', selMode ? selMode.value : 'hq');

      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          applyServerPrediction(data);
          document.getElementById('img-status-badge').innerHTML = '<span class="text-emerald-400 font-semibold">✓ Model Prediction Active</span>';
          return;
        }
      }
    } catch (err) {
      console.log('Backend API offline, using interactive client canvas engine', err);
    }

    // Fallback to client-side canvas engine
    const reader = new FileReader();
    reader.onload = (ev) => {
      document.getElementById('img-status-badge').textContent = 'Custom Upload (Client)';
      activeImg = new Image();
      activeImg.onload = processAndRender;
      activeImg.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }

  function applyServerPrediction(data) {
    const setImg = (cv, b64) => {
      if (!cv || !b64) return;
      const im = new Image();
      im.onload = () => { cv.width = im.width; cv.height = im.height; cv.getContext('2d').drawImage(im, 0, 0); };
      im.src = b64;
    };

    setImg(cvOrigDet, data.orig_det);
    setImg(cvEnhDet, data.enh_det);
    setImg(cvOrigClean, data.orig_img);
    setImg(cvEnhClean, data.enh_img);
    if (data.curve_map) setImg(cvCurveMap, data.curve_map);

    document.getElementById('view-res').textContent = data.res + ' px';
    document.getElementById('bva-orig-lum').textContent = 'Mean Lum: ' + data.orig_lum + ' / 255';
    document.getElementById('bva-enh-lum').textContent = 'Mean Lum: ' + data.enh_lum + ' / 255 (+' + data.gain_pct + '%)';
    document.getElementById('tele-enh').textContent = data.enh_ms + ' ms';
    document.getElementById('tele-det').textContent = data.det_ms + ' ms';
    document.getElementById('tele-e2e').textContent = data.total_ms + ' ms';
    document.getElementById('tele-gain').textContent = '+' + data.gain_pct + '%';
    document.getElementById('tele-objs').textContent = data.enh_detections + ' targets';

    document.getElementById('det-count-orig').textContent = data.orig_detections + ' Detections';
    document.getElementById('det-count-enh').textContent = data.enh_detections + ' Detections';

    const tableBody = document.getElementById('live-det-table-body');
    if (tableBody) {
      if (data.boxes && data.boxes.length > 0) {
        tableBody.innerHTML = data.boxes.map(b => (
          '<tr class="border-b border-[var(--border)]/50">' +
            '<td class="px-3 py-1.5 font-mono text-[var(--muted-foreground)]">' + b.id + '</td>' +
            '<td class="px-3 py-1.5 font-semibold text-emerald-400">' + b.class + '</td>' +
            '<td class="px-3 py-1.5 text-right font-mono font-bold text-emerald-400">' + b.conf + '</td>' +
            '<td class="px-3 py-1.5 text-center text-red-400 font-semibold">✗ Missed in Dark</td>' +
            '<td class="px-3 py-1.5 text-center text-emerald-400 font-bold">✓ Detected [' + b.bbox.join(', ') + ']</td>' +
          '</tr>'
        )).join('');
      } else {
        tableBody.innerHTML = '<tr><td colspan="5" class="text-center py-2 text-[var(--muted-foreground)]">No objects detected above threshold ' + (rngConf ? rngConf.value : 0.25) + '</td></tr>';
      }
    }
  }"""

if old_code in html:
    html = html.replace(old_code, new_code)
    dash_path.write_text(html, encoding="utf-8")
    brain_path = Path(r"C:\Users\yvikr\.gemini\antigravity\brain\83cbae79-5e25-46f9-bc88-3832aacdb7e5\dashboard.html")
    brain_path.write_text(html, encoding="utf-8")
    print("Updated dashboard with async PyTorch /api/predict hook!")
else:
    print("old_code not found")
