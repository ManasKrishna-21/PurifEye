// PurifEye Dashboard Frontend — v0.2

document.addEventListener("DOMContentLoaded", () => {

  // ── Tab Navigation ──────────────────────────────────────────
  const navTabs = document.querySelectorAll(".nav-tab");
  const pages   = document.querySelectorAll(".page");

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      navTabs.forEach(t => { t.classList.remove("active"); t.removeAttribute("aria-current"); });
      pages.forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      tab.setAttribute("aria-current", "page");
      const target = document.getElementById(tab.getAttribute("data-page"));
      if (target) target.classList.add("active");
    });
  });

  // ── Telemetry Polling (Live Detect) ─────────────────────────
  // Only polls /api/telemetry; pipeline starts only when /video_feed is loaded.
  const footerStatus = document.getElementById("footer-status");
  const pipeState    = document.getElementById("live-pipeline-state");

  setInterval(async () => {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      const d = await res.json();

      const fps = d.fps ?? 0;
      document.getElementById("live-fps").textContent      = fps.toFixed(1);
      document.getElementById("live-inf").textContent      = (d.inf_ms ?? 0).toFixed(1) + " ms";
      document.getElementById("live-e2e").textContent      = (d.e2e_ms ?? 0).toFixed(1) + " ms";
      document.getElementById("live-active").textContent   = d.active_detections ?? 0;
      document.getElementById("live-counted").textContent  = d.counted_bottles   ?? 0;
      document.getElementById("live-dropped").textContent  = d.dropped_frames    ?? 0;

      if (pipeState) {
        pipeState.textContent = fps > 0 ? "RUNNING" : "STANDBY";
      }
    } catch (_) {
      // quiet fail on disconnect
    }
  }, 500);

  // ── Helper: show image, hide placeholder ─────────────────────
  function showResult(imgEl, placeholderEl, url) {
    imgEl.src = url;
    imgEl.style.display = "block";
    if (placeholderEl) placeholderEl.style.display = "none";
  }

  // ── Image Analyzer ──────────────────────────────────────────
  const analyzeBtn     = document.getElementById("analyze-btn");
  const analyzeImg     = document.getElementById("analyze-result-img");
  const analyzePH      = document.getElementById("analyze-placeholder");

  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      const fileInput = document.getElementById("analyze-file");
      if (!fileInput.files.length) { alert("Please select an image file first."); return; }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      formData.append("mode", document.getElementById("analyze-mode").value);
      formData.append("conf", document.getElementById("analyze-conf").value);

      analyzeBtn.textContent = "Analyzing…";
      analyzeBtn.disabled    = true;
      if (footerStatus) footerStatus.textContent = "Running YOLOv8 inference…";

      try {
        const res  = await fetch("/api/analyze", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) { alert("Server error: " + data.error); return; }

        showResult(analyzeImg, analyzePH, data.image_url + "?t=" + Date.now());
        document.getElementById("analyze-latency").textContent = data.latency_ms.toFixed(1) + " ms";
        document.getElementById("analyze-count").textContent   = data.bottle_count;
        if (footerStatus) footerStatus.textContent = `Analysis complete — ${data.bottle_count} detection(s) in ${data.latency_ms.toFixed(1)} ms`;
      } catch (e) {
        alert("Error analyzing image: " + e);
      } finally {
        analyzeBtn.textContent = "Analyze Image";
        analyzeBtn.disabled    = false;
      }
    });
  }

  // ── Model Lab Compare ───────────────────────────────────────
  const compareBtn    = document.getElementById("compare-btn");
  const labYoloImg    = document.getElementById("lab-yolo-img");
  const labTfliteImg  = document.getElementById("lab-tflite-img");
  const labYoloPH     = document.getElementById("lab-yolo-placeholder");
  const labTflitePH   = document.getElementById("lab-tflite-placeholder");

  if (compareBtn) {
    compareBtn.addEventListener("click", async () => {
      const fileInput = document.getElementById("compare-file");
      if (!fileInput.files.length) { alert("Please select an image to compare."); return; }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      compareBtn.textContent = "Comparing…";
      compareBtn.disabled    = true;
      if (footerStatus) footerStatus.textContent = "Running dual model inference…";

      try {
        const res  = await fetch("/api/compare", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) { alert("Server error: " + data.error); return; }

        const ts = "?t=" + Date.now();
        showResult(labYoloImg,   labYoloPH,   data.yolo.image_url   + ts);
        showResult(labTfliteImg, labTflitePH, data.tflite.image_url + ts);

        document.getElementById("lab-yolo-count").textContent   = data.yolo.count;
        document.getElementById("lab-yolo-lat").textContent     = data.yolo.latency_ms.toFixed(1) + " ms";
        document.getElementById("lab-tflite-count").textContent = data.tflite.count;
        document.getElementById("lab-tflite-lat").textContent   = data.tflite.latency_ms.toFixed(1) + " ms";
        if (footerStatus) footerStatus.textContent = `Comparison done — YOLO: ${data.yolo.latency_ms.toFixed(0)} ms · TFLite: ${data.tflite.latency_ms.toFixed(0)} ms`;
      } catch (e) {
        alert("Error in comparison: " + e);
      } finally {
        compareBtn.textContent = "Run Dual Comparison";
        compareBtn.disabled    = false;
      }
    });
  }
});
