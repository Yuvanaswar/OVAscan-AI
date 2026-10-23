/**
 * OvaScan AI — Clinical Ultrasound PACS Workstation Client Logic
 * Top-2 Dual Architecture & Authentic Real Clinical Scans (v6.0)
 */

document.addEventListener("DOMContentLoaded", () => {

    // ==================== STATE ====================
    let currentImageFile = null;
    let currentExampleName = null;
    let activeModelKey = "convnext_v6_3class";
    let currentExplanations = null;
    let currentActiveMode = "standard";
    const sessionHistory = [];

    // ==================== DOM ELEMENTS ====================
    // Top Bar & Model Switcher
    const modelBtns = document.querySelectorAll(".model-switch-btn");
    const btnNewScan = document.getElementById("btnNewScan");
    const btnQuickDemos = document.getElementById("btnQuickDemos");
    const quickDemosMenu = document.getElementById("quickDemosMenu");
    const demoItems = document.querySelectorAll(".demo-select-item");
    const btnHistory = document.getElementById("btnHistory");
    const btnCriteria = document.getElementById("btnCriteria");

    // Ingestion & File Input
    const fileInput = document.getElementById("fileInput");
    const heroDropzone = document.getElementById("heroDropzone");
    const samplePillBtns = document.querySelectorAll(".sample-pill-btn");
    const btnChangeActiveScan = document.getElementById("btnChangeActiveScan");
    const btnRetryGate = document.getElementById("btnRetryGate");

    // Viewport States
    const resultStateEmpty = document.getElementById("resultStateEmpty");
    const resultStateAnalyzing = document.getElementById("resultStateAnalyzing");
    const resultStateActive = document.getElementById("resultStateActive");
    const resultStateUnsupported = document.getElementById("resultStateUnsupported");
    const unsupportedMessageText = document.getElementById("unsupportedMessageText");
    const analyzingStepText = document.getElementById("analyzingStepText");
    const scanProgressFill = document.getElementById("scanProgressFill");

    // Viewport Active Elements
    const modeTabBtns = document.querySelectorAll("#explanationModeBar .mode-tab-btn");
    const gradcamImg = document.getElementById("gradcamImg");
    const gradcamAnnotations = document.getElementById("gradcamAnnotations");
    const annoRegion = document.getElementById("annoRegion");
    const annoSalience = document.getElementById("annoSalience");
    const colorbarContainer = document.getElementById("colorbarContainer");
    const btnDownloadGradcam = document.getElementById("btnDownloadGradcam");
    const hudAttentionText = document.getElementById("hudAttentionText");
    const hudModelTag = document.getElementById("hudModelTag");
    const hudScanId = document.getElementById("hudScanId");

    // Follicle Table
    const follicleInventoryCard = document.getElementById("follicleInventoryCard");
    const follicleInventoryCount = document.getElementById("follicleInventoryCount");
    const follicleTableBody = document.getElementById("follicleTableBody");

    // Diagnostic Verdict Card (Right Column)
    const predictionBanner = document.getElementById("predictionBanner");
    const verdictHeadline = document.getElementById("verdictHeadline");
    const verdictDescription = document.getElementById("verdictDescription");
    const verdictPill = document.getElementById("verdictPill");
    const confidencePill = document.getElementById("confidencePill");

    // 3-Class Probability Bars
    const probRowDf = document.getElementById("probRowDf");
    const probValDf = document.getElementById("probValDf");
    const probBarDf = document.getElementById("probBarDf");

    const probRowPcos = document.getElementById("probRowPcos");
    const probValPcos = document.getElementById("probValPcos");
    const probBarPcos = document.getElementById("probBarPcos");

    const probRowNormal = document.getElementById("probRowNormal");
    const probValNormal = document.getElementById("probValNormal");
    const probBarNormal = document.getElementById("probBarNormal");

    // Quantitative Biomarkers
    const metricAfc = document.getElementById("metricAfc");
    const metricRotterdamBadge = document.getElementById("metricRotterdamBadge");
    const metricDominantCount = document.getElementById("metricDominantCount");
    const metricDominantBadge = document.getElementById("metricDominantBadge");
    const metricMeanDiameter = document.getElementById("metricMeanDiameter");
    const metricDiameterRange = document.getElementById("metricDiameterRange");
    const metricPattern = document.getElementById("metricPattern");
    const metricAiConcordance = document.getElementById("metricAiConcordance");

    // Operating Specs
    const specModel = document.getElementById("specModel");
    const specBenchmark = document.getElementById("specBenchmark");
    const specThreshold = document.getElementById("specThreshold");
    const specDate = document.getElementById("specDate");

    // Modals
    const modalHistory = document.getElementById("modalHistory");
    const modalHowItWorks = document.getElementById("modalHowItWorks");
    const btnCloseHistory = document.getElementById("btnCloseHistory");
    const btnCloseHowItWorks = document.getElementById("btnCloseHowItWorks");
    const historyListContainer = document.getElementById("historyListContainer");

    const MODE_CONFIG = {
        standard: { showColorbar: true, showAnno: true },
        roi_focus: { showColorbar: false, showAnno: false },
        radiology_hotspot: { showColorbar: true, showAnno: false },
        tri_panel: { showColorbar: false, showAnno: false },
        edge_guided: { showColorbar: true, showAnno: false }
    };

    // ==================== INITIALIZATION ====================
    resetWorkstation();
    setupIngestionHandlers();
    setupModeSwitcher();
    setupTop2ModelToggle();
    setupModals();
    setupQuickDemosDropdown();

    // ==================== RESET WORKSTATION ====================
    function resetWorkstation() {
        resultStateEmpty.classList.remove("hidden");
        resultStateAnalyzing.classList.add("hidden");
        resultStateActive.classList.add("hidden");
        resultStateUnsupported.classList.add("hidden");
        follicleInventoryCard.classList.add("hidden");

        predictionBanner.className = "diagnostic-verdict-card";
        verdictHeadline.textContent = "Awaiting Diagnostic Input";
        verdictDescription.textContent = "Upload an ovarian ultrasound scan or select a clinical test sample to initiate dual-model analysis & follicle bounding.";
        verdictPill.textContent = "Awaiting Scan";
        confidencePill.textContent = "--";

        probValDf.textContent = "0.0%"; probBarDf.style.width = "0%";
        probValPcos.textContent = "0.0%"; probBarPcos.style.width = "0%";
        probValNormal.textContent = "0.0%"; probBarNormal.style.width = "0%";

        metricAfc.textContent = "--";
        metricRotterdamBadge.textContent = "--"; metricRotterdamBadge.className = "bm-badge";
        metricDominantCount.textContent = "--";
        metricDominantBadge.textContent = "--"; metricDominantBadge.className = "bm-badge";
        metricMeanDiameter.textContent = "-- mm";
        metricDiameterRange.textContent = "Range: -- mm";
        metricPattern.textContent = "--";
        metricAiConcordance.textContent = "AI Concordance: --%";

        hudAttentionText.innerHTML = "Status: <strong>Ready for Scan Ingestion</strong>";
        hudScanId.innerHTML = "Scan ID: <strong>--</strong>";

        currentImageFile = null;
        currentExampleName = null;
        currentExplanations = null;
        currentActiveMode = "standard";
    }

    // ==================== INGESTION & FILE HANDLERS ====================
    function setupIngestionHandlers() {
        // Upload button triggers
        btnNewScan.addEventListener("click", () => fileInput.click());
        btnChangeActiveScan.addEventListener("click", () => fileInput.click());
        btnRetryGate.addEventListener("click", () => fileInput.click());

        if (heroDropzone) {
            heroDropzone.addEventListener("click", () => fileInput.click());

            ["dragenter", "dragover"].forEach(evt => {
                heroDropzone.addEventListener(evt, (e) => {
                    e.preventDefault(); e.stopPropagation();
                    heroDropzone.classList.add("dragover");
                });
            });

            ["dragleave", "drop"].forEach(evt => {
                heroDropzone.addEventListener(evt, (e) => {
                    e.preventDefault(); e.stopPropagation();
                    heroDropzone.classList.remove("dragover");
                });
            });

            heroDropzone.addEventListener("drop", (e) => {
                const dt = e.dataTransfer;
                if (dt.files && dt.files.length > 0) handleFileSelected(dt.files[0]);
            });
        }

        // Global Drag & drop over viewport
        const viewport = document.getElementById("gradcamViewport");
        if (viewport) {
            ["dragenter", "dragover"].forEach(evt => {
                viewport.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); });
            });
            viewport.addEventListener("drop", (e) => {
                e.preventDefault(); e.stopPropagation();
                const dt = e.dataTransfer;
                if (dt.files && dt.files.length > 0) handleFileSelected(dt.files[0]);
            });
        }

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) handleFileSelected(e.target.files[0]);
        });

        // Quick Clinical Samples buttons on Hero
        samplePillBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const demoName = btn.getAttribute("data-demo");
                if (demoName) analyzeClinicalSample(demoName);
            });
        });
    }

    function setupQuickDemosDropdown() {
        btnQuickDemos.addEventListener("click", (e) => {
            e.stopPropagation();
            quickDemosMenu.classList.toggle("hidden");
        });

        document.addEventListener("click", (e) => {
            if (!quickDemosMenu.contains(e.target) && e.target !== btnQuickDemos) {
                quickDemosMenu.classList.add("hidden");
            }
        });

        demoItems.forEach(item => {
            item.addEventListener("click", () => {
                const demoName = item.getAttribute("data-demo");
                quickDemosMenu.classList.add("hidden");
                if (demoName) analyzeClinicalSample(demoName);
            });
        });
    }

    function handleFileSelected(file) {
        if (!file.type.match("image.*")) {
            alert("Please select a valid ultrasound image file (JPG, PNG, JPEG, WEBP).");
            return;
        }

        currentImageFile = file;
        currentExampleName = null;

        const formData = new FormData();
        formData.append("file", file);
        runInference(formData, false);
    }

    function analyzeClinicalSample(filename) {
        currentImageFile = null;
        currentExampleName = filename;
        runInference({ example_filename: filename }, true);
    }

    // ==================== INFERENCE PIPELINE ====================
    async function runInference(payload, isJson) {
        showAnalyzingState();
        const startTime = Date.now();
        const minDuration = 1100;

        animateProgress();

        try {
            const options = {
                method: "POST",
                headers: isJson ? { "Content-Type": "application/json" } : {},
                body: isJson ? JSON.stringify(payload) : payload
            };

            const response = await fetch("/api/predict", options);
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.message || "Diagnostic inference failed.");
            }

            const data = await response.json();
            const elapsed = Date.now() - startTime;
            const remaining = Math.max(0, minDuration - elapsed);

            setTimeout(() => {
                hideAnalyzingState();
                renderResults(data);
            }, remaining);

        } catch (err) {
            console.error("Inference Error:", err);
            hideAnalyzingState();
            resultStateEmpty.classList.remove("hidden");
            alert(`Diagnostic Notice: ${err.message}`);
        }
    }

    function showAnalyzingState() {
        resultStateEmpty.classList.add("hidden");
        resultStateActive.classList.add("hidden");
        resultStateUnsupported.classList.add("hidden");
        resultStateAnalyzing.classList.remove("hidden");
        scanProgressFill.style.width = "0%";
    }

    function hideAnalyzingState() {
        resultStateAnalyzing.classList.add("hidden");
    }

    function animateProgress() {
        const steps = [
            { text: "Acquiring acoustic ultrasound matrix...", p: 25, d: 0 },
            { text: "Segmenting hypoechoic follicles & measuring calipers...", p: 60, d: 350 },
            { text: "Evaluating 3-class posterior probabilities via deep ConvNet...", p: 85, d: 700 },
            { text: "Generating LayerCAM attribution maps...", p: 100, d: 1000 }
        ];

        steps.forEach(s => {
            setTimeout(() => {
                if (analyzingStepText) analyzingStepText.textContent = s.text;
                if (scanProgressFill) scanProgressFill.style.width = `${s.p}%`;
            }, s.d);
        });
    }

    // ==================== RENDER CLINICAL RESULTS ====================
    function renderResults(data) {
        // Refusal / OOD Quality Gate
        if (data.is_valid_input === false) {
            resultStateActive.classList.add("hidden");
            resultStateUnsupported.classList.remove("hidden");
            if (unsupportedMessageText) {
                unsupportedMessageText.textContent = data.message || "Image does not match expected B-mode ovarian ultrasound acoustics.";
            }
            return;
        }

        resultStateUnsupported.classList.add("hidden");
        resultStateActive.classList.remove("hidden");

        // 1. Primary Diagnosis Verdict Card
        predictionBanner.className = `diagnostic-verdict-card ${data.status_code}`;
        verdictPill.textContent = data.prediction;
        confidencePill.textContent = `${data.confidence} Certainty (${data.top_probability}%)`;

        if (data.prediction === "Dominant Follicle") {
            verdictHeadline.textContent = "Dominant Follicle (DF) Detected";
        } else if (data.prediction === "PCOS") {
            verdictHeadline.textContent = "PCOS-Associated Morphology";
        } else {
            verdictHeadline.textContent = "Normal Ovarian Architecture";
        }
        verdictDescription.textContent = data.prediction_summary;

        // 2. 3-Class Probability Bars
        const probs = data.class_probabilities || {};
        const pDf = probs["Dominant Follicle"] !== undefined ? probs["Dominant Follicle"] : data.df_probability || 0;
        const pPcos = probs["PCOS"] !== undefined ? probs["PCOS"] : data.pcos_probability || 0;
        const pNormal = probs["Normal Ovary"] !== undefined ? probs["Normal Ovary"] : data.normal_probability || 0;

        probValDf.textContent = `${pDf.toFixed(1)}%`;
        probBarDf.style.width = `${Math.min(100, Math.max(0, pDf))}%`;

        probValPcos.textContent = `${pPcos.toFixed(1)}%`;
        probBarPcos.style.width = `${Math.min(100, Math.max(0, pPcos))}%`;

        probValNormal.textContent = `${pNormal.toFixed(1)}%`;
        probBarNormal.style.width = `${Math.min(100, Math.max(0, pNormal))}%`;

        [probRowDf, probRowPcos, probRowNormal].forEach(r => r.classList.remove("highlight-top"));
        if (data.pred_class_idx === 2) probRowDf.classList.add("highlight-top");
        else if (data.pred_class_idx === 1) probRowPcos.classList.add("highlight-top");
        else probRowNormal.classList.add("highlight-top");

        // 3. Quantitative Rotterdam Biomarkers
        const fm = data.follicle_metrics || {};
        const afc = fm.afc !== undefined ? fm.afc : 0;
        metricAfc.textContent = `${afc} Follicles`;

        const isRotterdam = fm.rotterdam_afc_met || false;
        metricRotterdamBadge.textContent = isRotterdam ? "Rotterdam Met (≥12)" : "Normal Reserve (<12)";
        metricRotterdamBadge.className = isRotterdam ? "bm-badge pcos" : "bm-badge normal";

        // Dominant Follicle Count
        const domCount = fm.dominant_count || 0;
        metricDominantCount.textContent = domCount > 0 ? `${domCount} Detected` : "0 Detected";
        metricDominantBadge.textContent = domCount > 0 ? "Pre-Ovulatory Expansion" : "No Dominant Follicle";
        metricDominantBadge.className = domCount > 0 ? "bm-badge amber" : "bm-badge normal";

        // Diameters
        metricMeanDiameter.textContent = `${fm.mean_diameter_mm || 0} mm`;
        metricDiameterRange.textContent = `Range: ${fm.diameter_range_mm || 'N/A'}`;

        // Spatial Distribution & AI Concordance
        metricPattern.textContent = fm.pattern ? fm.pattern.split("(")[0].trim() : "Follicular Reserve";
        metricAiConcordance.textContent = `AI Concordance: ${fm.ai_concordance_pct || 0}%`;

        // 4. Viewport HUD
        hudAttentionText.innerHTML = `Identified: <strong>${afc} Follicles</strong> &bull; Dominant: <strong>${domCount} &ge;10mm</strong> &bull; Subcapsular: <strong>${fm.subcapsular_pct || 0}%</strong>`;
        hudModelTag.innerHTML = `Model: <strong>${data.model_name ? data.model_name.split("(")[0].trim() : "ConvNeXt-Tiny V6"}</strong>`;
        hudScanId.innerHTML = `Scan ID: <strong>#${data.analysis_id || 'SCAN'}</strong>`;

        // Operating Specs (Right)
        if (specModel) specModel.textContent = data.model_name ? data.model_name.split("(")[0].trim() : "ConvNeXt-Tiny V6";
        if (specBenchmark) specBenchmark.textContent = activeModelKey === "convnext_v6_3class" ? "84.9% Acc • 0.8506 F1" : "81.4% Acc • 96.6% DF Recall";
        if (specThreshold) specThreshold.textContent = `${data.operating_threshold || 0.50} (Calibrated)`;
        if (specDate) specDate.textContent = data.analysis_date || "Just now";

        // 5. Measured Follicle Calipers Table
        const inventory = fm.inventory || [];
        if (inventory.length > 0) {
            follicleInventoryCount.textContent = `${inventory.length} Follicles Measured`;
            follicleTableBody.innerHTML = "";

            inventory.forEach(f => {
                const tr = document.createElement("tr");
                const isDom = f.diameter_mm >= 10.0;
                const tagCls = isDom ? "dominant" : (f.is_subcapsular ? "subcapsular" : "central");
                const tagText = isDom ? "Dominant (≥10mm)" : (f.is_subcapsular ? "Subcapsular Peripheral" : "Central Antral");
                const camScore = Math.round((f.cam_activation || 0) * 100);

                tr.innerHTML = `
                    <td><strong class="foll-id-badge">#F${f.id}</strong></td>
                    <td><span class="caliper-diam-val">${f.diameter_mm} mm</span></td>
                    <td><span class="caliper-tag ${tagCls}">${tagText}</span></td>
                    <td><code>[${f.bbox.join(", ")}]</code></td>
                    <td>${f.is_subcapsular ? "Peripheral Ring" : "Central Stroma"}</td>
                    <td><span class="cam-score-val">${camScore}% Focus</span></td>
                `;
                follicleTableBody.appendChild(tr);
            });
            follicleInventoryCard.classList.remove("hidden");
        } else {
            follicleInventoryCard.classList.add("hidden");
        }

        // 6. Multi-Mode Explanations
        currentExplanations = data.explanations;
        currentActiveMode = "standard";
        setActiveViewMode("standard");

        // History
        sessionHistory.unshift({
            id: data.analysis_id,
            date: data.analysis_date,
            prediction: data.prediction,
            confidence: `${data.top_probability}%`,
            status_code: data.status_code,
            imageUrl: data.image_url
        });
        renderHistory();
    }

    // ==================== VIEWPORT MODE SWITCHER ====================
    function setupModeSwitcher() {
        modeTabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const mode = btn.getAttribute("data-mode");
                setActiveViewMode(mode);
            });
        });
    }

    function setActiveViewMode(mode) {
        if (!currentExplanations || !currentExplanations[mode]) return;
        currentActiveMode = mode;

        modeTabBtns.forEach(btn => {
            btn.classList.toggle("active", btn.getAttribute("data-mode") === mode);
        });

        const exp = currentExplanations[mode];
        gradcamImg.src = `${exp.url}?t=${Date.now()}`;
        btnDownloadGradcam.href = exp.url;
        btnDownloadGradcam.download = `pcos_${mode}_view.png`;

        const cfg = MODE_CONFIG[mode] || { showColorbar: true, showAnno: true };
        colorbarContainer.style.display = cfg.showColorbar ? "flex" : "none";
        gradcamAnnotations.style.display = cfg.showAnno ? "block" : "none";
    }

    // ==================== TOP-2 MODEL SWITCHER ====================
    function setupTop2ModelToggle() {
        modelBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetModel = btn.getAttribute("data-model");
                if (targetModel !== activeModelKey) {
                    switchModel(targetModel);
                }
            });
        });
    }

    async function switchModel(modelKey) {
        try {
            const res = await fetch("/api/set-model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_key: modelKey })
            });
            const data = await res.json();
            if (data.status === "success") {
                activeModelKey = modelKey;
                modelBtns.forEach(b => {
                    b.classList.toggle("active", b.getAttribute("data-model") === modelKey);
                });

                // Automatically re-analyze currently active scan
                if (currentExampleName) {
                    analyzeClinicalSample(currentExampleName);
                } else if (currentImageFile) {
                    const formData = new FormData();
                    formData.append("file", currentImageFile);
                    runInference(formData, false);
                }
            }
        } catch (err) {
            console.error("Failed to switch model:", err);
        }
    }

    // ==================== MODALS & HISTORY ====================
    function setupModals() {
        btnHistory.addEventListener("click", () => modalHistory.classList.remove("hidden"));
        btnCriteria.addEventListener("click", () => modalHowItWorks.classList.remove("hidden"));
        btnCloseHistory.addEventListener("click", () => modalHistory.classList.add("hidden"));
        btnCloseHowItWorks.addEventListener("click", () => modalHowItWorks.classList.add("hidden"));

        [modalHistory, modalHowItWorks].forEach(m => {
            m.addEventListener("click", (e) => {
                if (e.target === m) m.classList.add("hidden");
            });
        });
    }

    function renderHistory() {
        if (!historyListContainer) return;
        if (sessionHistory.length === 0) {
            historyListContainer.innerHTML = '<p class="history-empty">No clinical analyses recorded in this session yet.</p>';
            return;
        }

        historyListContainer.innerHTML = "";
        sessionHistory.forEach(item => {
            const row = document.createElement("div");
            row.className = "history-item-row";
            row.innerHTML = `
                <img src="${item.imageUrl}" class="history-item-thumb" alt="Scan">
                <div class="history-item-info">
                    <strong class="history-item-title">${item.prediction} (${item.confidence})</strong>
                    <span class="history-item-date">${item.date} &bull; #${item.id}</span>
                </div>
            `;
            historyListContainer.appendChild(row);
        });
    }

});
