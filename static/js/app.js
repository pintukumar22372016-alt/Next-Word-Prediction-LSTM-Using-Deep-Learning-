document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const inputPrompt = document.getElementById("input-prompt");
    const ghostPreview = document.getElementById("ghost-preview");
    const quickWordChips = document.getElementById("quick-word-chips");
    const predictionsList = document.getElementById("predictions-list");
    const btnClear = document.getElementById("btn-clear-text");
    const statusIndicator = document.getElementById("status-indicator");
    const statusText = document.getElementById("status-text");
    const inferenceTimeBadge = document.getElementById("inference-time-badge");

    // Sliders
    const sliderWords = document.getElementById("slider-words");
    const labelNumWords = document.getElementById("label-num-words");
    const sliderTemp = document.getElementById("slider-temp");
    const labelTemperature = document.getElementById("label-temperature");

    // Generation
    const btnGenerate = document.getElementById("btn-generate-stream");
    const generationBox = document.getElementById("generation-output-box");
    const generationText = document.getElementById("generation-text");
    const btnCopy = document.getElementById("btn-copy-output");

    // Specs
    const specArch = document.getElementById("spec-arch");
    const specVocab = document.getElementById("spec-vocab");
    const specTop1 = document.getElementById("spec-top1");
    const specTop5 = document.getElementById("spec-top5");

    // Modal
    const btnToggleInfo = document.getElementById("btn-toggle-info");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const infoModal = document.getElementById("info-modal");

    // State
    let debounceTimer = null;
    let currentTopCandidate = "";
    let isModelReady = false;

    // Initialize & Fetch Model Info
    async function initModelInfo() {
        try {
            const res = await fetch("/api/info");
            const data = await res.json();

            if (data.ready) {
                isModelReady = true;
                statusIndicator.className = "status-pill status-ready";
                statusText.textContent = "Model Online (BiLSTM)";
                if (specArch) specArch.textContent = data.model_architecture || "BiLSTM (96) + LSTM (128)";
                if (specVocab) specVocab.textContent = `${data.total_words.toLocaleString()} tokens`;
                if (specTop1) {
                    specTop1.textContent = data.test_top1_accuracy !== null 
                        ? `${(data.test_top1_accuracy * 100).toFixed(1)}%` 
                        : (data.final_accuracy ? `${(data.final_accuracy * 100).toFixed(1)}%` : "N/A");
                }
                if (specTop5) {
                    specTop5.textContent = data.test_top5_accuracy !== null 
                        ? `${(data.test_top5_accuracy * 100).toFixed(1)}%` 
                        : "Active";
                }
            } else {
                statusIndicator.className = "status-pill status-loading";
                statusText.textContent = "Training / Initializing...";
                setTimeout(initModelInfo, 3000);
            }
        } catch (err) {
            console.error("Failed to connect to API:", err);
            statusIndicator.className = "status-pill status-loading";
            statusText.textContent = "Offline (Reconnecting...)";
            setTimeout(initModelInfo, 4000);
        }
    }

    // Prediction Header Elements
    const sourceBadge = document.getElementById("source-badge");
    const predictionDesc = document.getElementById("prediction-source-desc");
    const spellingBanner = document.getElementById("spelling-banner");
    const spellingText = document.getElementById("spelling-text");

    // Debounced Predict Function
    function triggerPrediction() {
        clearTimeout(debounceTimer);
        const text = inputPrompt.value.trim();

        if (!text) {
            renderEmptyState();
            ghostPreview.textContent = "";
            currentTopCandidate = "";
            if (spellingBanner) spellingBanner.classList.add("hidden");
            if (sourceBadge) {
                sourceBadge.className = "source-pill source-none";
                sourceBadge.textContent = "Awaiting Input";
            }
            if (predictionDesc) predictionDesc.textContent = "Please enter text above to see predictions.";
            return;
        }

        debounceTimer = setTimeout(() => {
            fetchPredictions(text);
        }, 180);
    }

    async function fetchPredictions(text) {
        const startTime = performance.now();
        if (predictionsList) predictionsList.classList.add("loading-pulse");

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, top_k: 5 }),
            });

            const elapsed = Math.round(performance.now() - startTime);
            inferenceTimeBadge.textContent = `⚡ ${elapsed}ms`;

            if (predictionsList) predictionsList.classList.remove("loading-pulse");

            if (!res.ok) {
                renderErrorState();
                return;
            }

            const data = await res.json();

            // Update Prediction Source Badge & Message
            if (sourceBadge) {
                if (data.source === "lstm") {
                    sourceBadge.className = "source-pill source-lstm";
                    sourceBadge.textContent = "⚡ LSTM Model";
                } else if (data.source === "fallback") {
                    sourceBadge.className = "source-pill source-fallback";
                    sourceBadge.textContent = "📚 N-Gram Fallback";
                } else {
                    sourceBadge.className = "source-pill source-none";
                    sourceBadge.textContent = "Awaiting Input";
                }
            }

            if (predictionDesc && data.message) {
                predictionDesc.textContent = data.message;
            }

            // Spelling correction notice
            if (spellingBanner) {
                if (data.has_spelling_correction && data.corrected_text) {
                    spellingBanner.classList.remove("hidden");
                    if (spellingText) spellingText.textContent = `"${data.corrected_text}"`;
                } else {
                    spellingBanner.classList.add("hidden");
                }
            }

            renderPredictions(data.predictions, data.source);
        } catch (err) {
            if (predictionsList) predictionsList.classList.remove("loading-pulse");
            console.error("Predict error:", err);
            renderErrorState();
        }
    }

    function renderPredictions(candidates, source) {
        if (!candidates || candidates.length === 0) {
            quickWordChips.innerHTML = '<span class="empty-hint">Awaiting input...</span>';
            predictionsList.innerHTML = `
                <div class="placeholder-state">
                    <div class="placeholder-icon">🔮</div>
                    <p>No suggestions available</p>
                    <span>Type words to see real-time suggestions.</span>
                </div>
            `;
            ghostPreview.textContent = "";
            currentTopCandidate = "";
            return;
        }

        currentTopCandidate = candidates[0].word;
        ghostPreview.textContent = `Next candidate: "${currentTopCandidate}" (Press Tab)`;

        // Render Quick Chips
        quickWordChips.innerHTML = candidates
            .slice(0, 4)
            .map(
                (c, idx) => `
                <button class="candidate-pill" data-word="${c.word}" title="${source === 'lstm' ? 'Model Confidence' : 'N-Gram Frequency'}: ${c.percentage}%">
                    <span>${c.word}</span>
                    <span class="pill-prob">${c.percentage}%</span>
                </button>
            `
            )
            .join("");

        // Render Probability Bars
        predictionsList.innerHTML = candidates
            .map(
                (c) => {
                    const typeLabel = c.score_type === "softmax_confidence" ? "Softmax" : "N-Gram";
                    return `
                    <div class="prediction-row" data-word="${c.word}" title="${typeLabel} Score: ${c.percentage}% (Relative Share: ${c.relative_share || c.percentage}%)">
                        <div class="pred-header">
                            <span class="pred-word">${c.word}</span>
                            <span class="pred-pct">
                                ${c.percentage}%
                                <span class="score-type-label">${typeLabel}</span>
                            </span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: ${Math.max(c.relative_share || c.percentage, 6)}%"></div>
                        </div>
                    </div>
                `;
                }
            )
            .join("");
    }

    function renderEmptyState() {
        quickWordChips.innerHTML = '<span class="empty-hint">Start typing above to trigger real-time predictions...</span>';
        predictionsList.innerHTML = `
            <div class="placeholder-state">
                <div class="placeholder-icon">🔮</div>
                <p>Awaiting input...</p>
                <span>Type text in the input box to visualize next-word predictions.</span>
            </div>
        `;
    }

    function renderErrorState() {
        quickWordChips.innerHTML = '<span class="empty-hint">Inference engine initializing...</span>';
    }

    function appendWord(word) {
        if (!word) return;
        const current = inputPrompt.value.trim();
        inputPrompt.value = current ? `${current} ${word} ` : `${word} `;
        inputPrompt.focus();
        triggerPrediction();
    }

    // Tab Key Autocomplete Handler
    inputPrompt.addEventListener("keydown", (e) => {
        if (e.key === "Tab" && currentTopCandidate) {
            e.preventDefault();
            appendWord(currentTopCandidate);
        }
    });

    // Keystroke typing listener
    inputPrompt.addEventListener("input", triggerPrediction);

    // Click candidate pill or probability bar
    document.addEventListener("click", (e) => {
        const pill = e.target.closest(".candidate-pill") || e.target.closest(".prediction-row");
        if (pill && pill.dataset.word) {
            appendWord(pill.dataset.word);
        }

        const chipBtn = e.target.closest(".chip-btn");
        if (chipBtn && chipBtn.dataset.seed) {
            inputPrompt.value = chipBtn.dataset.seed + " ";
            inputPrompt.focus();
            triggerPrediction();
        }
    });

    // Clear input
    btnClear.addEventListener("click", () => {
        inputPrompt.value = "";
        triggerPrediction();
        inputPrompt.focus();
    });

    // Sliders
    sliderWords.addEventListener("input", (e) => {
        labelNumWords.textContent = `${e.target.value} words`;
    });

    sliderTemp.addEventListener("input", (e) => {
        labelTemperature.textContent = Number(e.target.value).toFixed(1);
    });

    // Generate Continuation Handler
    btnGenerate.addEventListener("click", async () => {
        const text = inputPrompt.value.trim();
        if (!text) {
            alert("Please type a starting prompt before generating continuation.");
            inputPrompt.focus();
            return;
        }

        btnGenerate.disabled = true;
        btnGenerate.innerHTML = `
            <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10"></path>
            </svg>
            <span>Generating...</span>
        `;

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text,
                    num_words: parseInt(sliderWords.value, 10),
                    temperature: parseFloat(sliderTemp.value),
                }),
            });

            const data = await res.json();
            if (res.ok && data.completed_text) {
                generationBox.classList.remove("hidden");
                animateGeneratedTokens(data.original_text, data.generated_words);
            } else {
                alert(data.error || "Generation error.");
            }
        } catch (err) {
            console.error(err);
            alert("Failed to communicate with generation backend.");
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                <span>Generate More Words</span>
            `;
        }
    });

    function animateGeneratedTokens(prefix, tokens) {
        generationText.innerHTML = `<span>${prefix}</span> `;
        let idx = 0;

        const interval = setInterval(() => {
            if (idx >= tokens.length) {
                clearInterval(interval);
                return;
            }
            const span = document.createElement("span");
            span.className = "generated-token";
            span.textContent = tokens[idx];
            generationText.appendChild(span);
            generationText.appendChild(document.createTextNode(" "));
            idx++;
        }, 120);
    }

    // Copy Generated Output
    btnCopy.addEventListener("click", () => {
        const text = generationText.textContent.trim();
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            const btnSpan = btnCopy.querySelector("span");
            const orig = btnSpan.textContent;
            btnSpan.textContent = "Copied!";
            setTimeout(() => {
                btnSpan.textContent = orig;
            }, 1800);
        });
    });

    // Info Modal
    btnToggleInfo.addEventListener("click", () => {
        infoModal.classList.remove("hidden");
    });

    btnCloseModal.addEventListener("click", () => {
        infoModal.classList.add("hidden");
    });

    infoModal.addEventListener("click", (e) => {
        if (e.target === infoModal) {
            infoModal.classList.add("hidden");
        }
    });

    // Start initialization
    initModelInfo();
});
