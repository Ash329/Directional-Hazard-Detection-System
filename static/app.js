document.addEventListener("DOMContentLoaded", () => {
    initLiveDetection();
});

function initLiveDetection() {
    const stage = document.querySelector("[data-camera-stage]");
    const video = document.querySelector("[data-camera-view]");
    const overlayCanvas = document.querySelector("[data-overlay-canvas]");
    const captureCanvas = document.querySelector("[data-capture-canvas]");
    const placeholder = document.querySelector("[data-camera-placeholder]");
    const liveRegion = document.querySelector("[data-live-region]");
    const cameraButton = document.querySelector("[data-camera-toggle]");
    const flipButton = document.querySelector("[data-camera-flip]");
    const audioButton = document.querySelector("[data-audio-toggle]");
    const cameraLabel = document.querySelector("[data-camera-label]");
    const audioLabel = document.querySelector("[data-audio-label]");
    const statusPill = document.querySelector("[data-status]");
    const statusText = document.querySelector("[data-status-text]");
    const placeholderTitle = document.querySelector("[data-placeholder-title]");
    const placeholderText = document.querySelector("[data-placeholder-text]");
    const alertCard = document.querySelector("[data-alert]");
    const alertLevel = document.querySelector("[data-alert-level]");
    const alertText = document.querySelector("[data-alert-text]");
    const directionCells = document.querySelectorAll("[data-direction-cell]");

    if (!stage || !video || !overlayCanvas || !captureCanvas || !cameraButton || !flipButton || !audioButton) {
        return;
    }

    const supportsSpeech = "speechSynthesis" in window;
    let stream = null;
    let facingMode = "environment";
    let detectionTimer = null;
    let isAnalyzing = false;
    let speechEnabled = supportsSpeech;
    // Speech buffer: one alert plays at a time, at most one waits behind it.
    const REPEAT_COOLDOWN_MS = 8000;  // don't re-announce the same object within this window...
    const PENDING_MAX_AGE_MS = 3000;  // ...and drop a queued alert if the scene has likely moved on
    const announcedAt = new Map();    // "label|direction" -> { at, urgency }
    let currentAlert = null;          // { utterance, urgency }
    let pendingAlert = null;          // { key, text, urgency, createdAt }
    let failedRequests = 0;

    const SEVERITY_COLORS = {
        hazard: "#e5383b",
        caution: "#f5b301",
        ignore: "#ffffff",
    };
    const SEVERITY_RANK = { hazard: 2, caution: 1 };

    const setStatus = (state, text) => {
        if (statusPill) {
            statusPill.dataset.state = state;
        }
        if (statusText) {
            statusText.textContent = text;
        }
    };

    const setPlaceholder = (title, text) => {
        if (placeholderTitle) {
            placeholderTitle.textContent = title;
        }
        if (placeholderText) {
            placeholderText.textContent = text;
        }
    };

    const setAlert = (level, label, text) => {
        if (!alertCard) {
            return;
        }
        alertCard.dataset.level = level;
        alertLevel.textContent = label;
        alertText.textContent = text;
    };

    const setDirectionLevels = (levels) => {
        directionCells.forEach((cell) => {
            const level = levels[cell.dataset.directionCell];
            if (level) {
                cell.dataset.level = level;
            } else {
                delete cell.dataset.level;
            }
        });
    };

    const updateControls = () => {
        const cameraLive = Boolean(stream);
        cameraButton.classList.toggle("is-active", cameraLive);
        cameraButton.setAttribute("aria-pressed", cameraLive ? "true" : "false");
        flipButton.disabled = !cameraLive;
        flipButton.setAttribute("aria-disabled", cameraLive ? "false" : "true");
        if (cameraLabel) {
            cameraLabel.textContent = cameraLive ? "Stop" : "Start";
        }
        if (audioLabel) {
            audioLabel.textContent = !supportsSpeech ? "No voice" : speechEnabled ? "Voice on" : "Voice off";
        }
        audioButton.disabled = !supportsSpeech;
        audioButton.classList.toggle("is-active", speechEnabled && supportsSpeech);
        audioButton.classList.toggle("is-muted", !speechEnabled || !supportsSpeech);
        audioButton.setAttribute("aria-pressed", speechEnabled && supportsSpeech ? "true" : "false");
    };

    const updateStageState = () => {
        const hasVideo = Boolean(stream) && video.readyState >= 2 && video.videoWidth > 0;
        stage.classList.toggle("is-live", hasVideo);

        if (placeholder) {
            placeholder.hidden = hasVideo;
        }
    };

    const clearOverlay = () => {
        const context = overlayCanvas.getContext("2d");
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    const sizeOverlayCanvas = () => {
        const bounds = stage.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        overlayCanvas.width = Math.max(1, Math.floor(bounds.width * ratio));
        overlayCanvas.height = Math.max(1, Math.floor(bounds.height * ratio));

        const frame = getFrameTransform(bounds.width, bounds.height);
        stage.style.setProperty("--frame-left", `${frame.offsetX}px`);
        stage.style.setProperty("--frame-width", `${frame.width}px`);
    };

    const matchStageToVideo = () => {
        if (video.videoWidth && video.videoHeight) {
            stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
        }
    };

    // The video uses object-fit: contain, so the whole frame the model sees is visible
    // but may be letterboxed. Map normalized frame coordinates onto the stage the same way.
    const getFrameTransform = (stageWidth, stageHeight) => {
        const videoWidth = video.videoWidth || stageWidth;
        const videoHeight = video.videoHeight || stageHeight;
        const scale = Math.min(stageWidth / videoWidth, stageHeight / videoHeight);
        const drawnWidth = videoWidth * scale;
        const drawnHeight = videoHeight * scale;
        return {
            width: drawnWidth,
            height: drawnHeight,
            offsetX: (stageWidth - drawnWidth) / 2,
            offsetY: (stageHeight - drawnHeight) / 2,
        };
    };

    const stopSpeech = () => {
        pendingAlert = null;
        currentAlert = null;
        if (supportsSpeech) {
            window.speechSynthesis.cancel();
        }
    };

    const stopDetectionLoop = () => {
        if (detectionTimer) {
            window.clearInterval(detectionTimer);
            detectionTimer = null;
        }
        isAnalyzing = false;
        clearOverlay();
    };

    const stopCamera = () => {
        stopDetectionLoop();
        stopSpeech();

        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }

        video.pause();
        video.srcObject = null;
        failedRequests = 0;
        setStatus("off", "Camera off");
        setPlaceholder("Camera is off", "Press Start to begin scanning for hazards.");
        setAlert("idle", "Standby", "Camera is off.");
        setDirectionLevels({});
        updateStageState();
        updateControls();
    };

    // Approaching objects outrank everything else, then hazard over caution.
    const alertUrgency = (detection) => {
        const severityScore = detection.severity === "hazard" ? 2 : detection.severity === "caution" ? 1 : 0;
        if (!severityScore) {
            return 0;
        }
        return severityScore + (detection.motion === "approaching" ? 2 : 0);
    };

    const alertPhrase = (detection) => {
        const directions = { left: "on your left", center: "ahead", right: "on your right" };
        const label = detection.label.charAt(0).toUpperCase() + detection.label.slice(1);
        const where = directions[detection.direction] || "";
        return detection.motion === "approaching" ? `${label} ${where}, approaching.` : `${label} ${where}.`;
    };

    // Skip an object we just announced, unless it has become more urgent since.
    const isOnCooldown = (alert, now) => {
        const previous = announcedAt.get(alert.key);
        return Boolean(previous) && now - previous.at < REPEAT_COOLDOWN_MS && alert.urgency <= previous.urgency;
    };

    const speakAlert = (alert) => {
        const utterance = new SpeechSynthesisUtterance(alert.text);
        utterance.rate = 1.05;

        const finish = () => {
            // Ignore events from an utterance that was preempted.
            if (!currentAlert || currentAlert.utterance !== utterance) {
                return;
            }
            currentAlert = null;

            const next = pendingAlert;
            pendingAlert = null;
            const now = Date.now();
            if (next && speechEnabled && now - next.createdAt < PENDING_MAX_AGE_MS && !isOnCooldown(next, now)) {
                speakAlert(next);
            }
        };
        utterance.onend = finish;
        utterance.onerror = finish;

        currentAlert = { utterance, urgency: alert.urgency };
        announcedAt.set(alert.key, { at: Date.now(), urgency: alert.urgency });
        if (liveRegion) {
            liveRegion.textContent = alert.text;
        }
        window.speechSynthesis.speak(utterance);
    };

    const announceDetections = (detections) => {
        if (!supportsSpeech || !speechEnabled) {
            return;
        }

        const now = Date.now();
        announcedAt.forEach((entry, key) => {
            if (now - entry.at > REPEAT_COOLDOWN_MS) {
                announcedAt.delete(key);
            }
        });

        // Backend order (severity, size, confidence) breaks ties between equally urgent alerts.
        let best = null;
        detections.forEach((detection) => {
            const urgency = alertUrgency(detection);
            if (!urgency) {
                return;
            }
            const alert = {
                key: `${detection.label}|${detection.direction}`,
                text: alertPhrase(detection),
                urgency,
                createdAt: now,
            };
            if (!isOnCooldown(alert, now) && (!best || urgency > best.urgency)) {
                best = alert;
            }
        });

        if (!best) {
            return;
        }

        if (!currentAlert) {
            speakAlert(best);
            return;
        }

        // Only something approaching may cut off an alert that is still being spoken.
        if (best.urgency >= 3 && best.urgency > currentAlert.urgency) {
            pendingAlert = null;
            currentAlert = null;
            window.speechSynthesis.cancel();
            speakAlert(best);
            return;
        }

        // Otherwise wait in the single buffer slot; a fresher or more urgent alert replaces it.
        if (!pendingAlert || best.urgency >= pendingAlert.urgency) {
            pendingAlert = best;
        }
    };

    const roundedRect = (context, x, y, w, h, r) => {
        context.beginPath();
        if (context.roundRect) {
            context.roundRect(x, y, w, h, r);
        } else {
            context.rect(x, y, w, h);
        }
    };

    const drawLabel = (context, text, x, y, color, maxX) => {
        context.font = "700 13px Manrope, system-ui, sans-serif";
        const paddingX = 8;
        const labelHeight = 22;
        const labelWidth = context.measureText(text).width + paddingX * 2;
        const labelX = Math.max(0, Math.min(x, maxX - labelWidth));
        const labelY = y - labelHeight - 4 >= 0 ? y - labelHeight - 4 : y + 4;

        context.fillStyle = color;
        roundedRect(context, labelX, labelY, labelWidth, labelHeight, 6);
        context.fill();

        context.fillStyle = "#0a1214";
        context.textBaseline = "middle";
        context.fillText(text, labelX + paddingX, labelY + labelHeight / 2 + 1);
    };

    const drawOverlay = (detections, primaryDirection) => {
        const context = overlayCanvas.getContext("2d");
        const ratio = window.devicePixelRatio || 1;
        const width = overlayCanvas.width / ratio;
        const height = overlayCanvas.height / ratio;

        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);

        const frame = getFrameTransform(width, height);
        const leftBoundary = frame.offsetX + frame.width / 3;
        const rightBoundary = frame.offsetX + (frame.width / 3) * 2;

        const primary = detections.find((detection) => detection.direction === primaryDirection);
        if (primary && SEVERITY_RANK[primary.severity]) {
            const zoneStarts = { left: frame.offsetX, center: leftBoundary, right: rightBoundary };
            const zoneEnds = { left: leftBoundary, center: rightBoundary, right: frame.offsetX + frame.width };
            const zoneStart = zoneStarts[primaryDirection];
            context.fillStyle = primary.severity === "hazard" ? "rgba(229, 56, 59, 0.16)" : "rgba(245, 179, 1, 0.14)";
            context.fillRect(zoneStart, 0, zoneEnds[primaryDirection] - zoneStart, height);
        }

        context.strokeStyle = "rgba(255, 255, 255, 0.28)";
        context.lineWidth = 1;
        context.setLineDash([6, 8]);
        context.beginPath();
        context.moveTo(leftBoundary, 0);
        context.lineTo(leftBoundary, height);
        context.moveTo(rightBoundary, 0);
        context.lineTo(rightBoundary, height);
        context.stroke();
        context.setLineDash([]);

        detections.forEach((detection) => {
            if (!detection.box) {
                return;
            }

            const x = frame.offsetX + detection.box.x1 * frame.width;
            const y = frame.offsetY + detection.box.y1 * frame.height;
            const boxWidth = (detection.box.x2 - detection.box.x1) * frame.width;
            const boxHeight = (detection.box.y2 - detection.box.y1) * frame.height;
            const color = SEVERITY_COLORS[detection.severity] || SEVERITY_COLORS.ignore;

            context.strokeStyle = color;
            context.lineWidth = 3;
            roundedRect(context, x, y, boxWidth, boxHeight, 10);
            context.stroke();

            const label = detection.proximity && detection.proximity !== "far"
                ? `${detection.label} · ${detection.proximity}`
                : detection.label;
            drawLabel(context, label, x, y, color, width);
        });
    };

    const describeDetection = (detection) => {
        const directions = { left: "on the left", center: "ahead", right: "on the right" };
        const label = detection.label.charAt(0).toUpperCase() + detection.label.slice(1);
        return `${label} ${directions[detection.direction] || ""}`.trim();
    };

    const updateReadout = (payload) => {
        const detections = payload.detections || [];
        const levels = {};
        let worst = null;

        detections.forEach((detection) => {
            const rank = SEVERITY_RANK[detection.severity] || 0;
            if (!rank) {
                return;
            }
            if (rank > (SEVERITY_RANK[levels[detection.direction]] || 0)) {
                levels[detection.direction] = detection.severity;
            }
            if (!worst || rank > SEVERITY_RANK[worst.severity]) {
                worst = detection;
            }
        });

        setDirectionLevels(levels);

        if (!worst) {
            const seen = detections.length ? `${detections.length} object${detections.length > 1 ? "s" : ""} in view, none close.` : "Nothing in the way.";
            setAlert("clear", "Path clear", seen);
            return;
        }

        const level = worst.severity;
        const text = payload.summary_text && payload.has_hazard
            ? payload.summary_text.replace(/^(Hazard|Caution): /, "")
            : describeDetection(worst);
        setAlert(level, level === "hazard" ? "Hazard" : "Caution", text);
    };

    const analyzeFrame = async () => {
        if (!stream || isAnalyzing || video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
            return;
        }

        isAnalyzing = true;

        const captureWidth = Math.min(640, video.videoWidth);
        const captureHeight = Math.max(1, Math.round(captureWidth * (video.videoHeight / video.videoWidth)));
        const captureContext = captureCanvas.getContext("2d");

        captureCanvas.width = captureWidth;
        captureCanvas.height = captureHeight;
        captureContext.drawImage(video, 0, 0, captureWidth, captureHeight);

        try {
            const response = await fetch("/api/live-detect", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    image: captureCanvas.toDataURL("image/jpeg", 0.72),
                }),
            });

            if (!response.ok) {
                throw new Error(`Detection request failed with ${response.status}`);
            }

            const payload = await response.json();
            failedRequests = 0;
            setStatus("live", "Scanning");
            drawOverlay(payload.detections || [], payload.primary_direction || null);
            updateReadout(payload);

            announceDetections(payload.detections || []);
        } catch (error) {
            console.error("Live detection failed.", error);
            clearOverlay();
            failedRequests += 1;
            if (stream && failedRequests >= 2) {
                setStatus("error", "No connection");
                setAlert("idle", "Connection issue", "Can't reach the detection server. Retrying...");
                setDirectionLevels({});
            }
        } finally {
            isAnalyzing = false;
        }
    };

    const startDetectionLoop = () => {
        stopDetectionLoop();
        detectionTimer = window.setInterval(analyzeFrame, 1000);
        analyzeFrame();
    };

    const startCamera = async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setStatus("error", "No camera");
            setPlaceholder("Camera not available", "This browser can't access a camera. Open the app over HTTPS or on localhost.");
            updateControls();
            return;
        }

        if (stream) {
            return;
        }

        setStatus("starting", "Starting");
        setPlaceholder("Starting camera", "Allow camera access if your browser asks.");

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: facingMode },
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            });

            video.srcObject = stream;
            await video.play();
            matchStageToVideo();
            sizeOverlayCanvas();
            updateStageState();
            updateControls();
            setAlert("idle", "Starting", "Checking the view...");
            startDetectionLoop();
        } catch (error) {
            console.error("Camera access failed.", error);
            stopCamera();
            const denied = error && error.name === "NotAllowedError";
            setStatus("error", denied ? "Blocked" : "No camera");
            setPlaceholder(
                denied ? "Camera access blocked" : "Couldn't start the camera",
                denied
                    ? "Allow camera access in your browser settings, then press Start."
                    : "Check that no other app is using the camera, then press Start."
            );
        }
    };

    const toggleCamera = async () => {
        if (stream) {
            stopCamera();
            return;
        }

        await startCamera();
    };

    const flipCamera = async () => {
        facingMode = facingMode === "environment" ? "user" : "environment";
        if (!stream) {
            return;
        }

        stopCamera();
        await startCamera();
    };

    const toggleAudio = () => {
        speechEnabled = !speechEnabled;
        if (!speechEnabled) {
            stopSpeech();
        }
        updateControls();
    };

    cameraButton.addEventListener("click", toggleCamera);
    flipButton.addEventListener("click", flipCamera);
    audioButton.addEventListener("click", toggleAudio);
    video.addEventListener("loadedmetadata", () => {
        matchStageToVideo();
        sizeOverlayCanvas();
        updateStageState();
    });
    window.addEventListener("resize", sizeOverlayCanvas);
    window.addEventListener("beforeunload", stopCamera);

    updateControls();
    startCamera();
}
