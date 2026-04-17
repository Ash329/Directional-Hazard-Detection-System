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

    if (!stage || !video || !overlayCanvas || !captureCanvas || !cameraButton || !flipButton || !audioButton) {
        return;
    }

    const supportsSpeech = "speechSynthesis" in window;
    let stream = null;
    let facingMode = "environment";
    let detectionTimer = null;
    let isAnalyzing = false;
    let speechEnabled = supportsSpeech;
    let lastSpokenText = "";
    let lastSpokenAt = 0;

    const updateControls = () => {
        const cameraLive = Boolean(stream);
        cameraButton.classList.toggle("is-active", cameraLive);
        cameraButton.setAttribute("aria-pressed", cameraLive ? "true" : "false");
        flipButton.disabled = !cameraLive;
        flipButton.setAttribute("aria-disabled", cameraLive ? "false" : "true");
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
        context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    const sizeOverlayCanvas = () => {
        const bounds = stage.getBoundingClientRect();
        overlayCanvas.width = Math.max(1, Math.floor(bounds.width));
        overlayCanvas.height = Math.max(1, Math.floor(bounds.height));
    };

    const stopSpeech = () => {
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
        updateStageState();
        updateControls();
    };

    const speakSummary = (summaryText) => {
        if (!supportsSpeech || !speechEnabled || !summaryText) {
            return;
        }

        const normalized = summaryText.trim();
        if (!normalized) {
            return;
        }

        const now = Date.now();
        const repeatedMessage = normalized === lastSpokenText;
        const minimumGap = repeatedMessage ? 7000 : 1800;
        if (now - lastSpokenAt < minimumGap) {
            return;
        }

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(normalized);
        utterance.rate = 1;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);

        lastSpokenText = normalized;
        lastSpokenAt = now;
    };

    const drawOverlay = (detections, primaryDirection) => {
        const context = overlayCanvas.getContext("2d");
        const width = overlayCanvas.width;
        const height = overlayCanvas.height;

        context.clearRect(0, 0, width, height);

        const leftBoundary = width / 3;
        const rightBoundary = (width / 3) * 2;
        const fillColors = {
            left: "rgba(29, 143, 135, 0.16)",
            center: "rgba(237, 139, 99, 0.16)",
            right: "rgba(22, 40, 48, 0.16)",
        };

        if (primaryDirection && fillColors[primaryDirection]) {
            const fillX = primaryDirection === "left" ? 0 : primaryDirection === "center" ? leftBoundary : rightBoundary;
            context.fillStyle = fillColors[primaryDirection];
            context.fillRect(fillX, 0, width / 3, height);
        }

        context.strokeStyle = "rgba(255, 255, 255, 0.22)";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(leftBoundary, 0);
        context.lineTo(leftBoundary, height);
        context.moveTo(rightBoundary, 0);
        context.lineTo(rightBoundary, height);
        context.stroke();

        detections.forEach((detection) => {
            if (!detection.box) {
                return;
            }

            const x = detection.box.x1 * width;
            const y = detection.box.y1 * height;
            const boxWidth = (detection.box.x2 - detection.box.x1) * width;
            const boxHeight = (detection.box.y2 - detection.box.y1) * height;
            const strokeColors = {
                left: "#1d8f87",
                center: "#ed8b63",
                right: "#102830",
            };

            context.strokeStyle = strokeColors[detection.direction] || "#102830";
            context.lineWidth = 3;
            context.strokeRect(x, y, boxWidth, boxHeight);

            context.fillStyle = context.strokeStyle;
            context.beginPath();
            context.arc(x + 8, y + 8, 4, 0, Math.PI * 2);
            context.fill();
        });
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
            drawOverlay(payload.detections || [], payload.primary_direction || null);

            if (liveRegion) {
                liveRegion.textContent = payload.summary_text || "";
            }

            speakSummary(payload.summary_text || "");
        } catch (error) {
            console.error("Live detection failed.", error);
            clearOverlay();
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
            updateControls();
            return;
        }

        if (stream) {
            return;
        }

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
            sizeOverlayCanvas();
            updateStageState();
            updateControls();
            startDetectionLoop();
        } catch (error) {
            console.error("Camera access failed.", error);
            stopCamera();
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
        sizeOverlayCanvas();
        updateStageState();
    });
    window.addEventListener("resize", sizeOverlayCanvas);
    window.addEventListener("beforeunload", stopCamera);

    updateControls();
    startCamera();
}
