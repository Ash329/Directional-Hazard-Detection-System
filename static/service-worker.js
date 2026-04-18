const VERSION = "hazard-detect-v1";
const APP_SHELL = [
    "/",
    "/static/styles.css",
    "/static/app.js",
    "/static/manifest.webmanifest",
    "/static/icons/icon-192.png",
    "/static/icons/icon-512.png",
    "/static/icons/apple-touch-icon.png",
    "/static/icons/favicon-32.png",
];

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(VERSION).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting())
    );
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches
            .keys()
            .then((keys) => Promise.all(keys.filter((key) => key !== VERSION).map((key) => caches.delete(key))))
            .then(() => self.clients.claim())
    );
});

self.addEventListener("fetch", (event) => {
    const { request } = event;

    if (request.method !== "GET") {
        return;
    }

    const url = new URL(request.url);

    // Never cache the detection API — it must always hit the network (fresh frames).
    if (url.pathname.startsWith("/api/")) {
        return;
    }

    // App shell: network-first for the HTML document, cache-first for static assets.
    if (request.mode === "navigate") {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    const copy = response.clone();
                    caches.open(VERSION).then((cache) => cache.put(request, copy));
                    return response;
                })
                .catch(() => caches.match(request).then((cached) => cached || caches.match("/")))
        );
        return;
    }

    if (url.origin === self.location.origin) {
        event.respondWith(
            caches.match(request).then((cached) => {
                if (cached) {
                    return cached;
                }
                return fetch(request)
                    .then((response) => {
                        if (response.ok && response.type === "basic") {
                            const copy = response.clone();
                            caches.open(VERSION).then((cache) => cache.put(request, copy));
                        }
                        return response;
                    })
                    .catch(() => cached);
            })
        );
    }
});
