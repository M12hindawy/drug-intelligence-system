// MedLink AI — Service Worker (PWA)
const CACHE_NAME = 'medlink-v2.0.0';
const STATIC_ASSETS = [
    '/',
    '/static/style.css',
    '/static/manifest.json',
    'https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
];

// Install: cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            console.log('[SW] Caching static assets');
            return cache.addAll(STATIC_ASSETS.map(url => new Request(url, { mode: 'no-cors' })));
        }).catch(err => console.log('[SW] Cache failed:', err))
    );
    self.skipWaiting();
});

// Activate: clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

// Fetch: Network first, fallback to cache
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Skip API calls and WebSocket — always go to network
    if (url.pathname.startsWith('/api/') ||
        url.pathname.startsWith('/socket.io') ||
        event.request.method !== 'GET') {
        return;
    }

    event.respondWith(
        fetch(event.request)
            .then(response => {
                // Cache successful responses
                if (response.ok) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return response;
            })
            .catch(() => {
                // Offline fallback
                return caches.match(event.request).then(cached => {
                    if (cached) return cached;
                    // Return offline page for navigation requests
                    if (event.request.mode === 'navigate') {
                        return caches.match('/');
                    }
                });
            })
    );
});
