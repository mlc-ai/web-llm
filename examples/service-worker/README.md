# WebLLM Service Worker Example

This example shows how we can create a page with Web-LLM running in service worker.

The worker constructs `ServiceWorkerMLCEngineHandler` during initial script
evaluation. Keep that initialization at the top level: an already-active service
worker can be restarted without receiving another `activate` event.

```bash
npm install
npm run build
```
