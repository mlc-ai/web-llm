import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const port = Number(globalThis.process.env.WEBLLM_BROWSER_TEST_PORT ?? 4178);
const harnessPath = fileURLToPath(
  new globalThis.URL("../../.browser-test/harness.js", import.meta.url),
);
const modelLibPath = globalThis.process.env.WEBLLM_TEST_MODEL_LIB_PATH;
const modelPath = globalThis.process.env.WEBLLM_TEST_MODEL_PATH;
const modelRoot = modelPath ? resolve(modelPath) : undefined;
const modelPrefix = "/model/resolve/main/";

function sendFile(response, path, contentType) {
  let stat;
  try {
    stat = statSync(path);
  } catch {
    response.writeHead(404).end();
    return;
  }
  if (!stat.isFile()) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, {
    "content-type": contentType,
    "content-length": stat.size,
    "cache-control": "no-store",
  });
  createReadStream(path)
    .on("error", () => response.destroy())
    .pipe(response);
}

createServer((request, response) => {
  if (request.url === "/model.wasm" && modelLibPath) {
    sendFile(response, modelLibPath, "application/wasm");
    return;
  }
  if (request.url?.startsWith(modelPrefix) && modelRoot) {
    const path = resolve(modelRoot, request.url.slice(modelPrefix.length));
    if (!path.startsWith(modelRoot + sep)) {
      response.writeHead(404).end();
      return;
    }
    sendFile(
      response,
      path,
      path.endsWith(".json") ? "application/json" : "application/octet-stream",
    );
    return;
  }
  if (request.url?.startsWith("/harness.js")) {
    response.writeHead(200, { "content-type": "text/javascript" });
    createReadStream(harnessPath).pipe(response);
    return;
  }
  response.writeHead(200, { "content-type": "text/html" });
  response.end(
    '<!doctype html><meta charset="utf-8"><script type="module" src="/harness.js"></script>',
  );
}).listen(port, "127.0.0.1");
