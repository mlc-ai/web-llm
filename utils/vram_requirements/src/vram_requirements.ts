import ModelRecord from "@mlc-ai/web-llm";
import appConfig from "./app-config"; // Modify this to inspect vram requirement for models of choice
import * as tvmjs from "@mlc-ai/web-runtime";
import log from "loglevel";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

interface AppConfig {
  model_list: Array<ModelRecord>;
}

const dtypeBytesMap = new Map<string, number>([
  ["uint32", 4],
  ["uint16", 2],
  ["float32", 4],
  ["float16", 4],
]);

async function main() {
  const config: AppConfig = appConfig;
  let report = "";
  for (let i = 0; i < config.model_list.length; ++i) {
    // 1. Read each model record
    const modelRecord: ModelRecord = config.model_list[i];
    const model_id = modelRecord.model_id;
    // 2. Load the wasm
    const wasmUrl = modelRecord.model_lib;
    const wasmSource = await (await fetch(wasmUrl)).arrayBuffer();
    report += `${model_id}: \n`;
    // 3. Initialize tvmjs instance and virtual machine using the wasm
    const tvm = await tvmjs.instantiate(
      new Uint8Array(wasmSource),
      tvmjs.createPolyfillWASI(),
      log.info,
    );
    const gpuDetectOutput = await tvmjs.detectGPUDevice();
    if (gpuDetectOutput == undefined) {
      throw Error("Cannot find WebGPU in the environment");
    }
    tvm.initWebGPU(gpuDetectOutput.device);
    tvm.beginScope();
    const vm = tvm.detachFromCurrentScope(
      tvm.createVirtualMachine(tvm.webgpu()),
    );
    // 4. Get metadata from the vm
    let fgetMetadata: any;
    try {
      fgetMetadata = vm.getFunction("_metadata");
    } catch (err) {
      log.error(
        "The wasm needs to have function `_metadata` to inspect vram requirement.",
        err,
      );
    }
    const ret_value = fgetMetadata();
    const metadataStr = tvm.detachFromCurrentScope(ret_value).toString();
    const metadata = JSON.parse(metadataStr);
    // 5. Parse the vram requirement
    // 5.1. Get bytes for loading params
    let paramBytes = 0;
    metadata.params.forEach((param: any) => {
      if (Math.min(...param.shape) > 0) {
        // Possible to have shape -1 signifying a dynamic shape -- we disregard them
        const dtypeBytes = dtypeBytesMap.get(param.dtype);
        if (dtypeBytes === undefined) {
          throw Error(
            "Cannot find size of " +
              param.dtype +
              ", add it to `dtypeBytesMap`.",
          );
        }
        const numParams = param.shape.reduce((a: number, b: number) => a * b);
        paramBytes += numParams * dtypeBytes;
      } else {
        log.info(
          `${model_id}'s ${param.name} has dynamic shape; excluded from vRAM calculation.`,
        );
      }
    });
    // 5.2. Get maximum bytes needed for temporary buffer across all functions
    let maxTempFuncBytes = 0;
    Object.entries(metadata.memory_usage).forEach(([funcName, funcBytes]) => {
      if (typeof funcBytes !== "number") {
        throw Error("`memory_usage` expects entry `funcName: funcBytes`.");
      }
      maxTempFuncBytes = Math.max(maxTempFuncBytes, funcBytes);
    });
    // 5.3. Get kv cache bytes
    const kv_cache_bytes: number = metadata.kv_cache_bytes;
    // 5.4. Get total vRAM needed
    const totalBytes = paramBytes + maxTempFuncBytes + kv_cache_bytes;
    // 6. Report vRAM Requirement
    report +=
      `totalBytes: ${(totalBytes / 1024 / 1024).toFixed(2)} MB\n` +
      `paramBytes: ${(paramBytes / 1024 / 1024).toFixed(2)} MB\n` +
      `maxTempFuncBytes: ${(maxTempFuncBytes / 1024 / 1024).toFixed(2)} MB\n` +
      `kv_cache_bytes: ${(kv_cache_bytes / 1024 / 1024).toFixed(2)} MB\n\n`;
    // 7. Dispose everything
    tvm.endScope();
    vm.dispose();
    tvm.dispose();
  }
  setLabel("report-label", report);
}

main();
