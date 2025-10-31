import appConfig from "./gh-config.js";
import { cleanModelUrl, findModelRecord } from "../../../src/support";

interface AppConfig {
  model_list: Array<any>;
}

async function getParamBytes(modelId: string): Promise<number> {
  const config: AppConfig = appConfig;
  const rec = findModelRecord(modelId, config);
  const base = cleanModelUrl(rec.model);
  const url = `${base}ndarray-cache.json`;
  const meta = (await (await fetch(url)).json()) as {
    metadata: { ParamBytes: number; ParamSize: number; BitsPerParam: number };
  };
  return meta.metadata.ParamBytes;
}

async function main() {
  const config: AppConfig = appConfig;
  for (const rec of config.model_list) {
    const bytes = await getParamBytes(rec.model_id);
    console.log(
      `${rec.model_id.padEnd(30)} : ` + `${(bytes / 1_048_576).toFixed(2)} MB`,
    );
  }
}

main().catch(console.error);
