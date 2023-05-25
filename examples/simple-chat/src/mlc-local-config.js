// config used when serving from local mlc-llm/dist
// use web-llm/script/serve_mlc_llm_dist.sh to start the artifact server
export default {
	"model_list": [
		{
			"model_url": "http://localhost:8000/RedPajama-INCITE-Chat-3B-v1-q4f32_0/params/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_0"
		},
		{
			"model_url": "http://localhost:8000/vicuna-v1-7b-q4f32_0/params/",
      		"local_id": "vicuna-v1-7b-q4f32_0"
		}
	],
	"model_lib_map": {
		"vicuna-v1-7b-q4f32_0": "http://localhost:8000/vicuna-v1-7b-q4f32_0/vicuna-v1-7b-q4f32_0-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f32_0": "http://localhost:8000/RedPajama-INCITE-Chat-3B-v1-q4f32_0/RedPajama-INCITE-Chat-3B-v1-q4f32_0-webgpu.wasm"
	}
}
