export default {
	"model_list": [
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_0/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_0"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0/resolve/main/",
			"local_id": "vicuna-v1-7b-q4f32_0"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_0/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_0",
			"required_features": ["shader-f16"],
		}
	],
	"model_lib_map": {
		"vicuna-v1-7b-q4f32_0": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/vicuna-v1-7b-q4f32_0-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f32_0": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_0-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f16_0": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f16_0-webgpu.wasm"
	},
	"use_web_worker": true
}
