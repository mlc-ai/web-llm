export default {
	"model_list": [
		{
			"model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-hf-q4f32_1-MLC/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f32_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-ctx4k_cs1k-MLC-webgpu.wasm",
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-hf-q4f16_1-MLC/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f16_1-ctx4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/Llama-2-13b-hf-q4f16_1-MLC/resolve/main/",
			"local_id": "Llama-2-13b-chat-hf-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-13b-chat-hf-q4f16_1-ctx4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/Llama-2-70b-hf-q4f16_1-MLC/resolve/main/",
			"local_id": "Llama-2-70b-chat-hf-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-70b-chat-hf-q4f16_1-ctx4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx4k_cs1k-MLC-webgpu.wasm",
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC/resolve/main/",
			"local_id": "WizardMath-7B-V1.1-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC/resolve/main/",
			"local_id": "Mistral-7B-Instruct-v0.2-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC/resolve/main/",
			"local_id": "OpenHermes-2.5-Mistral-7B-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC/resolve/main/",
			"local_id": "NeuralHermes-2.5-Mistral-7B-q4f16_1",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.2-q4f16_1-sw4k_cs1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		// Models below fit for 128MB buffer limit (e.g. webgpu on Android)
		{
			"model_url": "https://huggingface.co/mlc-ai/Llama-2-7b-hf-q4f16_1-MLC/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f16_1-1k",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f16_1-ctx1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx1k-MLC-webgpu.wasm",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k",
			"model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx1k-MLC-webgpu.wasm",
		},
	],
	"use_web_worker": true
}
