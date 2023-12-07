export default {
	"model_list": [
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f32_1/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f32_1"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f32_1/resolve/main/",
			"local_id": "Llama-2-13b-chat-hf-q4f32_1"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-13b-chat-hf-q4f16_1/resolve/main/",
			"local_id": "Llama-2-13b-chat-hf-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-70b-chat-hf-q4f16_1/resolve/main/",
			"local_id": "Llama-2-70b-chat-hf-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_1/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardCoder-15B-V1.0-q4f16_1/resolve/main/",
			"local_id": "WizardCoder-15B-V1.0-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardCoder-15B-V1.0-q4f32_1/resolve/main/",
			"local_id": "WizardCoder-15B-V1.0-q4f32_1"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardMath-7B-V1.0-q4f16_1/resolve/main/",
			"local_id": "WizardMath-7B-V1.0-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardMath-7B-V1.0-q4f32_1/resolve/main/",
			"local_id": "WizardMath-7B-V1.0-q4f32_1"
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardMath-13B-V1.0-q4f16_1/resolve/main/",
			"local_id": "WizardMath-13B-V1.0-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-WizardMath-70B-V1.0-q4f16_1/resolve/main/",
			"local_id": "WizardMath-70B-V1.0-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Mistral-7B-Instruct-v0.1-q4f16_1/resolve/main/",
			"local_id": "Mistral-7B-Instruct-v0.1-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Mistral-7B-Instruct-v0.1-q4f32_1/resolve/main/",
			"local_id": "Mistral-7B-Instruct-v0.1-q4f32_1",
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-OpenHermes-2.5-Mistral-7B-q4f16_1/resolve/main/",
			"local_id": "OpenHermes-2.5-Mistral-7B-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-OpenHermes-2.5-Mistral-7B-q4f32_1/resolve/main/",
			"local_id": "OpenHermes-2.5-Mistral-7B-q4f32_1",
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-NeuralHermes-2.5-Mistral-7B-q4f16_1/resolve/main/",
			"local_id": "NeuralHermes-2.5-Mistral-7B-q4f16_1",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-NeuralHermes-2.5-Mistral-7B-q4f32_1/resolve/main/",
			"local_id": "NeuralHermes-2.5-Mistral-7B-q4f32_1",
		},
		// Models below fit for 128MB buffer limit (e.g. webgpu on Android)
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-Llama-2-7b-chat-hf-q4f16_1-1k/resolve/main/",
			"local_id": "Llama-2-7b-chat-hf-q4f16_1-1k",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k",
			"required_features": ["shader-f16"],
		},
		{
			"model_url": "https://huggingface.co/mlc-ai/mlc-chat-RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k/resolve/main/",
			"local_id": "RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k"
		},
	],
	"model_lib_map": {
		"Llama-2-7b-chat-hf-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm",
		"Llama-2-13b-chat-hf-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-13b-chat-hf-q4f32_1-webgpu.wasm",
		"Llama-2-7b-chat-hf-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f16_1-webgpu.wasm",
		"Llama-2-13b-chat-hf-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-13b-chat-hf-q4f16_1-webgpu.wasm",
		"Llama-2-70b-chat-hf-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-70b-chat-hf-q4f16_1-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-webgpu.wasm",
		"WizardCoder-15B-V1.0-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/WizardCoder-15B-V1.0-q4f16_1-webgpu.wasm",
		"WizardCoder-15B-V1.0-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/WizardCoder-15B-V1.0-q4f32_1-webgpu.wasm",
		"WizardMath-7B-V1.0-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f16_1-webgpu.wasm",
		"WizardMath-7B-V1.0-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f32_1-webgpu.wasm",
		"WizardMath-13B-V1.0-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-13b-chat-hf-q4f16_1-webgpu.wasm",
		"WizardMath-70B-V1.0-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-70b-chat-hf-q4f16_1-webgpu.wasm",
		"Mistral-7B-Instruct-v0.1-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f16_1-sw4k_cs1k-webgpu.wasm",
		"Mistral-7B-Instruct-v0.1-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f32_1-sw4k_cs1k-webgpu.wasm",
		"OpenHermes-2.5-Mistral-7B-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f32_1-sw4k_cs1k-webgpu.wasm",
		"OpenHermes-2.5-Mistral-7B-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f32_1-sw4k_cs1k-webgpu.wasm",
		"NeuralHermes-2.5-Mistral-7B-q4f16_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f32_1-sw4k_cs1k-webgpu.wasm",
		"NeuralHermes-2.5-Mistral-7B-q4f32_1": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Mistral-7B-Instruct-v0.1-q4f32_1-sw4k_cs1k-webgpu.wasm",
		// Models below fit for 128MB buffer limit (e.g. webgpu on Android)
		"Llama-2-7b-chat-hf-q4f16_1-1k": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/Llama-2-7b-chat-hf-q4f16_1-1k-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f16_1-1k-webgpu.wasm",
		"RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/RedPajama-INCITE-Chat-3B-v1-q4f32_1-1k-webgpu.wasm",
	},
	"use_web_worker": true
}
