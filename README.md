First put your model into the artifact path (default to be dist/).
```
ln -s your_model_path dist/vicuna-7b/models
```
Then build a model into a TVM executable

```
python3 build.py --model vicuna-7b
```

If you just want to run a single forward pass, run deploy.py
```
python3 evaluate.py --model vicuna-7b
```

To run a chat bot, run generation.py

```
python3 chat.py --model vicuna-7b [--max-gen-len 128 (default to 32)]
```

your_model_path can be a local directory on your machine, or a huggingface model name. To run vicuna, specify the model path as the directory which contains the model weight and tokenizer.
