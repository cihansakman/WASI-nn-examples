# Example: LLM Inference from Wasm using Wasi-nn

This example demonstrates the capability to call LLM inferences from WebAssembly (Wasm) using the `wasi-nn` proposal. The example uses OpenVINO as the backend, requiring the [OpenVINO backend](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html) to be installed first.

## Setup OpenVINO 

```bash
source /opt/intel/openvino_2024/setupvars.sh
```


## Wasmtime with Wasi-nn and OpenVINO

Clone [Wasmtime](https://github.com/bytecodealliance/wasmtime.git) into the root of this repository. Alternatively, use an existing Wasmtime setup. Then, run the following command to build the Wasmtime CLI with the necessary features:

```bash
# Build Wasmtime with component-model and WASI-NN with PyTorch runtime support
cargo build --features component-model,wasi-nn,wasmtime-wasi-nn/onnx
```

## LLM Model from Hugging Face

This example uses the [ov-gpt2-fp32-no-cache model](https://huggingface.co/vuiseng9/ov-gpt2-fp32-no-cache/tree/main) from Hugging Face. To download the required model files:

```bash
wget https://huggingface.co/vuiseng9/ov-gpt2-fp32-no-cache/resolve/main/openvino_model.bin -P fixture/
wget https://huggingface.co/vuiseng9/ov-gpt2-fp32-no-cache/resolve/main/openvino_model.xml -P fixture/
```

## Build the Component

1. Build this example using:
   ```bash
   cargo component build
   ```

2. Run the generated Wasm file with Wasmtime, passing the appropriate arguments:
   ```bash
   /path/to/wasmtime -S nn --dir fixture/::fixture target/wasm32-wasip1/debug/llm-component-openvino.wasm
   ```
3. Give the prompt in the fixture/prompt file.
