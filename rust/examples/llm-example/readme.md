# Example: LLM Inference from Wasm using Wasi-nn

This example demonstrates the capability to call LLM inferences from WebAssembly (Wasm) using the `wasi-nn` proposal. The example uses PyTorch as the backend, requiring the C++ LibTorch library to be installed first.

## Setup C++ PyTorch Library

```bash
sudo apt install nvidia-cuda-toolkit
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu115.zip -O libtorch.zip
unzip libtorch.zip
export LIBTORCH=/path/to/libtorch
# LIBTORCH_INCLUDE must contain the `include` directory.
export LIBTORCH_INCLUDE=/path/to/libtorch/include
# LIBTORCH_LIB must contain the `lib` directory.
export LIBTORCH_LIB=/path/to/libtorch/lib
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

You can download the [C++ PyTorch library (LibTorch) version v2.4.0 here](https://pytorch.org/get-started/locally/).

## Wasmtime with Wasi-nn and PyTorch

Clone [Wasmtime](https://github.com/bytecodealliance/wasmtime.git) into the root of this repository. Alternatively, use an existing Wasmtime setup. Then, run the following command to build the Wasmtime CLI with the necessary features:

```bash
# Build Wasmtime with component-model and WASI-NN with PyTorch runtime support
cargo build --features component-model,wasi-nn,wasmtime-wasi-nn/pytorch
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
   /path/to/wasmtime -S nn --dir fixture/::fixture target/wasm32-wasip1/debug/llm-component-pytorch.wasm
   ```

3. The output may currently appear meaningless, as this example lacks advanced tokenization to fully utilize the model's capabilities. Further refinement is needed for effective model interaction.
