
# Onnx Backend Classification Component Example

## Setup OpenVINO Backend
This example uses openvino_2024.1.0.15008 using [archive](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)

This example demonstrates how to use the `wasi-nn` crate to run a classification using the
[ONNX Runtime](https://onnxruntime.ai/) backend from a WebAssembly component.

## Some ONNX Models
[ONNX Model ZOO](https://github.com/onnx/models/tree/main/validated/)
## Build
In this directory, run the following command to build the WebAssembly component:
```shell
cargo component build
```


## Wasmtime with Wasi-nn
Clone [Wasmtime](https://github.com/bytecodealliance/wasmtime.git) to root of this repo. In the Wasmtime root directory or your existed wasmtime setup, run the following command to build the Wasmtime CLI and run the WebAssembly component:
```shell
# build wasmtime with component-model and WASI-NN with ONNX runtime support
cargo build --features component-model,wasi-nn,wasmtime-wasi-nn/onnx

# run the component with wasmtime
/path/to/wasmtime run -Snn --dir fixture/::fixture target/wasm32-wasip1/debug/classification-component-onnx.wasm
```

You should get the following output:
```txt
Read ONNX model, size in bytes: 4956208
Loaded graph into wasi-nn
Created wasi-nn execution context.
Read ONNX Labels, # of labels: 1000
Set input tensor
Executed graph inference
Getting inferencing output
Retrieved output data with length: 4000
Index: n01667114 mud turtle - Probability: 0.6510642
Index: n01667778 terrapin - Probability: 0.28487134
Index: n01669191 box turtle, box tortoise - Probability: 0.05587133
```
