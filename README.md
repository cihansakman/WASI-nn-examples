# <code>wasi-nn</code>

This repository is a fork of the [wasi-nn project](https://github.com/bytecodealliance/wasi-nn), with added examples for running inference using Wasmtime and integrating additional models. These enhancements include:

1. An example for PyTorch-based image classification (adapted from [Wasmtime's classification example](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn/examples/classification-example-pytorch)).
2. A new example for large language model (LLM) inference using the [ov-gpt2-fp32-no-cache model](https://huggingface.co/vuiseng9/ov-gpt2-fp32-no-cache/tree/main) on Wasmtime.

All examples can be found in the `rust/examples/` directory.

## Introduction

`wasi-nn` provides high-level bindings for developing machine learning applications in WebAssembly. These bindings enable:

- Writing ML applications in Rust or AssemblyScript.
- Compiling applications to WebAssembly.
- Running them in WebAssembly runtimes that support the [wasi-nn specification](https://github.com/WebAssembly/wasi-nn), such as [Wasmtime](https://wasmtime.dev) or [WasmEdge](https://github.com/WasmEdge/WasmEdge).

**Note:** The original wasi-nn bindings are experimental and subject to upstream changes in the [wasi-nn specification](https://github.com/WebAssembly/wasi-nn).

## Enhancements in This Fork

### Added Examples

- **PyTorch Image Classification**:
  - Adapted from the [Wasmtime example](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn/examples/classification-example-pytorch).
  - Demonstrates how to perform image classification using a PyTorch model.

- **LLM Inference**:
  - Developed specifically for this fork.
  - Uses the [ov-gpt2-fp32-no-cache model](https://huggingface.co/vuiseng9/ov-gpt2-fp32-no-cache/tree/main).
  - Explains how to run large language models for inference in a WebAssembly environment.

## Usage

### Rust

1. Add `wasi-nn` as a dependency in your `Cargo.toml` file:
   ```toml
   [dependencies]
   wasi-nn = "0.6.0"
   ```
2. Compile your application to WebAssembly.

3. Run it with Wasmtime, passing the flag `--wasi-modules=experimental-wasi-nn` to enable wasi-nn.



### Running with WasmEdge

To run the examples in WasmEdge, install the [wasi-nn plugin](https://wasmedge.org/docs/category/ai-inference) and follow the [WasmEdge wasi-nn examples](https://github.com/second-state/WasmEdge-WASINN-examples).

## License

This project is licensed under the Apache 2.0 license. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository is based on the [wasi-nn project](https://github.com/bytecodealliance/wasi-nn) by the [Bytecode Alliance](https://bytecodealliance.org/). Examples have been adapted or developed to extend the project's functionality with additional ML use cases.
