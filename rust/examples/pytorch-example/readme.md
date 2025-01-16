This example project demonstrates using the `wasi-nn` API to perform PyTorch based inference. It consists of Rust code that is built using the `wasm32-wasip1` target.

## Setup C++ PyTorch Library

```shell
sudo apt install nvidia-cuda-toolkit
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu115.zip -O libtorch.zip
unzip libtorch.zip
export LIBTORCH=/path/to/libtorch
# LIBTORCH_INCLUDE must contain `include` directory.
export LIBTORCH_INCLUDE=/path/to/libtorch/
# LIBTORCH_LIB must contain `lib` directory.
export LIBTORCH_LIB=/path/to/libtorch/
```

[C++ PyTorch library (libtorch) in version v2.4.0](https://pytorch.org/get-started/locally/)


## Wasmtime with Wasi-nn/pytorch
Clone [Wasmtime](https://github.com/bytecodealliance/wasmtime.git) to root of this repo. In the Wasmtime root directory or your existed wasmtime setup, run the following command to build the Wasmtime CLI and run the WebAssembly component:
```shell
# build wasmtime with component-model and WASI-NN with Pytorch runtime support
cargo build --features component-model,wasi-nn,wasmtime-wasi-nn/pytorch

```


To run this example: 
1. Ensure you set appropriate Libtorch enviornment variables according to [tch-rs instructions]( https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#libtorch-manual-install). 
    - Requires the C++ PyTorch library (libtorch) in version *v2.4.0* to be available on
your system. 
    - `export LIBTORCH=/path/to/libtorch`
2. Build Wasmtime  with `wasmtime-wasi-nn/pytorch` feature.
3. Navigate to this example directory `crates/wasi-nn/examples/classification-example-pytorch`.
4. Download `squeezenet1_1.pt` model 
```
curl https://github.com/rahulchaphalkar/libtorch-models/releases/download/v0.1/squeezenet1_1.pt --output fixture/model.pt -L
```
4. Build this example `cargo build --target=wasm32-wasip1`.
5. Run the generated wasm file with wasmtime after mapping the directory containing squeezenet1.1 `model.pt` and sample image `kitten.png`
    ```
    /path/to/wasmtime -S nn --dir fixture/::fixture target/wasm32-wasip1/debug/wasi-nn-example-pytorch.wasm
    ```
6. Check that result `281` has highest probability, which corresponds to `tabby cat`.

