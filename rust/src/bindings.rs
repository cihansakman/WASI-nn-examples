#[allow(dead_code)]
pub mod wasi {
    #[allow(dead_code)]
    pub mod nn {
        #[allow(dead_code, clippy::all)]
        pub mod tensor {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            /// The dimensions of a tensor.
            ///
            /// The array length matches the tensor rank and each element in the array describes the size of
            /// each dimension
            pub type TensorDimensions = _rt::Vec<u32>;
            /// The type of the elements in a tensor.
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum TensorType {
                Fp16,
                Fp32,
                Fp64,
                Bf16,
                U8,
                I32,
                I64,
            }
            impl ::core::fmt::Debug for TensorType {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        TensorType::Fp16 => f.debug_tuple("TensorType::Fp16").finish(),
                        TensorType::Fp32 => f.debug_tuple("TensorType::Fp32").finish(),
                        TensorType::Fp64 => f.debug_tuple("TensorType::Fp64").finish(),
                        TensorType::Bf16 => f.debug_tuple("TensorType::Bf16").finish(),
                        TensorType::U8 => f.debug_tuple("TensorType::U8").finish(),
                        TensorType::I32 => f.debug_tuple("TensorType::I32").finish(),
                        TensorType::I64 => f.debug_tuple("TensorType::I64").finish(),
                    }
                }
            }
            impl TensorType {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> TensorType {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => TensorType::Fp16,
                        1 => TensorType::Fp32,
                        2 => TensorType::Fp64,
                        3 => TensorType::Bf16,
                        4 => TensorType::U8,
                        5 => TensorType::I32,
                        6 => TensorType::I64,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            /// The tensor data.
            ///
            /// Initially conceived as a sparse representation, each empty cell would be filled with zeros
            /// and the array length must match the product of all of the dimensions and the number of bytes
            /// in the type (e.g., a 2x2 tensor with 4-byte f32 elements would have a data array of length
            /// 16). Naturally, this representation requires some knowledge of how to lay out data in
            /// memory--e.g., using row-major ordering--and could perhaps be improved.
            pub type TensorData = _rt::Vec<u8>;
            #[derive(Debug)]
            #[repr(transparent)]
            pub struct Tensor {
                handle: _rt::Resource<Tensor>,
            }
            impl Tensor {
                #[doc(hidden)]
                pub unsafe fn from_handle(handle: u32) -> Self {
                    Self {
                        handle: _rt::Resource::from_handle(handle),
                    }
                }
                #[doc(hidden)]
                pub fn take_handle(&self) -> u32 {
                    _rt::Resource::take_handle(&self.handle)
                }
                #[doc(hidden)]
                pub fn handle(&self) -> u32 {
                    _rt::Resource::handle(&self.handle)
                }
            }
            unsafe impl _rt::WasmResource for Tensor {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(
                            wasm_import_module = "wasi:nn/tensor@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[resource-drop]tensor"]
                            fn drop(_: u32);
                        }
                        drop(_handle);
                    }
                }
            }
            impl Tensor {
                #[allow(unused_unsafe, clippy::all)]
                pub fn new(
                    dimensions: &TensorDimensions,
                    ty: TensorType,
                    data: &TensorData,
                ) -> Self {
                    unsafe {
                        let vec0 = dimensions;
                        let ptr0 = vec0.as_ptr().cast::<u8>();
                        let len0 = vec0.len();
                        let vec1 = data;
                        let ptr1 = vec1.as_ptr().cast::<u8>();
                        let len1 = vec1.len();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/tensor@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[constructor]tensor"]
                            fn wit_import(
                                _: *mut u8,
                                _: usize,
                                _: i32,
                                _: *mut u8,
                                _: usize,
                            ) -> i32;
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(
                            _: *mut u8,
                            _: usize,
                            _: i32,
                            _: *mut u8,
                            _: usize,
                        ) -> i32 {
                            unreachable!()
                        }
                        let ret = wit_import(
                            ptr0.cast_mut(),
                            len0,
                            ty.clone() as i32,
                            ptr1.cast_mut(),
                            len1,
                        );
                        Tensor::from_handle(ret as u32)
                    }
                }
            }
            impl Tensor {
                #[allow(unused_unsafe, clippy::all)]
                /// Describe the size of the tensor (e.g., 2x2x2x2 -> [2, 2, 2, 2]). To represent a tensor
                /// containing a single value, use `[1]` for the tensor dimensions.
                pub fn dimensions(&self) -> TensorDimensions {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/tensor@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]tensor.dimensions"]
                            fn wit_import(_: i32, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<*mut u8>();
                        let l2 = *ptr0.add(4).cast::<usize>();
                        let len3 = l2;
                        _rt::Vec::from_raw_parts(l1.cast(), len3, len3)
                    }
                }
            }
            impl Tensor {
                #[allow(unused_unsafe, clippy::all)]
                /// Describe the type of element in the tensor (e.g., `f32`).
                pub fn ty(&self) -> TensorType {
                    unsafe {
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/tensor@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]tensor.ty"]
                            fn wit_import(_: i32) -> i32;
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32) -> i32 {
                            unreachable!()
                        }
                        let ret = wit_import((self).handle() as i32);
                        TensorType::_lift(ret as u8)
                    }
                }
            }
            impl Tensor {
                #[allow(unused_unsafe, clippy::all)]
                /// Return the tensor data.
                pub fn data(&self) -> TensorData {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/tensor@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]tensor.data"]
                            fn wit_import(_: i32, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<*mut u8>();
                        let l2 = *ptr0.add(4).cast::<usize>();
                        let len3 = l2;
                        _rt::Vec::from_raw_parts(l1.cast(), len3, len3)
                    }
                }
            }
        }
        #[allow(dead_code, clippy::all)]
        pub mod errors {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum ErrorCode {
                /// Caller module passed an invalid argument.
                InvalidArgument,
                /// Invalid encoding.
                InvalidEncoding,
                /// The operation timed out.
                Timeout,
                /// Runtime Error.
                RuntimeError,
                /// Unsupported operation.
                UnsupportedOperation,
                /// Graph is too large.
                TooLarge,
                /// Graph not found.
                NotFound,
                /// The operation is insecure or has insufficient privilege to be performed.
                /// e.g., cannot access a hardware feature requested
                Security,
                /// The operation failed for an unspecified reason.
                Unknown,
            }
            impl ::core::fmt::Debug for ErrorCode {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        ErrorCode::InvalidArgument => {
                            f.debug_tuple("ErrorCode::InvalidArgument").finish()
                        }
                        ErrorCode::InvalidEncoding => {
                            f.debug_tuple("ErrorCode::InvalidEncoding").finish()
                        }
                        ErrorCode::Timeout => {
                            f.debug_tuple("ErrorCode::Timeout").finish()
                        }
                        ErrorCode::RuntimeError => {
                            f.debug_tuple("ErrorCode::RuntimeError").finish()
                        }
                        ErrorCode::UnsupportedOperation => {
                            f.debug_tuple("ErrorCode::UnsupportedOperation").finish()
                        }
                        ErrorCode::TooLarge => {
                            f.debug_tuple("ErrorCode::TooLarge").finish()
                        }
                        ErrorCode::NotFound => {
                            f.debug_tuple("ErrorCode::NotFound").finish()
                        }
                        ErrorCode::Security => {
                            f.debug_tuple("ErrorCode::Security").finish()
                        }
                        ErrorCode::Unknown => {
                            f.debug_tuple("ErrorCode::Unknown").finish()
                        }
                    }
                }
            }
            impl ErrorCode {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> ErrorCode {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => ErrorCode::InvalidArgument,
                        1 => ErrorCode::InvalidEncoding,
                        2 => ErrorCode::Timeout,
                        3 => ErrorCode::RuntimeError,
                        4 => ErrorCode::UnsupportedOperation,
                        5 => ErrorCode::TooLarge,
                        6 => ErrorCode::NotFound,
                        7 => ErrorCode::Security,
                        8 => ErrorCode::Unknown,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            #[derive(Debug)]
            #[repr(transparent)]
            pub struct Error {
                handle: _rt::Resource<Error>,
            }
            impl Error {
                #[doc(hidden)]
                pub unsafe fn from_handle(handle: u32) -> Self {
                    Self {
                        handle: _rt::Resource::from_handle(handle),
                    }
                }
                #[doc(hidden)]
                pub fn take_handle(&self) -> u32 {
                    _rt::Resource::take_handle(&self.handle)
                }
                #[doc(hidden)]
                pub fn handle(&self) -> u32 {
                    _rt::Resource::handle(&self.handle)
                }
            }
            unsafe impl _rt::WasmResource for Error {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(
                            wasm_import_module = "wasi:nn/errors@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[resource-drop]error"]
                            fn drop(_: u32);
                        }
                        drop(_handle);
                    }
                }
            }
            impl Error {
                #[allow(unused_unsafe, clippy::all)]
                /// Return the error code.
                pub fn code(&self) -> ErrorCode {
                    unsafe {
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/errors@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]error.code"]
                            fn wit_import(_: i32) -> i32;
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32) -> i32 {
                            unreachable!()
                        }
                        let ret = wit_import((self).handle() as i32);
                        ErrorCode::_lift(ret as u8)
                    }
                }
            }
            impl Error {
                #[allow(unused_unsafe, clippy::all)]
                /// Errors can propagated with backend specific status through a string value.
                pub fn data(&self) -> _rt::String {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/errors@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]error.data"]
                            fn wit_import(_: i32, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = *ptr0.add(0).cast::<*mut u8>();
                        let l2 = *ptr0.add(4).cast::<usize>();
                        let len3 = l2;
                        let bytes3 = _rt::Vec::from_raw_parts(l1.cast(), len3, len3);
                        _rt::string_lift(bytes3)
                    }
                }
            }
        }
        #[allow(dead_code, clippy::all)]
        pub mod inference {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            pub type Error = super::super::super::wasi::nn::errors::Error;
            pub type Tensor = super::super::super::wasi::nn::tensor::Tensor;
            /// Bind a `graph` to the input and output tensors for an inference.
            ///
            /// TODO: this may no longer be necessary in WIT
            /// (https://github.com/WebAssembly/wasi-nn/issues/43)
            #[derive(Debug)]
            #[repr(transparent)]
            pub struct GraphExecutionContext {
                handle: _rt::Resource<GraphExecutionContext>,
            }
            impl GraphExecutionContext {
                #[doc(hidden)]
                pub unsafe fn from_handle(handle: u32) -> Self {
                    Self {
                        handle: _rt::Resource::from_handle(handle),
                    }
                }
                #[doc(hidden)]
                pub fn take_handle(&self) -> u32 {
                    _rt::Resource::take_handle(&self.handle)
                }
                #[doc(hidden)]
                pub fn handle(&self) -> u32 {
                    _rt::Resource::handle(&self.handle)
                }
            }
            unsafe impl _rt::WasmResource for GraphExecutionContext {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(
                            wasm_import_module = "wasi:nn/inference@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[resource-drop]graph-execution-context"]
                            fn drop(_: u32);
                        }
                        drop(_handle);
                    }
                }
            }
            impl GraphExecutionContext {
                #[allow(unused_unsafe, clippy::all)]
                /// Define the inputs to use for inference.
                pub fn set_input(
                    &self,
                    name: &str,
                    tensor: Tensor,
                ) -> Result<(), Error> {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let vec0 = name;
                        let ptr0 = vec0.as_ptr().cast::<u8>();
                        let len0 = vec0.len();
                        let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/inference@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]graph-execution-context.set-input"]
                            fn wit_import(
                                _: i32,
                                _: *mut u8,
                                _: usize,
                                _: i32,
                                _: *mut u8,
                            );
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8, _: usize, _: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import(
                            (self).handle() as i32,
                            ptr0.cast_mut(),
                            len0,
                            (&tensor).take_handle() as i32,
                            ptr1,
                        );
                        let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                        match l2 {
                            0 => {
                                let e = ();
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l3 = *ptr1.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::errors::Error::from_handle(
                                        l3 as u32,
                                    )
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
            impl GraphExecutionContext {
                #[allow(unused_unsafe, clippy::all)]
                /// Compute the inference on the given inputs.
                ///
                /// Note the expected sequence of calls: `set-input`, `compute`, `get-output`. TODO: this
                /// expectation could be removed as a part of
                /// https://github.com/WebAssembly/wasi-nn/issues/43.
                pub fn compute(&self) -> Result<(), Error> {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/inference@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]graph-execution-context.compute"]
                            fn wit_import(_: i32, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = i32::from(*ptr0.add(0).cast::<u8>());
                        match l1 {
                            0 => {
                                let e = ();
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l2 = *ptr0.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::errors::Error::from_handle(
                                        l2 as u32,
                                    )
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
            impl GraphExecutionContext {
                #[allow(unused_unsafe, clippy::all)]
                /// Extract the outputs after inference.
                pub fn get_output(&self, name: &str) -> Result<Tensor, Error> {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let vec0 = name;
                        let ptr0 = vec0.as_ptr().cast::<u8>();
                        let len0 = vec0.len();
                        let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(
                            wasm_import_module = "wasi:nn/inference@0.2.0-rc-2024-08-19"
                        )]
                        extern "C" {
                            #[link_name = "[method]graph-execution-context.get-output"]
                            fn wit_import(_: i32, _: *mut u8, _: usize, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8, _: usize, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0.cast_mut(), len0, ptr1);
                        let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                        match l2 {
                            0 => {
                                let e = {
                                    let l3 = *ptr1.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::tensor::Tensor::from_handle(
                                        l3 as u32,
                                    )
                                };
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l4 = *ptr1.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::errors::Error::from_handle(
                                        l4 as u32,
                                    )
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
        }
        #[allow(dead_code, clippy::all)]
        pub mod graph {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            pub type Error = super::super::super::wasi::nn::errors::Error;
            pub type GraphExecutionContext = super::super::super::wasi::nn::inference::GraphExecutionContext;
            /// An execution graph for performing inference (i.e., a model).
            #[derive(Debug)]
            #[repr(transparent)]
            pub struct Graph {
                handle: _rt::Resource<Graph>,
            }
            impl Graph {
                #[doc(hidden)]
                pub unsafe fn from_handle(handle: u32) -> Self {
                    Self {
                        handle: _rt::Resource::from_handle(handle),
                    }
                }
                #[doc(hidden)]
                pub fn take_handle(&self) -> u32 {
                    _rt::Resource::take_handle(&self.handle)
                }
                #[doc(hidden)]
                pub fn handle(&self) -> u32 {
                    _rt::Resource::handle(&self.handle)
                }
            }
            unsafe impl _rt::WasmResource for Graph {
                #[inline]
                unsafe fn drop(_handle: u32) {
                    #[cfg(not(target_arch = "wasm32"))]
                    unreachable!();
                    #[cfg(target_arch = "wasm32")]
                    {
                        #[link(wasm_import_module = "wasi:nn/graph@0.2.0-rc-2024-08-19")]
                        extern "C" {
                            #[link_name = "[resource-drop]graph"]
                            fn drop(_: u32);
                        }
                        drop(_handle);
                    }
                }
            }
            /// Describes the encoding of the graph. This allows the API to be implemented by various
            /// backends that encode (i.e., serialize) their graph IR with different formats.
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum GraphEncoding {
                Openvino,
                Onnx,
                Tensorflow,
                Pytorch,
                Tensorflowlite,
                Ggml,
                Autodetect,
            }
            impl ::core::fmt::Debug for GraphEncoding {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        GraphEncoding::Openvino => {
                            f.debug_tuple("GraphEncoding::Openvino").finish()
                        }
                        GraphEncoding::Onnx => {
                            f.debug_tuple("GraphEncoding::Onnx").finish()
                        }
                        GraphEncoding::Tensorflow => {
                            f.debug_tuple("GraphEncoding::Tensorflow").finish()
                        }
                        GraphEncoding::Pytorch => {
                            f.debug_tuple("GraphEncoding::Pytorch").finish()
                        }
                        GraphEncoding::Tensorflowlite => {
                            f.debug_tuple("GraphEncoding::Tensorflowlite").finish()
                        }
                        GraphEncoding::Ggml => {
                            f.debug_tuple("GraphEncoding::Ggml").finish()
                        }
                        GraphEncoding::Autodetect => {
                            f.debug_tuple("GraphEncoding::Autodetect").finish()
                        }
                    }
                }
            }
            impl GraphEncoding {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> GraphEncoding {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => GraphEncoding::Openvino,
                        1 => GraphEncoding::Onnx,
                        2 => GraphEncoding::Tensorflow,
                        3 => GraphEncoding::Pytorch,
                        4 => GraphEncoding::Tensorflowlite,
                        5 => GraphEncoding::Ggml,
                        6 => GraphEncoding::Autodetect,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            /// Define where the graph should be executed.
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum ExecutionTarget {
                Cpu,
                Gpu,
                Tpu,
            }
            impl ::core::fmt::Debug for ExecutionTarget {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        ExecutionTarget::Cpu => {
                            f.debug_tuple("ExecutionTarget::Cpu").finish()
                        }
                        ExecutionTarget::Gpu => {
                            f.debug_tuple("ExecutionTarget::Gpu").finish()
                        }
                        ExecutionTarget::Tpu => {
                            f.debug_tuple("ExecutionTarget::Tpu").finish()
                        }
                    }
                }
            }
            impl ExecutionTarget {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> ExecutionTarget {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => ExecutionTarget::Cpu,
                        1 => ExecutionTarget::Gpu,
                        2 => ExecutionTarget::Tpu,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            /// The graph initialization data.
            ///
            /// This gets bundled up into an array of buffers because implementing backends may encode their
            /// graph IR in parts (e.g., OpenVINO stores its IR and weights separately).
            pub type GraphBuilder = _rt::Vec<u8>;
            impl Graph {
                #[allow(unused_unsafe, clippy::all)]
                pub fn init_execution_context(
                    &self,
                ) -> Result<GraphExecutionContext, Error> {
                    unsafe {
                        #[repr(align(4))]
                        struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                        let mut ret_area = RetArea(
                            [::core::mem::MaybeUninit::uninit(); 8],
                        );
                        let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                        #[cfg(target_arch = "wasm32")]
                        #[link(wasm_import_module = "wasi:nn/graph@0.2.0-rc-2024-08-19")]
                        extern "C" {
                            #[link_name = "[method]graph.init-execution-context"]
                            fn wit_import(_: i32, _: *mut u8);
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        fn wit_import(_: i32, _: *mut u8) {
                            unreachable!()
                        }
                        wit_import((self).handle() as i32, ptr0);
                        let l1 = i32::from(*ptr0.add(0).cast::<u8>());
                        match l1 {
                            0 => {
                                let e = {
                                    let l2 = *ptr0.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::inference::GraphExecutionContext::from_handle(
                                        l2 as u32,
                                    )
                                };
                                Ok(e)
                            }
                            1 => {
                                let e = {
                                    let l3 = *ptr0.add(4).cast::<i32>();
                                    super::super::super::wasi::nn::errors::Error::from_handle(
                                        l3 as u32,
                                    )
                                };
                                Err(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        }
                    }
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            /// Load a `graph` from an opaque sequence of bytes to use for inference.
            pub fn load(
                builder: &[GraphBuilder],
                encoding: GraphEncoding,
                target: ExecutionTarget,
            ) -> Result<Graph, Error> {
                unsafe {
                    #[repr(align(4))]
                    struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                    let mut ret_area = RetArea([::core::mem::MaybeUninit::uninit(); 8]);
                    let vec1 = builder;
                    let len1 = vec1.len();
                    let layout1 = _rt::alloc::Layout::from_size_align_unchecked(
                        vec1.len() * 8,
                        4,
                    );
                    let result1 = if layout1.size() != 0 {
                        let ptr = _rt::alloc::alloc(layout1).cast::<u8>();
                        if ptr.is_null() {
                            _rt::alloc::handle_alloc_error(layout1);
                        }
                        ptr
                    } else {
                        ::core::ptr::null_mut()
                    };
                    for (i, e) in vec1.into_iter().enumerate() {
                        let base = result1.add(i * 8);
                        {
                            let vec0 = e;
                            let ptr0 = vec0.as_ptr().cast::<u8>();
                            let len0 = vec0.len();
                            *base.add(4).cast::<usize>() = len0;
                            *base.add(0).cast::<*mut u8>() = ptr0.cast_mut();
                        }
                    }
                    let ptr2 = ret_area.0.as_mut_ptr().cast::<u8>();
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "wasi:nn/graph@0.2.0-rc-2024-08-19")]
                    extern "C" {
                        #[link_name = "load"]
                        fn wit_import(_: *mut u8, _: usize, _: i32, _: i32, _: *mut u8);
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: *mut u8, _: usize, _: i32, _: i32, _: *mut u8) {
                        unreachable!()
                    }
                    wit_import(
                        result1,
                        len1,
                        encoding.clone() as i32,
                        target.clone() as i32,
                        ptr2,
                    );
                    let l3 = i32::from(*ptr2.add(0).cast::<u8>());
                    if layout1.size() != 0 {
                        _rt::alloc::dealloc(result1.cast(), layout1);
                    }
                    match l3 {
                        0 => {
                            let e = {
                                let l4 = *ptr2.add(4).cast::<i32>();
                                Graph::from_handle(l4 as u32)
                            };
                            Ok(e)
                        }
                        1 => {
                            let e = {
                                let l5 = *ptr2.add(4).cast::<i32>();
                                super::super::super::wasi::nn::errors::Error::from_handle(
                                    l5 as u32,
                                )
                            };
                            Err(e)
                        }
                        _ => _rt::invalid_enum_discriminant(),
                    }
                }
            }
            #[allow(unused_unsafe, clippy::all)]
            /// Load a `graph` by name.
            ///
            /// How the host expects the names to be passed and how it stores the graphs for retrieval via
            /// this function is **implementation-specific**. This allows hosts to choose name schemes that
            /// range from simple to complex (e.g., URLs?) and caching mechanisms of various kinds.
            pub fn load_by_name(name: &str) -> Result<Graph, Error> {
                unsafe {
                    #[repr(align(4))]
                    struct RetArea([::core::mem::MaybeUninit<u8>; 8]);
                    let mut ret_area = RetArea([::core::mem::MaybeUninit::uninit(); 8]);
                    let vec0 = name;
                    let ptr0 = vec0.as_ptr().cast::<u8>();
                    let len0 = vec0.len();
                    let ptr1 = ret_area.0.as_mut_ptr().cast::<u8>();
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "wasi:nn/graph@0.2.0-rc-2024-08-19")]
                    extern "C" {
                        #[link_name = "load-by-name"]
                        fn wit_import(_: *mut u8, _: usize, _: *mut u8);
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: *mut u8, _: usize, _: *mut u8) {
                        unreachable!()
                    }
                    wit_import(ptr0.cast_mut(), len0, ptr1);
                    let l2 = i32::from(*ptr1.add(0).cast::<u8>());
                    match l2 {
                        0 => {
                            let e = {
                                let l3 = *ptr1.add(4).cast::<i32>();
                                Graph::from_handle(l3 as u32)
                            };
                            Ok(e)
                        }
                        1 => {
                            let e = {
                                let l4 = *ptr1.add(4).cast::<i32>();
                                super::super::super::wasi::nn::errors::Error::from_handle(
                                    l4 as u32,
                                )
                            };
                            Err(e)
                        }
                        _ => _rt::invalid_enum_discriminant(),
                    }
                }
            }
        }
    }
}
mod _rt {
    pub use alloc_crate::vec::Vec;
    use core::fmt;
    use core::marker;
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
    /// A type which represents a component model resource, either imported or
    /// exported into this component.
    ///
    /// This is a low-level wrapper which handles the lifetime of the resource
    /// (namely this has a destructor). The `T` provided defines the component model
    /// intrinsics that this wrapper uses.
    ///
    /// One of the chief purposes of this type is to provide `Deref` implementations
    /// to access the underlying data when it is owned.
    ///
    /// This type is primarily used in generated code for exported and imported
    /// resources.
    #[repr(transparent)]
    pub struct Resource<T: WasmResource> {
        handle: AtomicU32,
        _marker: marker::PhantomData<T>,
    }
    /// A trait which all wasm resources implement, namely providing the ability to
    /// drop a resource.
    ///
    /// This generally is implemented by generated code, not user-facing code.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait WasmResource {
        /// Invokes the `[resource-drop]...` intrinsic.
        unsafe fn drop(handle: u32);
    }
    impl<T: WasmResource> Resource<T> {
        #[doc(hidden)]
        pub unsafe fn from_handle(handle: u32) -> Self {
            debug_assert!(handle != u32::MAX);
            Self {
                handle: AtomicU32::new(handle),
                _marker: marker::PhantomData,
            }
        }
        /// Takes ownership of the handle owned by `resource`.
        ///
        /// Note that this ideally would be `into_handle` taking `Resource<T>` by
        /// ownership. The code generator does not enable that in all situations,
        /// unfortunately, so this is provided instead.
        ///
        /// Also note that `take_handle` is in theory only ever called on values
        /// owned by a generated function. For example a generated function might
        /// take `Resource<T>` as an argument but then call `take_handle` on a
        /// reference to that argument. In that sense the dynamic nature of
        /// `take_handle` should only be exposed internally to generated code, not
        /// to user code.
        #[doc(hidden)]
        pub fn take_handle(resource: &Resource<T>) -> u32 {
            resource.handle.swap(u32::MAX, Relaxed)
        }
        #[doc(hidden)]
        pub fn handle(resource: &Resource<T>) -> u32 {
            resource.handle.load(Relaxed)
        }
    }
    impl<T: WasmResource> fmt::Debug for Resource<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Resource").field("handle", &self.handle).finish()
        }
    }
    impl<T: WasmResource> Drop for Resource<T> {
        fn drop(&mut self) {
            unsafe {
                match self.handle.load(Relaxed) {
                    u32::MAX => {}
                    other => T::drop(other),
                }
            }
        }
    }
    pub use alloc_crate::string::String;
    pub unsafe fn string_lift(bytes: Vec<u8>) -> String {
        if cfg!(debug_assertions) {
            String::from_utf8(bytes).unwrap()
        } else {
            String::from_utf8_unchecked(bytes)
        }
    }
    pub unsafe fn invalid_enum_discriminant<T>() -> T {
        if cfg!(debug_assertions) {
            panic!("invalid enum discriminant")
        } else {
            core::hint::unreachable_unchecked()
        }
    }
    pub use alloc_crate::alloc;
    extern crate alloc as alloc_crate;
}
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.31.0:wasi:nn@0.2.0-rc-2024-08-19:ml:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 1550] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\x95\x0b\x01A\x02\x01\
A\x0c\x01B\x11\x01py\x04\0\x11tensor-dimensions\x03\0\0\x01m\x07\x04FP16\x04FP32\
\x04FP64\x04BF16\x02U8\x03I32\x03I64\x04\0\x0btensor-type\x03\0\x02\x01p}\x04\0\x0b\
tensor-data\x03\0\x04\x04\0\x06tensor\x03\x01\x01i\x06\x01@\x03\x0adimensions\x01\
\x02ty\x03\x04data\x05\0\x07\x04\0\x13[constructor]tensor\x01\x08\x01h\x06\x01@\x01\
\x04self\x09\0\x01\x04\0\x19[method]tensor.dimensions\x01\x0a\x01@\x01\x04self\x09\
\0\x03\x04\0\x11[method]tensor.ty\x01\x0b\x01@\x01\x04self\x09\0\x05\x04\0\x13[m\
ethod]tensor.data\x01\x0c\x03\x01\"wasi:nn/tensor@0.2.0-rc-2024-08-19\x05\0\x01B\
\x08\x01m\x09\x10invalid-argument\x10invalid-encoding\x07timeout\x0druntime-erro\
r\x15unsupported-operation\x09too-large\x09not-found\x08security\x07unknown\x04\0\
\x0aerror-code\x03\0\0\x04\0\x05error\x03\x01\x01h\x02\x01@\x01\x04self\x03\0\x01\
\x04\0\x12[method]error.code\x01\x04\x01@\x01\x04self\x03\0s\x04\0\x12[method]er\
ror.data\x01\x05\x03\x01\"wasi:nn/errors@0.2.0-rc-2024-08-19\x05\x01\x02\x03\0\x01\
\x05error\x02\x03\0\0\x06tensor\x02\x03\0\0\x0btensor-data\x01B\x12\x02\x03\x02\x01\
\x02\x04\0\x05error\x03\0\0\x02\x03\x02\x01\x03\x04\0\x06tensor\x03\0\x02\x02\x03\
\x02\x01\x04\x04\0\x0btensor-data\x03\0\x04\x04\0\x17graph-execution-context\x03\
\x01\x01h\x06\x01i\x03\x01i\x01\x01j\0\x01\x09\x01@\x03\x04self\x07\x04names\x06\
tensor\x08\0\x0a\x04\0)[method]graph-execution-context.set-input\x01\x0b\x01@\x01\
\x04self\x07\0\x0a\x04\0'[method]graph-execution-context.compute\x01\x0c\x01j\x01\
\x08\x01\x09\x01@\x02\x04self\x07\x04names\0\x0d\x04\0*[method]graph-execution-c\
ontext.get-output\x01\x0e\x03\x01%wasi:nn/inference@0.2.0-rc-2024-08-19\x05\x05\x02\
\x03\0\x02\x17graph-execution-context\x01B\x1a\x02\x03\x02\x01\x02\x04\0\x05erro\
r\x03\0\0\x02\x03\x02\x01\x03\x04\0\x06tensor\x03\0\x02\x02\x03\x02\x01\x06\x04\0\
\x17graph-execution-context\x03\0\x04\x04\0\x05graph\x03\x01\x01m\x07\x08openvin\
o\x04onnx\x0atensorflow\x07pytorch\x0etensorflowlite\x04ggml\x0aautodetect\x04\0\
\x0egraph-encoding\x03\0\x07\x01m\x03\x03cpu\x03gpu\x03tpu\x04\0\x10execution-ta\
rget\x03\0\x09\x01p}\x04\0\x0dgraph-builder\x03\0\x0b\x01h\x06\x01i\x05\x01i\x01\
\x01j\x01\x0e\x01\x0f\x01@\x01\x04self\x0d\0\x10\x04\0$[method]graph.init-execut\
ion-context\x01\x11\x01p\x0c\x01i\x06\x01j\x01\x13\x01\x0f\x01@\x03\x07builder\x12\
\x08encoding\x08\x06target\x0a\0\x14\x04\0\x04load\x01\x15\x01@\x01\x04names\0\x14\
\x04\0\x0cload-by-name\x01\x16\x03\x01!wasi:nn/graph@0.2.0-rc-2024-08-19\x05\x07\
\x04\x01\x1ewasi:nn/ml@0.2.0-rc-2024-08-19\x04\0\x0b\x08\x01\0\x02ml\x03\0\0\0G\x09\
producers\x01\x0cprocessed-by\x02\x0dwit-component\x070.216.0\x10wit-bindgen-rus\
t\x060.31.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
