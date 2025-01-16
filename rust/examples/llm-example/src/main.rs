use bytemuck::cast_slice;
use serde_json; // For JSON parsing
use std::collections::HashMap; // For HashMap
use std::convert::TryInto;
use std::fs;
use wasi_nn::{
    graph::{load, ExecutionTarget, Graph, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};

pub fn main() {
    let xml = fs::read_to_string("fixture/openvino_model.xml").unwrap();
    println!("Read graph XML, first 50 characters: {}", &xml[..50]);

    let weights = fs::read("fixture/openvino_model.bin").unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let graph = load(
        &[xml.into_bytes(), weights],
        GraphEncoding::Openvino,
        ExecutionTarget::Cpu,
    )
    .unwrap();
    println!("Loaded graph into wasi-nn");

    let context = unsafe { Graph::init_execution_context(&graph).unwrap() };
    println!("Created wasi-nn execution context!");

    let input_text = "Who are you?";

    // Load vocabulary from vocab.json
    let vocab: HashMap<i32, String> = load_vocab("fixture/vocab.json");
    println!("Vocabulary loaded with {} tokens.", vocab.len());

    // Tokenize with fixed length
    let max_length = 1024; // Update to match model's expected sequence length
    let indexed_tokens: Vec<i64> = tokenize_with_fixed_length(input_text, &vocab, max_length)
        .iter()
        .map(|&token| token as i64)
        .collect();

    // Define tensor dimensions: [1, sequence_length]
    let dimensions: TensorDimensions = vec![1, indexed_tokens.len() as u32];
    println!("Input tensor dimensions: {:?}", dimensions);

    // Create input tensor
    let data: TensorData = cast_slice::<i64, u8>(&indexed_tokens).to_vec();
    println!("Tensor data length (bytes): {}", data.len());

    let tensor_a = Tensor::new(&dimensions, TensorType::I64, &data);

    // Set the input tensor with the correct name from the model: "input_ids"
    context.set_input("input_ids", tensor_a).unwrap();
    println!("Set input tensor!");

    // Compute inference
    unsafe {
        context.compute().unwrap();
    }
    println!("Executed graph inference");

    // Retrieve the output tensor
    let output_tensor = context.get_output("logits").unwrap();
    println!("Retrieved output tensor");

    // Convert the tensor data (Vec<u8>) into Vec<f32>
    let output_data_bytes = output_tensor.data(); // Returns Vec<u8>
    let output_data: Vec<f32> = bytes_to_f32(output_data_bytes);
    println!("Output len: {}", output_data.len());

    println!(
        "Logits length: {}, Vocab size: {}, Logits/Vocab: {}",
        output_data.len(),
        vocab.len(),
        output_data.len() % vocab.len()
    );

    let decoded_text = decode_logits_to_text(&output_data, &vocab);
    println!("Decoded text: {}", decoded_text);
}
// Tokenization function (basic example)
fn tokenize(input: &str, vocab: &HashMap<i32, String>) -> Vec<i32> {
    let reverse_vocab: HashMap<String, i32> = vocab
        .iter()
        .map(|(id, token)| (token.clone(), *id))
        .collect();

    input
        .split_whitespace()
        .map(|word| *reverse_vocab.get(word).unwrap_or(&0)) // Map to vocab ID, default to <UNK> (0)
        .collect()
}

fn tokenize_with_fixed_length(
    input: &str,
    vocab: &HashMap<i32, String>,
    max_length: usize,
) -> Vec<i32> {
    let mut tokens = tokenize(input, vocab);
    if tokens.len() > max_length {
        tokens.truncate(max_length);
    } else if tokens.len() < max_length {
        tokens.extend(vec![0; max_length - tokens.len()]); // Assuming 0 is <PAD>
    }
    tokens
}

// Helper function to convert bytes to f32
fn bytes_to_f32(data: Vec<u8>) -> Vec<f32> {
    data.chunks(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

// Decoding function (softmax example)
fn decode_logits(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    exp_logits.into_iter().map(|x| x / sum_exp_logits).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum_exp_logits).collect()
}

// Decode the logits into a sequence of tokens
fn decode_logits_to_text(logits: &[f32], vocab: &HashMap<i32, String>) -> String {
    let probabilities = logits.chunks(vocab.len()).map(softmax);
    let mut decoded_tokens = Vec::new();

    for prob in probabilities {
        if let Some((idx, _)) = prob
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            let token = vocab
                .get(&(idx as i32))
                .unwrap_or(&"<UNK>".to_string())
                .clone();
            decoded_tokens.push(token);
        }
    }

    decoded_tokens.join(" ")
}

fn load_vocab(file_path: &str) -> HashMap<i32, String> {
    let file = fs::File::open(file_path).expect("Failed to open vocab file");
    let raw_vocab: HashMap<String, String> =
        serde_json::from_reader(file).expect("Failed to parse vocab JSON");

    //println!("Raw vocab: {:?}", raw_vocab);

    raw_vocab
        .into_iter()
        .map(|(key, value)| {
            let key_as_i32 = key.parse::<i32>().expect("Key is not a valid i32");
            (key_as_i32, value)
        })
        .collect()
}
