use bytemuck::cast_slice;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufRead, Write};
use rand::prelude::*;
use wasi_nn::{
    graph::{load, ExecutionTarget, Graph, GraphEncoding},
    tensor::{Tensor, TensorData, TensorDimensions, TensorType},
};

const MAX_LENGTH: usize = 1024;
const GENERATE_STEPS: usize = 50;

fn main() {
    // Load vocab.txt and merges.txt
    let vocab: HashMap<String, u32> = load_vocab("fixture/vocab.json");
    let merges: HashMap<(String, String), usize> = load_merges("fixture/merges.txt");

    // Tokenize the input prompt file
    let prompt = std::fs::read_to_string("fixture/prompt")
        .expect("Failed to read prompt file from fixture/prompt")
        .trim()
        .to_string();

    let mut token_ids: Vec<i64> = gpt2_tokenize(&prompt, &vocab, &merges).into_iter().map(|x| x as i64).collect();

    let token_strings: Vec<String> = token_ids.iter()
    .map(|&id| vocab.iter().find(|(_, &v)| v == id as u32).map(|(k, _)| k.clone()).unwrap_or("<unk>".to_string()))
    .collect();

    println!("[DEBUG] Token strings passed to model: {:?}", token_strings);

    let xml = std::fs::read_to_string("fixture/openvino_model.xml").unwrap();
    let weights = std::fs::read("fixture/openvino_model.bin").unwrap();
    let graph = load(&[xml.into_bytes(), weights], GraphEncoding::Openvino, ExecutionTarget::Cpu).unwrap();

    print!("{}", prompt);

    for _ in 0..GENERATE_STEPS {
        let context = Graph::init_execution_context(&graph).unwrap();

        // Ensure input is trimmed to last MAX_LENGTH
        let input_len = token_ids.len();
        if input_len > MAX_LENGTH {
            token_ids = token_ids[input_len - MAX_LENGTH..].to_vec();
        }

        let mut padded_tokens = token_ids.clone();
        padded_tokens.resize(MAX_LENGTH, 0);

        let dimensions: TensorDimensions = vec![1, MAX_LENGTH as u32];
        let data: TensorData = cast_slice::<i64, u8>(&padded_tokens).to_vec();
        let tensor = Tensor::new(&dimensions, TensorType::I64, &data);
        context.set_input("input_ids", tensor).unwrap();
        context.compute().unwrap();

        let output_tensor = context.get_output("logits").unwrap();
        let output_data_bytes = output_tensor.data();
        let output_data: Vec<f32> = bytes_to_f32(output_data_bytes);

        let vocab_size = vocab.len();
        let logits = &output_data[(input_len - 1) * vocab_size .. input_len * vocab_size];
        
        // To see what is the best 5 predictions next
        // let mut logits_with_index: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        // logits_with_index.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // for (i, (idx, logit)) in logits_with_index.iter().take(5).enumerate() {
        //     let tok = vocab.iter().find(|(_, &v)| v as usize == *idx).map(|(k, _)| k.clone()).unwrap_or("<unk>".to_string());
        //     println!("Top {}: Token ID: {}, Logit: {:.4}, Token: {}", i + 1, idx, logit, tok);
        // }

        let temperature = 1.0;
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        let probabilities = softmax(&scaled_logits);
        let predicted_token_id = sample_from_distribution(&probabilities);
        // check <|endoftext|>
        if predicted_token_id == 50256 {
        println!("\n[INFO] End-of-text token generated. Stopping early.");
        break;
        }   
        token_ids.push(predicted_token_id as i64);
        // Decode only the last token (optional optimization)
        let decoded = decode_tokens(&token_ids[token_ids.len() - 1..], &vocab);
        print!("{}", decoded);
        io::stdout().flush().unwrap(); // Force it to print immediately
    }

    println!("\n{:?}", token_ids.iter().take(10).collect::<Vec<_>>());
    // Decode all tokens
    let full_text = decode_tokens(&token_ids, &vocab);
    // println!("Token preview: {}", full_text);
    // println!("\n=== Generated Text ===\n{}", full_text);
}

fn load_vocab(path: &str) -> HashMap<String, u32> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let raw: HashMap<String, String> = serde_json::from_reader(reader).unwrap();

    raw.into_iter()
        .map(|(id_str, token)| {
            let id = id_str.parse::<u32>().expect("Invalid ID in vocab.json");
            (token, id)
        })
        .collect()
}

fn load_merges(path: &str) -> HashMap<(String, String), usize> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    reader
        .lines()
        .skip(1) // skip "#version: ..."
        .enumerate()
        .map(|(i, line)| {
            let parts: Vec<String> = line.unwrap().split_whitespace().map(|s| s.to_string()).collect();
            ((parts[0].clone(), parts[1].clone()), i)
        })
        .collect()
}


fn byte_to_unicode(byte: u8) -> char {
    match byte {
        33..=126 | 161..=172 | 174..=255 => byte as char,
        _ => std::char::from_u32(byte as u32 + 0x0100).unwrap(),
    }
}





fn gpt2_tokenize(text: &str, vocab: &HashMap<String, u32>, merges: &HashMap<(String, String), usize>) -> Vec<u32> {
    let mut tokens = Vec::new();

    for word in text.split_inclusive(char::is_whitespace) {
        let byte_encoded: String = word
            .as_bytes()
            .iter()
            .map(|&b| byte_to_unicode(b).to_string())
            .collect();

        let word_tokens = bpe(&byte_encoded, vocab, merges);
        tokens.extend(word_tokens);
    }

    tokens
}




fn bpe(word: &str, vocab: &HashMap<String, u32>, merges: &HashMap<(String, String), usize>) -> Vec<u32> {
    let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    if chars.is_empty() {
        return vec![];
    }

    let mut pairs = get_pairs(&chars);

    while !pairs.is_empty() {
        let min_pair = pairs.iter()
            .filter_map(|pair| merges.get(pair).map(|&rank| (pair.clone(), rank)))
            .min_by_key(|(_, rank)| *rank);

        if let Some((pair_to_merge, _)) = min_pair {
            let (a, b) = pair_to_merge;

            // println!("Merging: ({:?}, {:?})", a, b);

            let mut new_word = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                if i < chars.len() - 1 && chars[i] == a && chars[i + 1] == b {
                    new_word.push(format!("{}{}", a, b));
                    i += 2;
                } else {
                    new_word.push(chars[i].clone());
                    i += 1;
                }
            }

            chars = new_word;
            // println!("Current tokens: {:?}", chars);
            pairs = get_pairs(&chars);
            // println!("Next pairs: {:?}", pairs);
        } else {
            break;
        }
    }

    chars
        .into_iter()
        .map(|s| *vocab.get(&s).unwrap_or(&0)) // fallback to <unk>
        .collect()
}







fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    for i in 0..word.len() - 1 {
        pairs.insert((word[i].clone(), word[i + 1].clone()));
    }
    pairs
}

fn bytes_to_f32(data: Vec<u8>) -> Vec<f32> {
    use std::convert::TryInto;
    data.chunks(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum_exp_logits).collect()
}

fn sample_from_distribution(probabilities: &[f32]) -> usize {
    let mut rng = thread_rng();
    let mut cumulative = 0.0;
    let choice: f32 = rng.gen();
    for (i, &prob) in probabilities.iter().enumerate() {
        cumulative += prob;
        if choice < cumulative {
            return i;
        }
    }
    probabilities.len() - 1
}


fn decode_tokens(tokens: &[i64], vocab: &HashMap<String, u32>) -> String {
    // Invert vocab: ID -> token
    let mut id_to_token: Vec<&str> = vec!["<UNK>"; vocab.len()];
    for (tok, &id) in vocab.iter() {
        if (id as usize) < id_to_token.len() {
            id_to_token[id as usize] = tok;
        }
    }

    let mut output = String::new();
    for &id in tokens {
        let tok = id_to_token.get(id as usize).unwrap_or(&"<UNK>");

        if let Some(rest) = tok.strip_prefix('\u{0120}') {
            output.push(' ');
            output.push_str(rest);
        } else if *tok == "Ċ" || *tok == "\n" {
            output.push('\n');
        } else {
            output.push_str(tok);
        }
    }


    output
}




