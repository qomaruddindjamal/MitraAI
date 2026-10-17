pub mod config;
pub mod model;
pub mod safetensors;
pub mod server;
pub mod tensor;
pub mod tokenizer;

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use config::Qwen3Config;
use model::{KVCache, Qwen3Model};
use safetensors::SafeTensorsFile;
use tokenizer::Qwen3Tokenizer;

fn sample_next_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    if temperature <= 0.0 || top_k == 1 {
        // Greedy argmax
        let mut best_id = 0;
        let mut best_val = logits[0];
        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > best_val {
                best_val = val;
                best_id = i;
            }
        }
        return best_id;
    }

    // Top-K sampling
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();

    // Partial sort descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(indexed.len());
    let top = &mut indexed[..k];

    // Softmax pada top-k
    let max_val = top[0].1;
    let mut sum_exp = 0.0f32;
    for item in top.iter_mut() {
        item.1 = (item.1 - max_val).exp();
        sum_exp += item.1;
    }

    // Pseudo-random sampling sederhana namun deterministik/cepat
    let r = 0.5f32 * sum_exp; // Default median sampler untuk reproducible test
    let mut accum = 0.0f32;
    for item in top.iter() {
        accum += item.1;
        if accum >= r {
            return item.0;
        }
    }

    top[0].0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("     MITRAAI NATIVE RUST INFERENCE RUNTIME — QWEN3 CORE     ");
    println!("============================================================");

    let model_dir = "models/Qwen3-0.6B";
    let config_path = format!("{}/config.json", model_dir);
    let weights_path = format!("{}/model.safetensors", model_dir);
    let vocab_path = format!("{}/vocab.json", model_dir);

    if !Path::new(&weights_path).exists() {
        eprintln!("[MitraAI Notice] Model file not found at: {}", weights_path);
        eprintln!("Please ensure Qwen3-0.6B model is downloaded into 'models/Qwen3-0.6B/'");
        return Ok(());
    }

    println!("[MitraAI] Loading configuration from {}...", config_path);
    let config_str = fs::read_to_string(&config_path)?;
    let config = Qwen3Config::from_json_str(&config_str)?;
    println!(
        "[MitraAI] Architecture: layers={}, hidden={}, heads={}, kv_heads={}, head_dim={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim
    );

    println!("[MitraAI] Loading tokenizer from {}...", vocab_path);
    let tokenizer = Qwen3Tokenizer::new(&vocab_path)?;
    println!("[MitraAI] Vocab loaded ({} entries).", tokenizer.token_to_id.len());

    println!("[MitraAI] Opening safetensors file {}...", weights_path);
    let load_start = Instant::now();
    let st = SafeTensorsFile::open(&weights_path)?;
    let model = Qwen3Model::load_from_safetensors(&st, config.clone())?;
    let load_time = load_start.elapsed();
    println!("[MitraAI] Model loaded in {:.2?}", load_time);

    let mut kv_cache = KVCache::new(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        4096,
    );

    // Check CLI argument first: --serve or normal prompt / interactive REPL
    let args: Vec<String> = std::env::args().collect();
    let is_serve_mode = args.iter().any(|a| a == "--serve" || a == "-s")
        || std::env::var("PORT").is_ok();

    if is_serve_mode {
        let mut port: u16 = std::env::var("PORT")
            .unwrap_or_else(|_| "1234".to_string())
            .parse()
            .unwrap_or(1234);

        for i in 0..args.len() {
            if (args[i] == "--port" || args[i] == "-p") && i + 1 < args.len() {
                if let Ok(p) = args[i + 1].parse::<u16>() {
                    port = p;
                }
            }
        }
        let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());

        let engine = std::sync::Arc::new(std::sync::Mutex::new(crate::server::ModelEngine {
            model,
            tokenizer,
            config,
            kv_cache,
        }));

        println!("\n============================================================");
        println!("     MITRAAI NATIVE API SERVER — OPENAI COMPATIBLE          ");
        println!("============================================================");
        crate::server::start_server(&host, port, engine)?;
        return Ok(());
    }

    let initial_prompt = if args.len() > 1 {
        Some(args[1..].join(" "))
    } else {
        None
    };

    // Mode 1: Single CLI Prompt
    if let Some(prompt_text) = initial_prompt {
        println!("\n[MitraAI] Processing prompt: {}\n", prompt_text);
        let formatted_prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt_text
        );
        let prompt_tokens = tokenizer.encode(&formatted_prompt);
        let mut logits = vec![0.0f32; config.vocab_size];
        kv_cache.reset();

        let prefill_start = Instant::now();
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let is_last = pos == prompt_tokens.len() - 1;
            model.forward_token(token, pos, &mut kv_cache, &mut logits, is_last);
        }
        let prefill_time = prefill_start.elapsed();

        let max_gen_tokens = 256;
        let mut current_pos = prompt_tokens.len();
        let mut generated_tokens = Vec::new();

        print!("MitraAI > ");
        io::stdout().flush()?;

        let gen_start = Instant::now();
        for _ in 0..max_gen_tokens {
            let next_token = sample_next_token(&logits, 0.0, 1);
            if next_token == 151645 || next_token == 151643 {
                break;
            }
            generated_tokens.push(next_token);
            let token_text = tokenizer.decode(&[next_token]);
            print!("{}", token_text);
            io::stdout().flush()?;
            model.forward_token(next_token, current_pos, &mut kv_cache, &mut logits, true);
            current_pos += 1;
        }
        let gen_time = gen_start.elapsed();
        println!();

        let tps = if gen_time.as_secs_f64() > 0.0 {
            (generated_tokens.len() as f64) / gen_time.as_secs_f64()
        } else {
            0.0
        };
        println!("\x1b[90m[{:.1} t/s | TTFT: {:.2?} | {} tokens]\x1b[0m", tps, prefill_time, generated_tokens.len());
        return Ok(());
    }

    // Mode 2: Interactive REPL Loop
    println!("\n============================================================");
    println!("     MITRAAI INTERACTIVE CHAT MODE (Ketik 'exit' untuk keluar)");
    println!("============================================================\n");

    loop {
        print!("\nAnda > ");
        io::stdout().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }
        let trimmed = input.trim().to_string();
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("[MitraAI] Menutup sesi chat. Sampai jumpa!");
            break;
        }
        let user_prompt = trimmed;

        let formatted_prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_prompt
        );

        let prompt_tokens = tokenizer.encode(&formatted_prompt);
        let mut logits = vec![0.0f32; config.vocab_size];

        // Reset KV Cache untuk setiap turn percakapan
        kv_cache.reset();

        // Prefill KV Cache
        let prefill_start = Instant::now();
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let is_last = pos == prompt_tokens.len() - 1;
            model.forward_token(token, pos, &mut kv_cache, &mut logits, is_last);
        }
        let prefill_time = prefill_start.elapsed();

        // Autoregressive generation
        let max_gen_tokens = 256;
        let mut current_pos = prompt_tokens.len();
        let mut generated_tokens = Vec::new();

        print!("MitraAI > ");
        io::stdout().flush()?;

        let gen_start = Instant::now();
        for _ in 0..max_gen_tokens {
            let next_token = sample_next_token(&logits, 0.0, 1); // greedy decoding
            if next_token == 151645 || next_token == 151643 {
                // <|im_end|> or <|endoftext|>
                break;
            }

            generated_tokens.push(next_token);
            let token_text = tokenizer.decode(&[next_token]);
            print!("{}", token_text);
            io::stdout().flush()?;

            model.forward_token(next_token, current_pos, &mut kv_cache, &mut logits, true);
            current_pos += 1;
        }
        let gen_time = gen_start.elapsed();
        println!();

        let tps = if gen_time.as_secs_f64() > 0.0 {
            (generated_tokens.len() as f64) / gen_time.as_secs_f64()
        } else {
            0.0
        };
        println!("\x1b[90m[{:.1} t/s | TTFT: {:.2?} | {} tokens]\x1b[0m", tps, prefill_time, generated_tokens.len());
    }

    Ok(())
}
