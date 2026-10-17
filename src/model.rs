use crate::config::Qwen3Config;
use crate::safetensors::SafeTensorsFile;
use crate::tensor::*;
use rayon::prelude::*;

pub struct Qwen3LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<Qwen3LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f32>>,
}

pub struct KVCache {
    /// key_cache: [layer, num_kv_heads * max_seq_len * head_dim]
    pub keys: Vec<Vec<f32>>,
    /// val_cache: [layer, num_kv_heads * max_seq_len * head_dim]
    pub values: Vec<Vec<f32>>,
    pub max_seq_len: usize,
    pub cur_seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let layer_size = num_kv_heads * max_seq_len * head_dim;
        let mut keys = Vec::with_capacity(num_layers);
        let mut values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            keys.push(vec![0.0f32; layer_size]);
            values.push(vec![0.0f32; layer_size]);
        }
        Self {
            keys,
            values,
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cur_seq_len = 0;
    }
}

impl Qwen3Model {
    pub fn load_from_safetensors(st: &SafeTensorsFile, config: Qwen3Config) -> Result<Self, String> {
        println!("[MitraAI] Loading embedding table...");
        let (embed_tokens, _) = st
            .get_tensor_f32("model.embed_tokens.weight")
            .ok_or("Missing model.embed_tokens.weight")?;

        println!("[MitraAI] Loading final norm...");
        let (final_norm, _) = st
            .get_tensor_f32("model.norm.weight")
            .ok_or("Missing model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            st.get_tensor_f32("lm_head.weight").map(|(w, _)| w)
        };

        println!("[MitraAI] Loading {} transformer layers in parallel...", config.num_hidden_layers);
        let layers: Result<Vec<Qwen3LayerWeights>, String> = (0..config.num_hidden_layers)
            .into_par_iter()
            .map(|i| {
                let prefix = format!("model.layers.{}.", i);
                let (input_layernorm, _) = st
                    .get_tensor_f32(&format!("{}input_layernorm.weight", prefix))
                    .ok_or_else(|| format!("Missing {}input_layernorm.weight", prefix))?;

                let (q_proj, _) = st
                    .get_tensor_f32(&format!("{}self_attn.q_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.q_proj.weight", prefix))?;

                let (k_proj, _) = st
                    .get_tensor_f32(&format!("{}self_attn.k_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.k_proj.weight", prefix))?;

                let (v_proj, _) = st
                    .get_tensor_f32(&format!("{}self_attn.v_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.v_proj.weight", prefix))?;

                let (o_proj, _) = st
                    .get_tensor_f32(&format!("{}self_attn.o_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.o_proj.weight", prefix))?;

                let (q_norm, _) = st
                    .get_tensor_f32(&format!("{}self_attn.q_norm.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.q_norm.weight", prefix))?;

                let (k_norm, _) = st
                    .get_tensor_f32(&format!("{}self_attn.k_norm.weight", prefix))
                    .ok_or_else(|| format!("Missing {}self_attn.k_norm.weight", prefix))?;

                let (post_attention_layernorm, _) = st
                    .get_tensor_f32(&format!("{}post_attention_layernorm.weight", prefix))
                    .ok_or_else(|| format!("Missing {}post_attention_layernorm.weight", prefix))?;

                let (gate_proj, _) = st
                    .get_tensor_f32(&format!("{}mlp.gate_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}mlp.gate_proj.weight", prefix))?;

                let (up_proj, _) = st
                    .get_tensor_f32(&format!("{}mlp.up_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}mlp.up_proj.weight", prefix))?;

                let (down_proj, _) = st
                    .get_tensor_f32(&format!("{}mlp.down_proj.weight", prefix))
                    .ok_or_else(|| format!("Missing {}mlp.down_proj.weight", prefix))?;

                Ok(Qwen3LayerWeights {
                    input_layernorm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    post_attention_layernorm,
                    gate_proj,
                    up_proj,
                    down_proj,
                })
            })
            .collect();
        let layers = layers?;

        println!("[MitraAI] All layers successfully loaded into RAM!");
        Ok(Self {
            config,
            embed_tokens,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// Single token forward pass untuk next-token prediction dengan KV-Cache
    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KVCache,
        logits: &mut [f32],
        compute_logits: bool,
    ) {
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let intermediate_size = self.config.intermediate_size;
        let num_queries_per_kv = num_heads / num_kv_heads;
        let rope_theta = self.config.rope_theta;
        let eps = self.config.rms_norm_eps;

        // 1. Embedding lookup
        let mut x = vec![0.0f32; hidden_size];
        let emb_offset = token_id * hidden_size;
        x.copy_from_slice(&self.embed_tokens[emb_offset..emb_offset + hidden_size]);

        // Buffers untuk layer execution
        let mut norm_x = vec![0.0f32; hidden_size];
        let mut q = vec![0.0f32; num_heads * head_dim];
        let mut k = vec![0.0f32; num_kv_heads * head_dim];
        let mut v = vec![0.0f32; num_kv_heads * head_dim];
        let mut attn_out = vec![0.0f32; num_heads * head_dim];
        let mut proj_out = vec![0.0f32; hidden_size];

        let mut gate = vec![0.0f32; intermediate_size];
        let mut up = vec![0.0f32; intermediate_size];
        let mut mlp_act = vec![0.0f32; intermediate_size];
        let mut mlp_out = vec![0.0f32; hidden_size];

        let mut attn_scores = vec![0.0f32; pos + 1];

        // 2. Transformer layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &self.layers[layer_idx];

            // RMSNorm input
            rms_norm(&mut norm_x, &x, &layer.input_layernorm, eps);

            // Q, K, V Projections
            matmul_vec_f32(&mut q, &layer.q_proj, &norm_x, num_heads * head_dim, hidden_size);
            matmul_vec_f32(&mut k, &layer.k_proj, &norm_x, num_kv_heads * head_dim, hidden_size);
            matmul_vec_f32(&mut v, &layer.v_proj, &norm_x, num_kv_heads * head_dim, hidden_size);

            // Q-Norm & K-Norm per head (Karakteristik penting Qwen3!)
            for h in 0..num_heads {
                let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
                rms_norm_inplace(q_head, &layer.q_norm, eps);
                // Apply RoPE
                apply_rope_to_head(q_head, pos, head_dim, rope_theta);
            }

            for kh in 0..num_kv_heads {
                let k_head = &mut k[kh * head_dim..(kh + 1) * head_dim];
                rms_norm_inplace(k_head, &layer.k_norm, eps);
                // Apply RoPE
                apply_rope_to_head(k_head, pos, head_dim, rope_theta);
            }

            // Simpan K dan V ke KV-Cache pada posisi `pos`
            let key_layer = &mut kv_cache.keys[layer_idx];
            let val_layer = &mut kv_cache.values[layer_idx];

            for kh in 0..num_kv_heads {
                let kv_offset = (kh * kv_cache.max_seq_len + pos) * head_dim;
                let k_src = &k[kh * head_dim..(kh + 1) * head_dim];
                let v_src = &v[kh * head_dim..(kh + 1) * head_dim];
                key_layer[kv_offset..kv_offset + head_dim].copy_from_slice(k_src);
                val_layer[kv_offset..kv_offset + head_dim].copy_from_slice(v_src);
            }

            // Multi-head / Grouped Query Attention computation
            let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();

            for h in 0..num_heads {
                let kh = h / num_queries_per_kv;
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                // Attention score untuk setiap posisi 0..=pos
                for t in 0..=pos {
                    let k_offset = (kh * kv_cache.max_seq_len + t) * head_dim;
                    let k_t = &key_layer[k_offset..k_offset + head_dim];

                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_head[d] * k_t[d];
                    }
                    attn_scores[t] = dot * inv_sqrt_d;
                }

                // Causal Softmax
                softmax_inplace(&mut attn_scores[0..=pos]);

                // Weighted sum over V
                let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                out_head.fill(0.0f32);

                for t in 0..=pos {
                    let score = attn_scores[t];
                    let v_offset = (kh * kv_cache.max_seq_len + t) * head_dim;
                    let v_t = &val_layer[v_offset..v_offset + head_dim];

                    for d in 0..head_dim {
                        out_head[d] += score * v_t[d];
                    }
                }
            }

            // Output projection
            matmul_vec_f32(&mut proj_out, &layer.o_proj, &attn_out, hidden_size, num_heads * head_dim);

            // Residual connection: x = x + proj_out
            for i in 0..hidden_size {
                x[i] += proj_out[i];
            }

            // Post-attention RMSNorm
            rms_norm(&mut norm_x, &x, &layer.post_attention_layernorm, eps);

            // MLP: SwiGLU
            matmul_vec_f32(&mut gate, &layer.gate_proj, &norm_x, intermediate_size, hidden_size);
            matmul_vec_f32(&mut up, &layer.up_proj, &norm_x, intermediate_size, hidden_size);
            swiglu(&mut mlp_act, &gate, &up);
            matmul_vec_f32(&mut mlp_out, &layer.down_proj, &mlp_act, hidden_size, intermediate_size);

            // Residual connection: x = x + mlp_out
            for i in 0..hidden_size {
                x[i] += mlp_out[i];
            }
        }

        if !compute_logits {
            return;
        }

        // 3. Final RMSNorm
        rms_norm(&mut norm_x, &x, &self.final_norm, eps);

        // 4. Output projection ke logits (Vocab size)
        let head_weight = match &self.lm_head {
            Some(w) => w,
            None => &self.embed_tokens,
        };

        matmul_vec_f32(logits, head_weight, &norm_x, self.config.vocab_size, hidden_size);
    }
}
