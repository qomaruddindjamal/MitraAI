use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default = "default_tie_embeddings")]
    pub tie_word_embeddings: bool,
}

fn default_tie_embeddings() -> bool {
    true
}

impl Qwen3Config {
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
