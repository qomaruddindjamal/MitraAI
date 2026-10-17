use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct Qwen3Tokenizer {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub special_tokens: HashMap<String, usize>,
    pub id_to_special: HashMap<usize, String>,
    pub byte_to_char: [char; 256],
    pub char_to_byte: HashMap<char, u8>,
}

impl Qwen3Tokenizer {
    pub fn new<P: AsRef<Path>>(vocab_json_path: P) -> Result<Self, String> {
        let mut file = File::open(vocab_json_path).map_err(|e| format!("Failed to open vocab file: {}", e))?;
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| format!("Failed to read vocab file: {}", e))?;

        let token_to_id: HashMap<String, usize> =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse vocab.json: {}", e))?;

        let mut id_to_token = HashMap::with_capacity(token_to_id.len());
        for (k, v) in &token_to_id {
            id_to_token.insert(*v, k.clone());
        }

        // Setup GPT-2 / Qwen Byte Level mapping
        let (byte_to_char, char_to_byte) = Self::bytes_to_unicode();

        // Special tokens standard Qwen
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|endoftext|>".to_string(), 151643);
        special_tokens.insert("<|im_start|>".to_string(), 151644);
        special_tokens.insert("<|im_end|>".to_string(), 151645);

        let mut id_to_special = HashMap::new();
        for (k, v) in &special_tokens {
            id_to_special.insert(*v, k.clone());
        }

        Ok(Self {
            token_to_id,
            id_to_token,
            special_tokens,
            id_to_special,
            byte_to_char,
            char_to_byte,
        })
    }

    /// Pemetaan Byte to Unicode standard GPT2/Qwen byte fallback
    fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = HashMap::new();

        let mut n = 0usize;
        for b in 0u8..=255 {
            let is_printable = (b >= b'!' && b <= b'~')
                || (b >= 161 && b <= 172)
                || b >= 174;

            let c = if is_printable {
                b as char
            } else {
                let mapped = char::from_u32(256 + n as u32).unwrap();
                n += 1;
                mapped
            };

            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b);
            if b == 255 {
                break;
            }
        }

        (byte_to_char, char_to_byte)
    }

    /// Encode teks ke rangkaian token id
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < text.len() {
            // Cek apakah match special token pada posisi ini
            let mut matched_special = false;
            for (st, id) in &self.special_tokens {
                if text[pos..].starts_with(st) {
                    tokens.push(*id);
                    pos += st.len();
                    matched_special = true;
                    break;
                }
            }
            if matched_special {
                continue;
            }

            // Greedy longest match BPE pieces
            let remaining = &text[pos..];
            let next_special_idx = self
                .special_tokens
                .keys()
                .filter_map(|st| remaining.find(st))
                .min()
                .unwrap_or(remaining.len());

            let chunk = &remaining[..next_special_idx];
            if chunk.is_empty() {
                continue;
            }

            // Encode chunk bytes ke unicode string
            let mut byte_encoded = String::with_capacity(chunk.len());
            for b in chunk.bytes() {
                byte_encoded.push(self.byte_to_char[b as usize]);
            }

            // Greedy forward matching subwords
            let mut sub_pos = 0;
            while sub_pos < byte_encoded.len() {
                let mut best_len = 0;
                let mut best_id = None;

                // Cari token terpanjang yang match
                for char_len in (1..=(byte_encoded.len() - sub_pos)).rev() {
                    // Cari slice valid char boundary
                    if let Some(sub_slice) = byte_encoded.get(sub_pos..sub_pos + char_len) {
                        if let Some(&id) = self.token_to_id.get(sub_slice) {
                            best_len = char_len;
                            best_id = Some(id);
                            break;
                        }
                    }
                }

                if let Some(id) = best_id {
                    tokens.push(id);
                    sub_pos += best_len;
                } else {
                    // Fallback satu karakter
                    let c = byte_encoded[sub_pos..].chars().next().unwrap();
                    let c_str = c.to_string();
                    if let Some(&id) = self.token_to_id.get(&c_str) {
                        tokens.push(id);
                    }
                    sub_pos += c.len_utf8();
                }
            }

            pos += chunk.len();
        }

        tokens
    }

    /// Decode satu atau rangkaian token id menjadi teks
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut raw_bytes = Vec::new();

        for &id in tokens {
            if let Some(st) = self.id_to_special.get(&id) {
                raw_bytes.extend_from_slice(st.as_bytes());
                continue;
            }

            if let Some(token_str) = self.id_to_token.get(&id) {
                for c in token_str.chars() {
                    if let Some(&b) = self.char_to_byte.get(&c) {
                        raw_bytes.push(b);
                    } else {
                        // Regular utf-8 fallback
                        let mut buf = [0u8; 4];
                        let s = c.encode_utf8(&mut buf);
                        raw_bytes.extend_from_slice(s.as_bytes());
                    }
                }
            }
        }

        String::from_utf8_lossy(&raw_bytes).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode_coverage() {
        let (byte_to_char, char_to_byte) = Qwen3Tokenizer::bytes_to_unicode();
        assert_eq!(byte_to_char.len(), 256);
        assert_eq!(char_to_byte.len(), 256);
        for b in 0u8..=255 {
            let c = byte_to_char[b as usize];
            assert_eq!(char_to_byte.get(&c), Some(&b));
        }
    }

    #[test]
    fn test_special_tokens_handling() {
        let mut token_to_id = HashMap::new();
        token_to_id.insert("hello".to_string(), 100);
        let mut id_to_token = HashMap::new();
        id_to_token.insert(100, "hello".to_string());

        let (byte_to_char, char_to_byte) = Qwen3Tokenizer::bytes_to_unicode();
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 151644);
        special_tokens.insert("<|im_end|>".to_string(), 151645);

        let mut id_to_special = HashMap::new();
        for (k, v) in &special_tokens {
            id_to_special.insert(*v, k.clone());
        }

        let tokenizer = Qwen3Tokenizer {
            token_to_id,
            id_to_token,
            special_tokens,
            id_to_special,
            byte_to_char,
            char_to_byte,
        };

        let encoded = tokenizer.encode("<|im_start|>hello<|im_end|>");
        assert_eq!(encoded, vec![151644, 100, 151645]);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, "<|im_start|>hello<|im_end|>");
    }
}
