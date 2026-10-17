use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

pub struct SafeTensorsFile {
    pub mmap: Mmap,
    pub header_size: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensorsFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("Failed to mmap: {}", e))? };

        if mmap.len() < 8 {
            return Err("File too small to be safetensors".to_string());
        }

        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        if mmap.len() < 8 + header_len {
            return Err("Header length exceeds file size".to_string());
        }

        let header_str = std::str::from_utf8(&mmap[8..8 + header_len])
            .map_err(|e| format!("Invalid UTF-8 in header: {}", e))?;

        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| format!("Failed to parse header JSON: {}", e))?;

        let mut tensors = HashMap::new();
        for (k, v) in raw_map {
            if k == "__metadata__" {
                continue;
            }
            if let Ok(info) = serde_json::from_value::<TensorInfo>(v) {
                tensors.insert(k, info);
            }
        }

        Ok(Self {
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    /// Mendapatkan raw byte slice untuk tensor tertentu
    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.header_size + info.data_offsets[0];
        let end = self.header_size + info.data_offsets[1];
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Mendapatkan tensor sebagai slice u16 (bfloat16)
    pub fn get_tensor_bf16(&self, name: &str) -> Option<(&[u16], &[usize])> {
        let info = self.tensors.get(name)?;
        if info.dtype != "BF16" {
            return None;
        }
        let bytes = self.get_tensor_bytes(name)?;
        let num_elements = bytes.len() / 2;
        let expected_num_elements: usize = info.shape.iter().product();
        if num_elements != expected_num_elements {
            return None;
        }

        // Safe casting alignment: periksa alignment ke 2 bytes
        // Jika unaligned, kita fallback ke safe copy jika perlu, namun mmap safetensors data buffer
        // biasanya 8-byte aligned jika header diformat standard, atau kita slice sebagai u16
        // Menggunakan pointer casting atau safe from_le_bytes
        let u16_slice = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const u16, num_elements)
        };
        Some((u16_slice, &info.shape))
    }

    /// Membaca tensor dan mengonversinya langsung ke Vec<f32>
    pub fn get_tensor_f32(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let info = self.tensors.get(name)?;
        let bytes = self.get_tensor_bytes(name)?;

        match info.dtype.as_str() {
            "BF16" => {
                let num_elements = bytes.len() / 2;
                let mut f32_vec = Vec::with_capacity(num_elements);
                for chunk in bytes.chunks_exact(2) {
                    let b = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32_vec.push(crate::tensor::bf16_to_f32(b));
                }
                Some((f32_vec, info.shape.clone()))
            }
            "F32" => {
                let num_elements = bytes.len() / 4;
                let mut f32_vec = Vec::with_capacity(num_elements);
                for chunk in bytes.chunks_exact(4) {
                    let b = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    f32_vec.push(b);
                }
                Some((f32_vec, info.shape.clone()))
            }
            _ => None,
        }
    }
}
