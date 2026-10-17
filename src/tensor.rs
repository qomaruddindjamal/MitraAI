use rayon::prelude::*;

/// Konversi bfloat16 ke IEEE 754 f32 standard.
#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Root Mean Square Normalization (RMSNorm).
/// y = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(out.len(), x.len());

    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / (x.len() as f32);
    let scale = 1.0f32 / (mean_sq + eps).sqrt();

    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// In-place RMSNorm.
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / (x.len() as f32);
    let scale = 1.0f32 / (mean_sq + eps).sqrt();

    for i in 0..x.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

/// Matmul: y = W * x
/// W shape: [out_dim, in_dim] dalam row-major format f32.
/// x shape: [in_dim]
/// y shape: [out_dim]
pub fn matmul_vec_f32(y: &mut [f32], w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    // Chunk size untuk rayon agar thread overhead minimal
    let chunk_size = (out_dim / rayon::current_num_threads()).max(16);

    y.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base_i = chunk_idx * chunk_size;
            for (offset, out_val) in chunk.iter_mut().enumerate() {
                let i = base_i + offset;
                let row_start = i * in_dim;
                let row = &w[row_start..row_start + in_dim];
                
                // 8-way accumulator untuk efisiensi FMA (Fused Multiply-Add)
                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                let mut sum3 = 0.0f32;
                let mut sum4 = 0.0f32;
                let mut sum5 = 0.0f32;
                let mut sum6 = 0.0f32;
                let mut sum7 = 0.0f32;

                let chunks = in_dim / 8;
                let remainder = in_dim % 8;

                for c in 0..chunks {
                    let idx = c * 8;
                    sum0 += row[idx] * x[idx];
                    sum1 += row[idx + 1] * x[idx + 1];
                    sum2 += row[idx + 2] * x[idx + 2];
                    sum3 += row[idx + 3] * x[idx + 3];
                    sum4 += row[idx + 4] * x[idx + 4];
                    sum5 += row[idx + 5] * x[idx + 5];
                    sum6 += row[idx + 6] * x[idx + 6];
                    sum7 += row[idx + 7] * x[idx + 7];
                }

                let mut total = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
                let rem_start = chunks * 8;
                for r in 0..remainder {
                    total += row[rem_start + r] * x[rem_start + r];
                }

                *out_val = total;
            }
        });
}

/// Matmul dari raw bfloat16 weights zero-allocation: y = W_bf16 * x_f32
/// Menghindari konversi seluruh bobot model ratusan MB ke RAM f32 terlebih dahulu.
pub fn matmul_vec_bf16(y: &mut [f32], w_bf16: &[u16], x: &[f32], out_dim: usize, in_dim: usize) {
    debug_assert_eq!(w_bf16.len(), out_dim * in_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(y.len(), out_dim);

    y.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let row_start = i * in_dim;
        let row = &w_bf16[row_start..row_start + in_dim];
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += bf16_to_f32(row[j]) * x[j];
        }
        *out_val = sum;
    });
}

/// SiLU activation function: x * sigmoid(x) = x / (1.0 + exp(-x))
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

/// SwiGLU activation and elementwise product:
/// out[i] = silu(gate[i]) * up[i]
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(out.len(), gate.len());
    debug_assert_eq!(out.len(), up.len());

    for i in 0..out.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Softmax in-place numerik stabil:
/// x[i] = exp(x[i] - max) / sum(exp(x[j] - max))
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = x[0];
    for &v in x.iter().skip(1) {
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum_exp = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }

    let inv_sum = 1.0f32 / sum_exp;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Rotary Position Embedding (RoPE) untuk satu head.
/// head_vec length = head_dim (misal 128 untuk Qwen3-0.6B)
/// Dalam Qwen / LLaMA standard RoPE:
/// split head_vec menjadi pairs atau dua halves (di HF Qwen: pairs atau interleaved tergantung arsitektur).
/// Periksa konvensi Qwen3: rotary_dim = head_dim, frekuensi theta_i = base^(-2*(i)/(dim)).
/// RoPE diaplikasikan pada setengah pertama x1 dan setengah kedua x2 atau interleaved.
/// Di Qwen2 / Qwen3:
/// freqs = 1.0 / (rope_theta ^ ((0..head_dim/2)*2 / head_dim))
/// x1 = head[i], x2 = head[i + head_dim/2]
/// out1 = x1 * cos - x2 * sin
/// out2 = x1 * sin + x2 * cos
pub fn apply_rope_to_head(
    head: &mut [f32],
    pos: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    debug_assert_eq!(head.len(), head_dim);
    let half_dim = head_dim / 2;

    for i in 0..half_dim {
        let freq = 1.0f32 / (rope_theta.powf((2 * i) as f32 / head_dim as f32));
        let angle = (pos as f32) * freq;
        let cos = angle.cos();
        let sin = angle.sin();

        let x1 = head[i];
        let x2 = head[i + half_dim];

        head[i] = x1 * cos - x2 * sin;
        head[i + half_dim] = x1 * sin + x2 * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        let one_bf16 = 0x3f80;
        let one_f32 = bf16_to_f32(one_bf16);
        assert!((one_f32 - 1.0).abs() < 1e-6);

        let zero_bf16 = 0x0000;
        assert_eq!(bf16_to_f32(zero_bf16), 0.0);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rms_norm(&mut out, &x, &weight, 1e-6);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_softmax() {
        let mut scores = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[2] > scores[1] && scores[1] > scores[0]);
    }

    #[test]
    fn test_swiglu() {
        let gate = vec![0.0, 2.0];
        let up = vec![5.0, 3.0];
        let mut out = vec![0.0; 2];
        swiglu(&mut out, &gate, &up);
        // silu(0) = 0 * sigmoid(0) = 0
        assert_eq!(out[0], 0.0);
        let expected_1 = (2.0 / (1.0 + (-2.0f32).exp())) * 3.0;
        assert!((out[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_vec() {
        // W = [[1, 2], [3, 4]]
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![0.5, 1.5];
        let mut y = vec![0.0; 2];
        matmul_vec_f32(&mut y, &w, &x, 2, 2);
        // y[0] = 1*0.5 + 2*1.5 = 3.5
        // y[1] = 3*0.5 + 4*1.5 = 7.5
        assert!((y[0] - 3.5).abs() < 1e-6);
        assert!((y[1] - 7.5).abs() < 1e-6);
    }
}
