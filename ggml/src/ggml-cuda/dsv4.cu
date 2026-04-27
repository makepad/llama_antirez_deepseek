#include "dsv4.cuh"

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

struct dsv4_rope_corr_dims {
    float v[2];
};

static __device__ float dsv4_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ void dsv4_rope_yarn(
        const float theta_extrap, const float freq_scale, const dsv4_rope_corr_dims corr_dims, const int64_t i0,
        const float ext_factor, float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = dsv4_rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

static __device__ float dsv4_e4m3fn_dequant(const float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    // E4M3FN positive finite values are monotonic, so nearest-value
    // dequantization can be computed directly instead of searching all 126
    // candidates for every scalar.
    if (ax < 0.0146484375f) { // midpoint between max subnormal and min normal
        const int mant = max(0, min(7, __float2int_rn(ax * 512.0f)));
        return sign * (float(mant) * 0.001953125f);
    }

    int exp_unbiased;
    const float frac = frexpf(ax, &exp_unbiased); // ax = frac * 2^exp_unbiased, frac in [0.5, 1)
    exp_unbiased -= 1;

    int exp = max(1, min(15, exp_unbiased + 7));
    const float base = exp2f(float(exp - 7));
    int mant = __float2int_rn((ax / base - 1.0f) * 8.0f);

    if (mant >= 8) {
        mant = 0;
        exp = min(15, exp + 1);
    }
    mant = max(0, min(7, mant));

    return sign * ((1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7)));
}

static __global__ void dsv4_hc_split_sinkhorn_kernel(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        int n_hc, int sinkhorn_iters, int64_t n_rows, int64_t mix_hc, float eps) {
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= n_rows || n_hc <= 0 || n_hc > 16) {
        return;
    }

    const float * mix = mixes + r * mix_hc;
    float * out = dst + r * mix_hc;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }

        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }

            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

static __global__ void dsv4_hc_split_sinkhorn_hc4_kernel(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        int sinkhorn_iters, int64_t n_rows, float eps) {
    const int64_t r = int64_t(blockIdx.x);
    const int lane = int(threadIdx.x);
    if (r >= n_rows || lane >= 16) {
        return;
    }

    const float * mix = mixes + r * 24;
    float * out = dst + r * 24;

    __shared__ float c[16];

    if (lane < 4) {
        const float z = mix[lane] * scale[0] + base[lane];
        out[lane] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    if (lane >= 4 && lane < 8) {
        const int i = lane - 4;
        const float z = mix[4 + i] * scale[1] + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }

    const int src_hc = lane & 3;
    const int dst_hc = lane >> 2;
    const int idx = src_hc + dst_hc * 4;
    const int off = 8 + idx;

    const float v = mix[off] * scale[2] + base[off];
    c[idx] = v;
    __syncthreads();

    const float row_max =
        fmaxf(fmaxf(c[dst_hc * 4 + 0], c[dst_hc * 4 + 1]),
              fmaxf(c[dst_hc * 4 + 2], c[dst_hc * 4 + 3]));
    float e = expf(v - row_max);
    const float row_sum =
        expf(c[dst_hc * 4 + 0] - row_max) +
        expf(c[dst_hc * 4 + 1] - row_max) +
        expf(c[dst_hc * 4 + 2] - row_max) +
        expf(c[dst_hc * 4 + 3] - row_max);
    c[idx] = e / row_sum + eps;
    __syncthreads();

    float col_sum = c[src_hc + 0 * 4] + c[src_hc + 1 * 4] + c[src_hc + 2 * 4] + c[src_hc + 3 * 4];
    c[idx] *= 1.0f / (col_sum + eps);
    __syncthreads();

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        const float row_denom = c[dst_hc * 4 + 0] + c[dst_hc * 4 + 1] + c[dst_hc * 4 + 2] + c[dst_hc * 4 + 3] + eps;
        c[idx] *= 1.0f / row_denom;
        __syncthreads();

        const float col_denom = c[src_hc + 0 * 4] + c[src_hc + 1 * 4] + c[src_hc + 2 * 4] + c[src_hc + 3 * 4] + eps;
        c[idx] *= 1.0f / col_denom;
        __syncthreads();
    }

    out[8 + idx] = c[idx];
}

static __global__ void dsv4_hc_expand_kernel(
        const char * __restrict__ block_out,
        const char * __restrict__ residual,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char       * __restrict__ dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        int64_t nb_block0, int64_t nb_block1,
        int64_t nb_res0, int64_t nb_res1, int64_t nb_res2,
        int64_t nb_post0, int64_t nb_post1,
        int64_t nb_comb0, int64_t nb_comb1, int64_t nb_comb2,
        int64_t nb0, int64_t nb1, int64_t nb2) {
    const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_hc * n_tokens;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d      = gid % n_embd;
    const int64_t tmp    = gid / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float block_v = *reinterpret_cast<const float *>(block_out + d * nb_block0 + t * nb_block1);
    const float post_v  = *reinterpret_cast<const float *>(post      + dst_hc * nb_post0 + t * nb_post1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *reinterpret_cast<const float *>(comb     + dst_hc * nb_comb0 + src_hc * nb_comb1 + t * nb_comb2);
        const float res_v  = *reinterpret_cast<const float *>(residual + d * nb_res0 + src_hc * nb_res1 + t * nb_res2);
        acc += comb_v * res_v;
    }

    *reinterpret_cast<float *>(dst + d * nb0 + dst_hc * nb1 + t * nb2) = acc;
}

static __global__ void dsv4_hc_weighted_sum_kernel(
        const char * __restrict__ x,
        const char * __restrict__ weights,
        char       * __restrict__ dst,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens,
        int64_t nb_x0, int64_t nb_x1, int64_t nb_x2,
        int64_t nb_w0, int64_t nb_w1,
        int64_t nb0, int64_t nb1) {
    const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = n_embd * n_tokens;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d = gid % n_embd;
    const int64_t t = gid / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        const float xv = *reinterpret_cast<const float *>(x       + d * nb_x0 + h * nb_x1 + t * nb_x2);
        const float wv = *reinterpret_cast<const float *>(weights + h * nb_w0 + t * nb_w1);
        acc += xv * wv;
    }

    *reinterpret_cast<float *>(dst + d * nb0 + t * nb1) = acc;
}

static __global__ void dsv4_fp8_kv_quantize_kernel(
        const char * __restrict__ src0,
        char       * __restrict__ dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
        int64_t n_rot) {
    __shared__ float scratch[64];

    const int64_t row = int64_t(blockIdx.x);
    const int64_t n_rows = ne01 * ne02 * ne03;
    if (row >= n_rows) {
        return;
    }

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src0 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    char       * dst_base = dst  + i1 * nb1  + i2 * nb2  + i3 * nb3;

    const int64_t n_nope = ne00 - n_rot;
    const int tid = threadIdx.x;

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64) {
            v = *reinterpret_cast<const float *>(src_base + (off + tid) * nb00);
            scratch[tid] = fabsf(v);
        }
        __syncthreads();

        for (int stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (tid < 64) {
            const float clamped = fminf(fmaxf(v / scale, -448.0f), 448.0f);
            const float q = dsv4_e4m3fn_dequant(clamped) * scale;
            *reinterpret_cast<float *>(dst_base + (off + tid) * nb0) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + tid; i < ne00; i += 64) {
        *reinterpret_cast<float *>(dst_base + i * nb0) =
            *reinterpret_cast<const float *>(src_base + i * nb00);
    }
}

static __global__ void dsv4_rope_tail_kernel(
        const char * __restrict__ src0,
        const int32_t * __restrict__ pos,
        const float * __restrict__ freq_factors,
        char * __restrict__ dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
        int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
        int n_dims, int mode, int inverse,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        dsv4_rope_corr_dims corr_dims, bool has_freq_factors) {
    const int64_t row = int64_t(blockIdx.x);
    if (row >= ne01 * ne02 * ne03) {
        return;
    }

    const int64_t i1 = row % ne01;
    const int64_t tmp = row / ne01;
    const int64_t i2 = tmp % ne02;
    const int64_t i3 = tmp / ne02;
    const int64_t n_nope = ne00 - n_dims;
    if (n_nope < 0) {
        return;
    }

    const char * src_base = src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char       * dst_base = dst  + i3 * nb3  + i2 * nb2  + i1 * nb1;

    const float theta_base = float(pos[i2]);
    const float inv_ndims = -1.0f / float(n_dims);
    const bool is_neox = mode == GGML_ROPE_TYPE_NEOX;

    for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        if (i0 < n_nope) {
            *reinterpret_cast<float *>(dst_base + i0 * nb0) =
                *reinterpret_cast<const float *>(src_base + i0 * nb00);
            continue;
        }

        const int64_t r = i0 - n_nope;
        if (is_neox) {
            const int64_t n_half = n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int64_t ic = r;
            const int64_t rel_i0 = 2 * ic;
            const float theta = theta_base * powf(freq_base, inv_ndims * float(rel_i0));
            const float freq_factor = has_freq_factors ? freq_factors[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta / freq_factor, freq_scale, corr_dims, rel_i0, ext_factor, attn_factor, cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int64_t j0 = n_nope + ic;
            const int64_t j1 = n_nope + ic + n_half;
            const float x0 = *reinterpret_cast<const float *>(src_base + j0 * nb00);
            const float x1 = *reinterpret_cast<const float *>(src_base + j1 * nb00);

            *reinterpret_cast<float *>(dst_base + j0 * nb0) = x0 * cos_theta - x1 * sin_theta;
            *reinterpret_cast<float *>(dst_base + j1 * nb0) = x0 * sin_theta + x1 * cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int64_t ic = r / 2;
            const float theta = theta_base * powf(freq_base, inv_ndims * float(r));
            const float freq_factor = has_freq_factors ? freq_factors[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta / freq_factor, freq_scale, corr_dims, r, ext_factor, attn_factor, cos_theta, sin_theta);
            if (inverse) {
                sin_theta = -sin_theta;
            }

            const int64_t j0 = n_nope + r;
            const int64_t j1 = j0 + 1;
            const float x0 = *reinterpret_cast<const float *>(src_base + j0 * nb00);
            const float x1 = *reinterpret_cast<const float *>(src_base + j1 * nb00);

            *reinterpret_cast<float *>(dst_base + j0 * nb0) = x0 * cos_theta - x1 * sin_theta;
            *reinterpret_cast<float *>(dst_base + j1 * nb0) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const int n_hc = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps = ggml_get_op_params_f32(dst, 2);
    const int64_t n_rows = ggml_nrows(mixes);
    const int64_t mix_hc = mixes->ne[0];

    if (n_hc == 4 && mix_hc == 24) {
        dsv4_hc_split_sinkhorn_hc4_kernel<<<n_rows, 16, 0, ctx.stream()>>>(
            static_cast<const float *>(mixes->data),
            static_cast<const float *>(dst->src[1]->data),
            static_cast<const float *>(dst->src[2]->data),
            static_cast<float *>(dst->data),
            sinkhorn_iters, n_rows, eps);
        return;
    }

    const int threads = 128;
    const int blocks = (n_rows + threads - 1) / threads;
    dsv4_hc_split_sinkhorn_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        static_cast<const float *>(mixes->data),
        static_cast<const float *>(dst->src[1]->data),
        static_cast<const float *>(dst->src[2]->data),
        static_cast<float *>(dst->data),
        n_hc, sinkhorn_iters, n_rows, mix_hc, eps);
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    const int64_t n_elem = dst->ne[0] * dst->ne[1];
    const int threads = int(std::min<int64_t>(256, std::max<int64_t>(1, n_elem)));
    const int blocks = (n_elem + threads - 1) / threads;
    dsv4_hc_weighted_sum_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        static_cast<const char *>(x->data),
        static_cast<const char *>(weights->data),
        static_cast<char *>(dst->data),
        dst->ne[0], x->ne[1], dst->ne[1],
        x->nb[0], x->nb[1], x->nb[2],
        weights->nb[0], weights->nb[1],
        dst->nb[0], dst->nb[1]);
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    const int64_t n_elem = dst->ne[0] * dst->ne[1] * dst->ne[2];
    const int threads = 256;
    const int blocks = (n_elem + threads - 1) / threads;
    dsv4_hc_expand_kernel<<<blocks, threads, 0, ctx.stream()>>>(
        static_cast<const char *>(block_out->data),
        static_cast<const char *>(residual->data),
        static_cast<const char *>(post->data),
        static_cast<const char *>(comb->data),
        static_cast<char *>(dst->data),
        dst->ne[0], dst->ne[1], dst->ne[2],
        block_out->nb[0], block_out->nb[1],
        residual->nb[0], residual->nb[1], residual->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int64_t n_rot = ggml_get_op_params_i32(dst, 0);

    dsv4_fp8_kv_quantize_kernel<<<n_rows, 64, 0, ctx.stream()>>>(
        static_cast<const char *>(src0->data),
        static_cast<char *>(dst->data),
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        n_rot);
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int n_dims     = ggml_get_op_params_i32(dst, 0);
    const int mode       = ggml_get_op_params_i32(dst, 1);
    const int n_ctx_orig = ggml_get_op_params_i32(dst, 2);
    const int inverse    = ggml_get_op_params_i32(dst, 3);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    dsv4_rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const int threads = int(std::min<int64_t>(256, std::max<int64_t>(1, src0->ne[0])));
    dsv4_rope_tail_kernel<<<n_rows, threads, 0, ctx.stream()>>>(
        static_cast<const char *>(src0->data),
        static_cast<const int32_t *>(src1->data),
        src2 ? static_cast<const float *>(src2->data) : nullptr,
        static_cast<char *>(dst->data),
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        n_dims, mode, inverse,
        freq_base, freq_scale, ext_factor, attn_factor,
        corr_dims, src2 != nullptr);
}
