#include "concat.cuh"

template <typename T>
static __global__ void concat_dim0_kernel(const T * x, const T * y, T * dst, const int ne0, const int ne00) {
    const int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    const int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) {
        const int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        const int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim1_kernel(const T * x, const T * y, T * dst, const int ne0, const int ne01) {
    const int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    const int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned) ne01) {
        const int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        const int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static __global__ void concat_dim2_kernel(const T * x, const T * y, T * dst, const int ne0, const int ne02) {
    const int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    const int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned) ne02) {
        const int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        const int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

template <typename T>
static void concat_cuda(const T * x, const T * y, T * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    const int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    const dim3 grid_dim(num_blocks, ne1, ne2);

    if (dim == 0) {
        concat_dim0_kernel<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_dim1_kernel<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_dim2_kernel<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
concat_non_cont_kernel(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = reinterpret_cast<const T *>(src0 + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
        } else if constexpr (dim == 0) {
            x = reinterpret_cast<const T *>(src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
        } else if constexpr (dim == 1) {
            x = reinterpret_cast<const T *>(src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
        } else if constexpr (dim == 2) {
            x = reinterpret_cast<const T *>(src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
        } else {
            x = reinterpret_cast<const T *>(src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
        }

        T * y = reinterpret_cast<T *>(dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);
        *y = *x;
    }
}

template <typename T>
static void concat_non_cont_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    cudaStream_t stream = ctx.stream();
    const int32_t dim = ((int32_t *) dst->op_params)[0];

    const dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);

    switch (dim) {
        case 0:
            concat_non_cont_kernel<T, 0><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case 1:
            concat_non_cont_kernel<T, 1><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case 2:
            concat_non_cont_kernel<T, 2><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case 3:
            concat_non_cont_kernel<T, 3><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        default:
            GGML_ABORT("Invalid dim: %d", dim);
            break;
    }
}

template <typename T>
static void ggml_cuda_op_concat_typed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    cudaStream_t stream = ctx.stream();
    const int32_t dim = ((int32_t *) dst->op_params)[0];

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const T * src0_d = reinterpret_cast<const T *>(src0->data);
        const T * src1_d = reinterpret_cast<const T *>(src1->data);
        T * dst_d = reinterpret_cast<T *>(dst->data);

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_cuda(
                    src0_d + i3 * (src0->nb[3] / sizeof(T)),
                    src1_d + i3 * (src1->nb[3] / sizeof(T)),
                    dst_d + i3 * (dst->nb[3] / sizeof(T)),
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0], dst->ne[1], dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d, reinterpret_cast<const void *>(src0_d), size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(dst_d) + size0, reinterpret_cast<const void *>(src1_d), size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        concat_non_cont_cuda<T>(ctx, dst);
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);

    switch (ggml_type_size(src0->type)) {
        case 1:
            ggml_cuda_op_concat_typed<uint8_t>(ctx, dst);
            break;
        case 2:
            ggml_cuda_op_concat_typed<uint16_t>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_concat_typed<uint32_t>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_concat_typed<uint64_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported concat type %s (size %zu)", __func__, ggml_type_name(src0->type), ggml_type_size(src0->type));
            break;
    }
}
