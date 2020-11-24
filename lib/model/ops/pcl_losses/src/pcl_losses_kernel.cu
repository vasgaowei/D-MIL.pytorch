#include<ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void PCLLossesForward(const int nthreads, const scalar_t* bottom_data, 
    const scalar_t* labels, const scalar_t* cls_loss_weights, const scalar_t* pc_labels,
    const scalar_t* pc_probs, const scalar_t* img_cls_loss_weights, 
    const scalar_t* im_labels, const int batch_size, const int num_positive, scalar_t* top_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        top_data[index] = 0;
        if (im_labels[index] != 0) {
            if (index == 0) {
                for (int i = 0; i < batch_size; i++) {
                    if (labels[i] == 0) {
                        top_data[index] -= cls_loss_weights[i] * log(bottom_data[i * nthreads + index]);
                    }
                }
            }
            else {
                for (int i = 0; i < num_positive; i++) {
                    if (pc_labels[i] == index) {
                        top_data[index] -= img_cls_loss_weights[i] * log(pc_probs[i]);
                    }
                }
            }
        }
    }
}

int PCLLossesForwardLaucher(
    const at::Tensor bottom_data, const at::Tensor labels, const at::Tensor cls_loss_weights,
    const at::Tensor pc_labels, const at::Tensor pc_probs, const at::Tensor img_cls_loss_weights,
    const at::Tensor im_labels, const int batch_size, const int channels, 
    const int num_positive, at::Tensor top_data)
{
    const int kThreadsPerBlock = 4;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.scalar_type(), "PCLLosses_forward", (
        [&] {
        PCLLossesForward<scalar_t>
        <<<(channels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
            channels, 
            bottom_data.contiguous().data<scalar_t>(), 
            labels.contiguous().data<scalar_t>(), 
            cls_loss_weights.contiguous().data<scalar_t>(), 
            pc_labels.contiguous().data<scalar_t>(), 
            pc_probs.contiguous().data<scalar_t>(), 
            img_cls_loss_weights.contiguous().data<scalar_t>(),
            im_labels.contiguous().data<scalar_t>(), 
            batch_size, 
            num_positive, 
            top_data.contiguous().data<scalar_t>());
    }
    ));
    THCudaCheck(cudaGetLastError());
    return 1;
}

template <typename scalar_t>
__global__ void PCLLossesBackward(const int nthreads, const scalar_t* prob_data, 
    const scalar_t* labels, const scalar_t* cls_loss_weights, const scalar_t* gt_assignment,
    const scalar_t* pc_labels, const scalar_t* pc_probs, const scalar_t* pc_count,
    const scalar_t* img_cls_loss_weights, const scalar_t* im_labels, const int channels, 
    scalar_t* bottom_diff) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        int i = index / channels;
        int c = index % channels;
        bottom_diff[index] = 0;

        if (im_labels[c] != 0) {
            if (c == 0) {
                if (labels[i] == 0) {
                    bottom_diff[index] = -cls_loss_weights[i] / prob_data[index];
                }
            }
            else {
                if (labels[i] == c) {
                    int pc_index = gt_assignment[i];
                    if (c != pc_labels[pc_index]) {
                        printf("labels mismatch.\n");
                    }
                    bottom_diff[index] = -img_cls_loss_weights[pc_index]
                        / (pc_count[pc_index] * pc_probs[pc_index]);
                }
            }
        }
    }
}

int PCLLossesBackwardLaucher(const at::Tensor top_diff, const at::Tensor prob_data, 
    const at::Tensor labels, const at::Tensor cls_loss_weights, const at::Tensor gt_assignment,
    const at::Tensor pc_labels, const at::Tensor pc_probs, const at::Tensor pc_count,
    const at::Tensor img_cls_loss_weights, const at::Tensor im_labels, const int batch_size, 
    const int channels, at::Tensor bottom_diff)
{
    const int kThreadsPerBlock = 16;
    auto output_size = batch_size * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.scalar_type(), "ROIPool_backward", (
        [&]{
        PCLLossesBackward<scalar_t>
        <<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
            output_size,
            prob_data.contiguous().data<scalar_t>(), 
            labels.contiguous().data<scalar_t>(), 
            cls_loss_weights.contiguous().data<scalar_t>(), 
            gt_assignment.contiguous().data<scalar_t>(), 
            pc_labels.contiguous().data<scalar_t>(),
            pc_probs.contiguous().data<scalar_t>(), 
            pc_count.contiguous().data<scalar_t>(),
            img_cls_loss_weights.contiguous().data<scalar_t>(), 
            im_labels.contiguous().data<scalar_t>(), 
            channels, 
            bottom_diff.contiguous().data<scalar_t>());
    }
    ));
    THCudaCheck(cudaGetLastError());

    return 1;
}