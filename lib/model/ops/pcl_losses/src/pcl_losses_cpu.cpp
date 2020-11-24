#include<TH/TH.h>
#include<math.h>
#include<torch/extension.h>

template <typename scalar_t>
void pcl_losses_forward_cpu_kernel(const scalar_t* pcl_probs, const scalar_t* labels, 
                       const scalar_t * cls_loss_weights, const scalar_t * pc_labels,
                       const scalar_t * pc_probs, const scalar_t * img_cls_loss_weights, 
                       const scalar_t * im_labels, scalar_t * output, int batch_size, int channels, int num_positive)
{
    // Grab the input tensor

    float eps = 1e-6;

    for (int c = 0; c < channels; c++) {
        output[c] = 0;
        if (im_labels[c] != 0) {
            if (c == 0) {
                for (int i = 0; i < batch_size; i++) {
                    if (labels[i] == 0) {
                        output[c] -= cls_loss_weights[i] * log(fmaxf(pcl_probs[i * channels + c], eps));
                    }
                }
            }
            else {
                for (int i = 0; i < num_positive; i++) {
                    if (pc_labels[i] == c) {
                        output[c] -= img_cls_loss_weights[i] * log(fmaxf(pc_probs[i], eps));
                    }
                }
            }
        }
    }
}

int pcl_losses_forward(const at::Tensor & pcl_probs, const at::Tensor & labels, 
                       const at::Tensor & cls_loss_weights, const at::Tensor & pc_labels,
                       const at::Tensor & pc_probs, const at::Tensor & img_cls_loss_weights, 
                       const at::Tensor & im_labels, at::Tensor & output)
{
    AT_ASSERTM(!pcl_probs.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!labels.type().is_cuda(), "input must be a CPU tensor");    
    AT_ASSERTM(!cls_loss_weights.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!pc_labels.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!pc_probs.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!img_cls_loss_weights.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!im_labels.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!output.type().is_cuda(), "input must be a CPU tensor");  

    int batch_size = pcl_probs.size(0);
    int channels = pcl_probs.size(1);
    int num_positive = pc_labels.size(1);

    AT_DISPATCH_FLOATING_TYPES(pcl_probs.scalar_type(), "PCLLosses_forward", [&]    {
        pcl_losses_forward_cpu_kernel<scalar_t>(
            pcl_probs.data<scalar_t>(), 
            labels.data<scalar_t>(), 
            cls_loss_weights.data<scalar_t>(), 
            pc_labels.data<scalar_t>(),
            pc_probs.data<scalar_t>(), 
            img_cls_loss_weights.data<scalar_t>(), 
            im_labels.data<scalar_t>(), 
            output.data<scalar_t>(), batch_size, channels, num_positive);
    });
    return 1;
}

template <typename scalar_t>
void pcl_losses_backward_cpu_kernel(const scalar_t* pcl_probs, const scalar_t* labels, 
                        const scalar_t* cls_loss_weights, const scalar_t* gt_assignment,
                        const scalar_t* pc_labels, const scalar_t* pc_probs, 
                        const scalar_t* pc_count, const scalar_t* img_cls_loss_weights, 
                        const scalar_t* im_labels, const scalar_t* top_grad, 
                        scalar_t* bottom_grad, int batch_size, int channels)
{
    // Grab the input tensor
    float eps = 1e-5;

    for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < channels; c++) {
            bottom_grad[i * channels + c] = 0;
            if (im_labels[c] != 0) {
                if (c == 0) {
                    if (labels[i] == 0) {
                        bottom_grad[i * channels + c] = -cls_loss_weights[i] 
                            / fmaxf(pcl_probs[i * channels + c], eps);
                    }
                }
                else {
                    if (labels[i] == c) {
                        int pc_index = gt_assignment[i];
                        if (c != pc_labels[pc_index]) {
                            printf("labels mismatch.\n");
                        }
                        bottom_grad[i * channels + c] = -img_cls_loss_weights[pc_index] 
                            / fmaxf(pc_count[pc_index] * pc_probs[pc_index], eps);
                    }
                }
            }
        }
    }
}

int pcl_losses_backward(const at::Tensor & pcl_probs, const at::Tensor & labels, 
                        const at::Tensor & cls_loss_weights, const at::Tensor & gt_assignment,
                        const at::Tensor & pc_labels, const at::Tensor & pc_probs, 
                        const at::Tensor & pc_count, const at::Tensor & img_cls_loss_weights, 
                        const at::Tensor & im_labels, const at::Tensor & top_grad, 
                        at::Tensor & bottom_grad)
{
    AT_ASSERTM(!pcl_probs.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!labels.type().is_cuda(), "input must be a CPU tensor");    
    AT_ASSERTM(!cls_loss_weights.type().is_cuda(), "input must be a CPU tensor"); 
    AT_ASSERTM(!gt_assignment.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!pc_labels.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!pc_probs.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!pc_count.type().is_cuda(), "input must be a CPU tensor"); 
    AT_ASSERTM(!img_cls_loss_weights.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!im_labels.type().is_cuda(), "input must be a CPU tensor");  
    AT_ASSERTM(!top_grad.type().is_cuda(), "input must be a CPU tensor");
    AT_ASSERTM(!bottom_grad.type().is_cuda(), "input must be a CPU tensor"); 

    int batch_size = pcl_probs.size(0);
    int channels = pcl_probs.size(1);

    AT_DISPATCH_FLOATING_TYPES(top_grad.type(), "PCLLosses_backword", [&] {
        pcl_losses_backward_cpu_kernel<scalar_t>(
        pcl_probs.data<scalar_t>(), 
        labels.data<scalar_t>(), 
        cls_loss_weights.data<scalar_t>(), 
        gt_assignment.data<scalar_t>(), 
        pc_labels.data<scalar_t>(), 
        pc_probs.data<scalar_t>(), 
        pc_count.data<scalar_t>(), 
        img_cls_loss_weights.data<scalar_t>(),
        im_labels.data<scalar_t>(), 
        top_grad.data<scalar_t>(), 
        bottom_grad.data<scalar_t>(), batch_size, channels);
    });
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcl_losses_forward, "pcl_losses forward (CPU)");
  m.def("backward", &pcl_losses_backward, "pcl_losses backward (CPU)");
}