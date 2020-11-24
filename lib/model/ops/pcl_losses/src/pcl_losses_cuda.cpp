#include<torch/extension.h>
#include <THC/THC.h>
#include<cmath>
#include<vector>

int PCLLossesForwardLaucher(
    const at::Tensor bottom_data, const at::Tensor labels, const at::Tensor cls_loss_weights,
    const at::Tensor pc_labels, const at::Tensor pc_probs, const at::Tensor img_cls_loss_weights,
    const at::Tensor im_labels, const int batch_size, const int channels, 
    const int num_positive, at::Tensor top_data);

int PCLLossesBackwardLaucher(const at::Tensor top_diff, const at::Tensor prob_data, 
    const at::Tensor labels, const at::Tensor cls_loss_weights, const at::Tensor gt_assignment,
    const at::Tensor pc_labels, const at::Tensor pc_probs, const at::Tensor pc_count,
    const at::Tensor img_cls_loss_weights, const at::Tensor im_labels, const int batch_size, 
    const int channels, at::Tensor bottom_diff);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

extern THCState *state;

int pcl_losses_forward_cuda(at::Tensor pcl_probs, at::Tensor labels, 
                            at::Tensor cls_loss_weights, at::Tensor pc_labels,
                            at::Tensor pc_probs, at::Tensor img_cls_loss_weights, 
                            at::Tensor im_labels, at::Tensor output)
{
    // Grab the input tensor
    CHECK_INPUT(pcl_probs);
    CHECK_INPUT(labels);
    CHECK_INPUT(cls_loss_weights);
    CHECK_INPUT(pc_labels);
    CHECK_INPUT(pc_probs);
    CHECK_INPUT(img_cls_loss_weights);
    CHECK_INPUT(im_labels);
    CHECK_INPUT(output);

    int batch_size = pcl_probs.size(0);
    int channels = pcl_probs.size(1);
    int num_positive = pc_labels.size(1);

    PCLLossesForwardLaucher(
        pcl_probs, labels, cls_loss_weights,
        pc_labels, pc_probs, img_cls_loss_weights, 
        im_labels, batch_size, channels, num_positive, 
        output);

    return 1;
}

int pcl_losses_backward_cuda(at::Tensor pcl_probs, at::Tensor labels, 
                             at::Tensor cls_loss_weights, at::Tensor gt_assignment,
                             at::Tensor pc_labels, at::Tensor pc_probs, 
                             at::Tensor pc_count, at::Tensor img_cls_loss_weights, 
                             at::Tensor im_labels, at::Tensor top_grad, 
                             at::Tensor bottom_grad)
{
    // Grab the input tensor
    CHECK_INPUT(pcl_probs);
    CHECK_INPUT(labels);
    CHECK_INPUT(cls_loss_weights);
    CHECK_INPUT(gt_assignment);
    CHECK_INPUT(pc_labels);
    CHECK_INPUT(pc_probs);
    CHECK_INPUT(pc_count);
    CHECK_INPUT(img_cls_loss_weights);
    CHECK_INPUT(im_labels);
    CHECK_INPUT(top_grad);
    CHECK_INPUT(bottom_grad);

    int batch_size = pcl_probs.size(0);
    int channels = pcl_probs.size(1);
    PCLLossesBackwardLaucher(
        top_grad, pcl_probs, labels, cls_loss_weights,
        gt_assignment, pc_labels, pc_probs, pc_count,
        img_cls_loss_weights, im_labels, batch_size, channels,
        bottom_grad);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcl_losses_forward_cuda, "pcl_losses forward (CUDA)");
  m.def("backward", &pcl_losses_backward_cuda, "pcl_losses backward (CUDA)");
}