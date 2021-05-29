#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cstdlib>

#include <torch/extension.h>
#include <core/graph/constants.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>

#include "common.h"

// declaration copied from DCNv2/src/dcn_v2_decl.h
at::Tensor
dcn_v2_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const int kernel_h,
               const int kernel_w,
               const int stride_h,
               const int stride_w,
               const int pad_h,
               const int pad_w,
               const int dilation_h,
               const int dilation_w,
               const int deformable_group);

namespace {

struct CustomKernelDCNv2 {
  CustomKernelDCNv2(Ort::CustomOpApi ort, const OrtKernelInfo* info, void* compute_stream)
      : ort_(ort), compute_stream_(compute_stream) {

#define DECLARE_INFO_ATTRIBUTE(name) \
    int name##_h_, name##_w_
#define GET_INPUT_VALUE(name) \
    name##_h_ = std::stoi(ort_.KernelInfoGetAttribute<std::string>(info, #name "_h")); \
    name##_w_ = std::stoi(ort_.KernelInfoGetAttribute<std::string>(info, #name "_w"));

    kernel_h_ = kernel_w_ = 3;
    GET_INPUT_VALUE(stride)
    GET_INPUT_VALUE(padding)
    GET_INPUT_VALUE(dilation)
    deformable_group_ = std::stoi(ort_.KernelInfoGetAttribute<std::string>(info, "deformable_groups"));
  }
private:
  DECLARE_INFO_ATTRIBUTE(kernel);
  DECLARE_INFO_ATTRIBUTE(stride);
  DECLARE_INFO_ATTRIBUTE(padding);
  DECLARE_INFO_ATTRIBUTE(dilation);
  int deformable_group_;

public:

  void Compute(OrtKernelContext* context) {
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_offset = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* input_mask = ort_.KernelContext_GetInput(context, 2);
    const OrtValue* input_weight = ort_.KernelContext_GetInput(context, 3);
    const OrtValue* input_bias = ort_.KernelContext_GetInput(context, 4);

#ifdef USE_CUDA
#define WRAP_ATEN_TENSOR(name) \
    at::Tensor ATen_##name = at::from_blob( \
    (void*) ort_.GetTensorData<float>(name),\
    ort_.GetTensorShape(ort_.GetTensorTypeAndShape(name)), \
    at::kCUDA)
#else
#define WRAP_ATEN_TENSOR(name) \
    at::Tensor ATen_##name = at::from_blob( \
    (void*) ort_.GetTensorData<float>(name),\
    ort_.GetTensorShape(ort_.GetTensorTypeAndShape(name)) \
    )
#endif

    WRAP_ATEN_TENSOR(input);
    WRAP_ATEN_TENSOR(input_weight);
    WRAP_ATEN_TENSOR(input_bias);
    WRAP_ATEN_TENSOR(input_offset);
    WRAP_ATEN_TENSOR(input_mask);

    at::Tensor ATen_output = dcn_v2_forward(
        ATen_input,
        ATen_input_weight,
        ATen_input_bias,
        ATen_input_offset,
        ATen_input_mask,
        kernel_h_, kernel_w_,
        stride_h_, stride_w_,
        padding_h_, padding_w_,
        dilation_h_, dilation_w_,
        deformable_group_
    );

    auto sizes = ATen_output.sizes();
    long total_element_count = 1;
    for (long size : sizes) {
      total_element_count *= size;
    }
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, sizes.data(), sizes.size());

    // TODO: Maybe we should modify dcn_v2_xxx_forward to wrap tensors to avoid additional copy
#ifdef USE_CUDA
    cudaMemcpy(ort_.GetTensorMutableData<float>(output),
               ATen_output.data_ptr<float>(),
               sizeof(float) * total_element_count,
               cudaMemcpyDeviceToDevice);
#else
    std::memcpy(ort_.GetTensorMutableData<float>(output), ATen_output.data_ptr<float>(), sizeof(float) * total_element_count);
#endif
  }

private:
  Ort::CustomOpApi ort_;
  void* compute_stream_;
};

struct CustomOpDCNv2 : Ort::CustomOpBase<CustomOpDCNv2, CustomKernelDCNv2> {
  CustomOpDCNv2(const char* provider, cudaStream_t stream) : provider_(provider), compute_stream_(stream) {}
  void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const {
    return new CustomKernelDCNv2(api, info, compute_stream_);
  }
  const char* GetName() const { return "DCNv2"; }
  const char* GetExecutionProviderType() const { return provider_; }

  std::size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(std::size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

  std::size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(std::size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

private:
  const char* provider_;
  void* compute_stream_;
};

}

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;

#ifdef USE_CUDA
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  CustomOpDCNv2 custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};
#else
  CustomOpDCNv2 custom_op{onnxruntime::kCpuExecutionProvider, nullptr};
#endif

  Ort::CustomOpDomain custom_op_domain("yolact");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions options;
  options.Add(custom_op_domain);

  Ort::Env env;

#ifdef USE_CUDA
  std::cout << "Running with CUDA provider" << std::endl;
  OrtCUDAProviderOptions cuda_options{
      0,
      OrtCudnnConvAlgoSearch::EXHAUSTIVE,
      std::numeric_limits<size_t>::max(),
      0,
      true,
  };
  options.AppendExecutionProvider_CUDA(cuda_options);
#else
  std::cout << "Running with default provider" << std::endl;
#endif

  Ort::Session session{env, "../yolact/yolact_plus.onnx", options};

  cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat input_image_float;
  input_image.convertTo(input_image_float, CV_32F);
  cv::Mat resized_image;
  cv::resize(input_image_float, resized_image, cv::Size(550, 550));

//#ifdef USE_CUDA
  // FIXME: GPU allocation causes Conv to fail during session.Run (CUDA failure 700: an illegal memory access was ...)
  // Use an allocator to CreateTensor causes GPU memory allocation,
  // for performance, we should move normalization after CreateTensor, and normalize with GPU when defined(USE_CUDA)
//  Ort::MemoryInfo memory_info("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
//  Ort::Allocator allocator(session, memory_info);
//  Ort::Value input_value = Ort::Value::CreateTensor<float>(allocator, input.ptr<float>(), count, input_size, 4);
//#else
  std::vector<cv::Mat> channels;
  cv::split(input_image_float, channels);
  for (int i = 0; i < 3; ++i) {
    channels[i] -= MEANS[i];
    channels[i] /= STD[i];
  }
  cv::Mat normalized;
  cv::merge(channels, normalized);
  cv::Mat input = cv::dnn::blobFromImage(normalized, 1, cv::Size(550, 550));

  std::int64_t input_size[4] = {1, 3, 550, 550};
  size_t count = input_size[0] * input_size[1] * input_size[2] * input_size[3];
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // CreateTensor(OrtMemoryInfo*, ....) create tensor with existing data on CPU without copying
  Ort::Value input_value = Ort::Value::CreateTensor<float>(memory_info, input.ptr<float>(), count, input_size, 4);
//#endif

  const char* input_names[] = {"image"};
  const char* output_names[] = {"loc", "conf", "mask", "proto"};

  auto&& results = session.Run(Ort::RunOptions{nullptr}, input_names, &input_value, 1, output_names, 4);

  auto locMat_size = results[0].GetTensorTypeAndShapeInfo().GetShape();
  cv::Mat locMat(locMat_size[1], locMat_size[2], CV_32F, results[0].GetTensorMutableData<float>());
  auto confMat_size = results[1].GetTensorTypeAndShapeInfo().GetShape();
  cv::Mat confMat(confMat_size[1], confMat_size[2], CV_32F, results[1].GetTensorMutableData<float>());
  auto maskMat_size = results[2].GetTensorTypeAndShapeInfo().GetShape();
  cv::Mat maskMat(maskMat_size[1], maskMat_size[2], CV_32F, results[2].GetTensorMutableData<float>());
  cv::Mat protoMat(138 * 138, 32, CV_32F, results[3].GetTensorMutableData<float>());

  cv::Mat priorsMat(NET_PROP, 32, CV_32F);
  std::ifstream ifs("../yolact/priors_plus.dat", std::ios::binary);
  for (int i = 0; i < NET_PROP; ++i) {
    for (int j = 0; j < 4; ++j) {
      ifs.read(reinterpret_cast<char*>(priorsMat.ptr<float>(i, j)), sizeof(float));
    }
  }

  // decode
  auto boxes = decode(locMat, priorsMat);
  // detect
  auto detections = detect(confMat, boxes, maskMat);
  // nms
  auto filtered_detections = traditional_nms(detections, maskMat);
  for (auto& det: filtered_detections) std::cout << det << std::endl;
  // post-process masks
  postprocess(filtered_detections, protoMat);

  auto masked_image = draw_masks(input_image_float / 255, filtered_detections);

  cv::imshow("original", input_image_float / 255);
  cv::imshow("masked", masked_image);
  cv::waitKey();

#ifdef USE_CUDA
  cudaStreamDestroy(compute_stream);
#endif

  return 0;
}
