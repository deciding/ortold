// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

class QLinearMatMul final : public OpKernel {
 public:
  QLinearMatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

ONNX_OPERATOR_KERNEL_EX(
    QLinearMatMul,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearMatMul);

Status QLinearMatMul::Compute(OpKernelContext* ctx) const {
  const auto* a = ctx->Input<Tensor>(0);
  const auto* b = ctx->Input<Tensor>(3);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  // validate offsets
  const auto* a_offset = ctx->Input<Tensor>(2);
  const auto* b_offset = ctx->Input<Tensor>(5);
  const auto* y_offset = ctx->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_offset),
              "QLinearMatmul : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_offset),
              "QLinearMatmul : weight zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_offset),
              "QLinearMatmul : result zero point must be a scalar or 1D tensor of size 1");

  // validate scale
  const auto* a_scale = ctx->Input<Tensor>(1);
  const auto* b_scale = ctx->Input<Tensor>(4);
  const auto* y_scale = ctx->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale),
              "QLinearMatmul : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale),
              "QLinearMatmul : weight scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(y_scale),
              "QLinearMatmul : result scale must be a scalar or 1D tensor of size 1");

  auto a_scale_data = *(a_scale->template Data<float>());
  auto b_scale_data = *(b_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (a_scale_data * b_scale_data) / y_scale_data;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) *
                                       static_cast<size_t>(helper.M()) * static_cast<size_t>(helper.N()));
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());

  MLAS_GEMM_U8X8_PARAMETERS gemm_params;
  gemm_params.M = static_cast<size_t>(helper.M());
  gemm_params.N = static_cast<size_t>(helper.N());
  gemm_params.K = static_cast<size_t>(helper.K());
  gemm_params.lda = gemm_params.K;
  gemm_params.ZeroPointA = *a_offset->template Data<uint8_t>();
  gemm_params.ldb = gemm_params.N;
  gemm_params.ZeroPointB = static_cast<const uint8_t*>(b_offset->DataRaw());
  gemm_params.BIsSigned = b->IsDataType<int8_t>();
  gemm_params.C = gemm_output;
  gemm_params.ldc = gemm_params.N;

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {
    gemm_params.A = a->template Data<uint8_t>() + helper.LeftOffsets()[i];
    gemm_params.B = static_cast<const uint8_t*>(b->DataRaw()) + helper.RightOffsets()[i];
    MlasGemm(&gemm_params, ctx->GetOperatorThreadPool());

    MlasRequantizeOutput(gemm_output,
                         y->template MutableData<uint8_t>() + helper.OutputOffsets()[i],
                         nullptr,
                         static_cast<size_t>(helper.M()),
                         static_cast<size_t>(helper.N()),
                         &real_multiplier,
                         false,
                         *y_offset->template Data<uint8_t>());
  }

  return Status::OK();
}

}  // namespace onnxruntime
