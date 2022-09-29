// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__PIPELINE_STATE_H
#define GL__PIPELINE_STATE_H

#include "ProgramObject.hpp"
#include "VertexArrayObject.hpp"
#include "FrameBuffer.hpp"
#include "Buffer.hpp"
#include "feather/graphics_interface.h"
#include <string>
#include <span>
#include <array>

namespace GL {
  struct Context;

  struct PipelineState
  {
    // This mostly inherits the standard backend pipeline state, but translates things into OpenGL equivalents
    uint64_t Members;
    Owned<ProgramObject> program;
    std::array<float, 4> BlendFactor;
    uint32_t SampleMask;
    GLint StencilRef;
    uint8_t StencilReadMask;
    uint8_t StencilWriteMask;
    uint8_t StencilFailOp;
    uint8_t StencilDepthFailOp;
    uint8_t StencilPassOp;
    uint8_t StencilFunc;
    uint8_t DepthFunc;
    uint8_t FillMode;
    uint8_t CullMode;
    uint8_t Primitive;
    uint16_t IndexType;
    uint16_t Flags;
    int DepthBias;
    float SlopeScaledDepthBias;
    uint32_t ForcedSampleCount;
    FG_Blend blend;
    VertexArrayObject vao;
    Framebuffer rt;

    GLExpected<void> apply(Context* ctx) noexcept;
    GLExpected<std::string> log() const noexcept;
    GLExpected<void> current(Context* ctx) noexcept;

    PipelineState(const PipelineState&)  = delete;
    PipelineState(PipelineState&& state) = default;
    PipelineState& operator=(const PipelineState&) = delete;
    PipelineState& operator=(PipelineState&& right) noexcept = default;

    static GLExpected<PipelineState*> create(const FG_PipelineState& state, FG_Resource rendertarget, FG_Blend blend,
                                             std::span<FG_Resource> vertexbuffers, GLsizei* strides,
                                             std::span<FG_VertexParameter> attributes, FG_Resource indexbuffer,
                                             uint8_t indexstride) noexcept;

  private:
#pragma warning(push)
#pragma warning(disable : 26495)
    PipelineState() {}
#pragma warning(pop)
  };

  static uint64_t COMPUTE_PIPELINE_FLAG = (1ULL << 63);
  struct ComputePipelineState
  {
    uint64_t Members;
    FG_Vec3i workgroup;
    uint32_t flags;
    ProgramObject program;

    static GLExpected<ComputePipelineState*> create(FG_Shader computeshader, FG_Vec3i workgroup, uint32_t flags) noexcept;
    GLExpected<void> apply(Context* ctx) noexcept;
    GLExpected<std::string> log() const noexcept;
  };
}

#endif
