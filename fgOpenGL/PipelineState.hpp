// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.hpp"

#ifndef GL__PIPELINE_STATE_H
#define GL__PIPELINE_STATE_H

#include "ProgramObject.hpp"
#include "VertexArrayObject.hpp"
#include "FrameBuffer.hpp"
#include "Buffer.hpp"
#include "feather/backend.h"
#include <string>
#include <span>
#include <array>

namespace GL {
  struct Context;

  struct PipelineState
  {
    // This mostly inherits the standard backend pipeline state, but translates things into OpenGL equivalents
    uint64_t Members;
    GLuint DepthStencil;
    ProgramObject program;
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
    uint8_t RenderTargetsCount;
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
    GLuint rt = 0;

    GLExpected<void> apply(Context* ctx) noexcept;
    GLExpected<std::string> log() const noexcept;
    GLExpected<void> current(Context* ctx) noexcept;

    PipelineState(const PipelineState&)  = delete;
    PipelineState(PipelineState&& state) = default;
    PipelineState& operator=(const PipelineState&) = delete;
    PipelineState& operator=(PipelineState&& right) noexcept = default;

    static GLExpected<PipelineState*> create(const FG_PipelineState& state, std::span<FG_Resource*> rendertargets,
                                             FG_Blend blend, std::span<FG_Resource*> vertexbuffers, GLsizei* strides,
                                             std::span<FG_ShaderParameter> attributes, FG_Resource* indexbuffer,
                                             uint8_t indexstride) noexcept;

    static void* operator new(std::size_t base, std::size_t renderTargetCount)
    {
      return ::operator new(base + sizeof(GLuint) * renderTargetCount);
    }

    static void operator delete(void* base) { return ::operator delete(base); }

  private:
#pragma warning(push)
#pragma warning(disable : 26495)
    PipelineState() {}
#pragma warning(pop)
  };
}

#endif
