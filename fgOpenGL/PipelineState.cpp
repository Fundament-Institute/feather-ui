// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.h"

#include "PipelineState.h"
#include "ShaderObject.h"
#include "BackendGL.h"
#include <assert.h>
#include <malloc.h>

using namespace GL;

GLExpected<PipelineState*> PipelineState::create(const FG_PipelineState& state, std::span<FG_Resource*> rendertargets,
                                                 FG_Blend blend, std::span<FG_Resource*> vertexbuffers, GLsizei* strides,
                                                 std::span<FG_ShaderParameter> attributes,
                                                 FG_Resource* indexbuffer) noexcept
{
  PipelineState* pipeline      = new(rendertargets.size()) PipelineState{};
  pipeline->RenderTargetsCount = rendertargets.size();
  GLuint* rts                  = reinterpret_cast<GLuint*>(pipeline + 1);

  for(int i = 0; i < rendertargets.size(); ++i)
    rts[i] = Buffer(rendertargets[i]).take();

  for(int i = 0; i < FG_ShaderStage_COUNT; ++i)
  {
    if(state.Shaders[i] != nullptr)
      if(auto e = pipeline->program.attach(ShaderObject(state.Shaders[i]))) {}
      else
        return std::move(e.error());
  }
  if(auto e = pipeline->program.link()) {}
  else
    return std::move(e.error());

  if(state.Shaders[FG_ShaderStage_Vertex] == nullptr)
    return GLError(ERR_INVALID_PARAMETER, "Can't have a null vertex shader on graphics pipeline!");

  auto vlist =
    reinterpret_cast<std::pair<GLuint, GLsizei>*>(_alloca(sizeof(std::pair<GLuint, GLsizei>) * vertexbuffers.size()));

  for(int i = 0; i < vertexbuffers.size(); ++i)
  {
    vlist[i] = { Buffer(vertexbuffers[i]).take(), strides[i] };
  }

  if(auto e = VertexArrayObject::create(ShaderObject(state.Shaders[FG_ShaderStage_Vertex]).take(), attributes,
                                        std::span(vlist, vertexbuffers.size()), Buffer(indexbuffer).take()))
  {
    pipeline->vao = std::move(e.value());
  }
  else
    return std::move(e.error());

  pipeline->Members = state.Members;
  pipeline->StencilRef = state.StencilRef;
  pipeline->DepthStencil = Buffer(state.DepthStencil).take();
  Context::ColorFloats(state.BlendFactor, pipeline->BlendFactor, false);
  pipeline->Primitive = state.Primitive;

  pipeline->Flags   = state.Flags;
  pipeline->SampleMask = state.SampleMask;
  pipeline->StencilReadMask = state.StencilReadMask;
  pipeline->StencilWriteMask   = state.StencilWriteMask;
  pipeline->StripCutValue      = state.StripCutValue;
  pipeline->DepthFunc        = state.DepthFunc;
  pipeline->FillMode         = state.FillMode;
  pipeline->CullMode         = state.CullMode;
  pipeline->DepthBias        = state.DepthBias;
  pipeline->DepthBiasClamp   = state.DepthBiasClamp;
  pipeline->ForcedSampleCount = state.ForcedSampleCount;
  pipeline->NodeMask          = state.NodeMask;
  pipeline->MultisampleCount  = state.MultisampleCount;
  pipeline->MultisampleQuality = state.MultisampleQuality;
  pipeline->blend              = blend;
  // ??? = state.RTFormats // not sure this has an OpenGL equivilent

  return pipeline;
}

GLExpected<std::string> PipelineState::log() const noexcept { return program.log(); }


GLExpected<void> PipelineState::apply(Context* ctx) noexcept
{
  glUseProgram(program);
  GL_ERROR("glUseProgram");
  vao.bind(); // should throw error for not checking return value

  ctx->ApplyBlend(blend, BlendFactor, false);
  ctx->ApplyFlags(Flags, CullMode, FillMode);
  
    FG_PIPELINE_FLAG_ALPHA_TO_COVERAGE_ENABLE = (1 << 0),
    FG_PIPELINE_FLAG_INDEPENDENT_BLEND_ENABLE = (1 << 1),
    FG_PIPELINE_FLAG_DEPTH_WRITE_ENABLE       = (1 << 4),
    FG_PIPELINE_FLAG_CONSERVATIVE_RASTER      = (1 << 5),
    FG_PIPELINE_FLAG_FRONT_COUNTER_CLOCKWISE  = (1 << 6),
    FG_PIPELINE_FLAG_DEPTH_CLIP_ENABLE        = (1 << 7),
    FG_PIPELINE_FLAG_MULTISAMPLE_ENABLE       = (1 << 8),
    FG_PIPELINE_FLAG_ANTIALIASED_LINE_ENABLE  = (1 << 9),
    FG_PIPELINE_FLAG_TOOL_DEBUG               = (1 << 10),

}