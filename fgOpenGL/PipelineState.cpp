// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "PipelineState.hpp"
#include "ShaderObject.hpp"
#include "BackendGL.hpp"
#include "EnumMapping.hpp"
#include <cassert>

using namespace GL;

GLExpected<PipelineState*> PipelineState::create(const FG_PipelineState& state, std::span<FG_Resource*> rendertargets,
                                                 FG_Blend blend, std::span<FG_Resource*> vertexbuffers, GLsizei* strides,
                                                 std::span<FG_ShaderParameter> attributes, FG_Resource* indexbuffer,
                                                 uint8_t indexstride) noexcept
{
  PipelineState* pipeline      = new(rendertargets.size()) PipelineState{};
  pipeline->RenderTargetsCount = rendertargets.size();
  GLuint* rts                  = reinterpret_cast<GLuint*>(pipeline + 1);

  for(int i = 0; i < rendertargets.size(); ++i)
    rts[i] = Buffer(rendertargets[i]).release();

  if(auto e = ProgramObject::create())
    pipeline->program = std::move(e.value());
  else
    return std::move(e.error());

  for(int i = 0; i < FG_ShaderStage_COUNT; ++i)
  {
    if(state.Shaders[i] != nullptr)
      RETURN_ERROR(pipeline->program.attach(ShaderObject(state.Shaders[i])));
  }
  RETURN_ERROR(pipeline->program.link());

  auto vlist =
    reinterpret_cast<std::pair<GLuint, GLsizei>*>(ALLOCA(sizeof(std::pair<GLuint, GLsizei>) * vertexbuffers.size()));

  for(int i = 0; i < vertexbuffers.size(); ++i)
  {
    // Have to be careful to use placement new here to avoid UB due to alloca
    new(&vlist[i]) std::pair{ Buffer(vertexbuffers[i]).release(), strides[i] };
  }

  if(auto e = VertexArrayObject::create(pipeline->program, attributes,
                                        std::span(vlist, vertexbuffers.size()), Buffer(indexbuffer).release()))
  {
    pipeline->vao = std::move(e.value());
  }
  else
    return std::move(e.error());

  pipeline->Members = state.Members;
  pipeline->StencilRef = state.StencilRef;
  pipeline->DepthStencil = Buffer(state.DepthStencil).release();
  Context::ColorFloats(state.BlendFactor, pipeline->BlendFactor, false);

  if(state.DepthFunc >= ArraySize(PrimitiveMapping))
    return GLError(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");

  pipeline->Primitive = PrimitiveMapping[state.Primitive];

  pipeline->Flags   = state.Flags;
  pipeline->SampleMask = state.SampleMask;
  pipeline->StencilReadMask = state.StencilReadMask;
  pipeline->StencilWriteMask = state.StencilWriteMask;

  if(state.DepthFunc >= ArraySize(ComparisonMapping))
    return GLError(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");

  if(state.StencilFunc >= ArraySize(ComparisonMapping))
    return GLError(ERR_INVALID_PARAMETER, "StencilFunc not valid comparison function");

  if(state.StencilFailOp >= ArraySize(StencilOpMapping))
    return GLError(ERR_INVALID_PARAMETER, "StencilFailOp not valid stencil operation");

  if(state.StencilDepthFailOp >= ArraySize(StencilOpMapping))
    return GLError(ERR_INVALID_PARAMETER, "StencilDepthFailOp not valid stencil operation");

  if(state.StencilPassOp >= ArraySize(StencilOpMapping))
    return GLError(ERR_INVALID_PARAMETER, "StencilPassOp not valid stencil operation");


  pipeline->StencilFailOp      = StencilOpMapping[state.StencilFailOp];
  pipeline->StencilDepthFailOp = StencilOpMapping[state.StencilDepthFailOp];
  pipeline->StencilPassOp      = StencilOpMapping[state.StencilPassOp];
  pipeline->StencilFunc        = ComparisonMapping[state.StencilFunc];
  pipeline->DepthFunc          = ComparisonMapping[state.DepthFunc];

  switch(indexstride)
  {
  case 1: pipeline->IndexType = GL_UNSIGNED_BYTE;
  case 2: pipeline->IndexType = GL_UNSIGNED_SHORT;
  case 4: pipeline->IndexType = GL_UNSIGNED_INT;
  case 0:
    if(!indexbuffer)
      break; // an indexstride of 0 is allowed if there is no indexbuffer
  default: return GLError(ERR_INVALID_PARAMETER, "Index stride must be 1, 2, or 4 bytes.");
  }
  pipeline->FillMode         = state.FillMode;
  pipeline->CullMode         = state.CullMode;
  pipeline->DepthBias        = state.DepthBias;
  pipeline->SlopeScaledDepthBias = state.SlopeScaledDepthBias;
  // pipeline->NodeMask          = state.NodeMask; // OpenGL does not support multi-GPU rendering without extensions
  pipeline->blend              = blend;
  // ??? = state.RTFormats // not sure this has an OpenGL equivilent

  return pipeline;
}

GLExpected<std::string> PipelineState::log() const noexcept { return program.log(); }

GLExpected<void> PipelineState::apply(Context* ctx) noexcept
{
  glUseProgram(program);
  GL_ERROR("glUseProgram");
  RETURN_ERROR(vao.bind());

  RETURN_ERROR(ctx->ApplyBlend(blend, BlendFactor, false));
  RETURN_ERROR(ctx->ApplyFlags(Flags, CullMode, FillMode));
  ctx->ApplyPrimitiveShader(Primitive, program, IndexType);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, DepthStencil, 0);
  GL_ERROR("glFramebufferTexture2D");
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, DepthStencil, 0);
  GL_ERROR("glFramebufferTexture2D");

  glDepthFunc(DepthFunc);
  GL_ERROR("glDepthFunc");
  glStencilFunc(StencilFunc, StencilRef, StencilReadMask);
  GL_ERROR("glStencilFunc");
  glStencilOp(StencilFailOp, StencilDepthFailOp, StencilPassOp);
  GL_ERROR("glStencilOp");
  glStencilMask(StencilWriteMask);
  GL_ERROR("glStencilMask");
  glPolygonOffset(SlopeScaledDepthBias, DepthBias);
  GL_ERROR("glPolygonOffset");

  return {};
}

// Stores the current openGL pipeline state in this object
GLExpected<void> current(Context* ctx) noexcept { return {}; }