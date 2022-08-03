// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "PipelineState.hpp"
#include "ShaderObject.hpp"
#include "BackendGL.hpp"
#include "EnumMapping.hpp"
#include <cassert>
#include <malloc.h> // for _alloca on windows
#include <unordered_set>

using namespace GL;

GLExpected<PipelineState*> PipelineState::create(const FG_PipelineState& state, FG_Resource rendertarget, FG_Blend blend,
                                                 std::span<FG_Resource> vertexbuffers, GLsizei* strides,
                                                 std::span<FG_VertexParameter> attributes, FG_Resource indexbuffer,
                                                 uint8_t indexstride) noexcept
{
  PipelineState* pipeline      = new PipelineState{};

  if(auto e = ProgramObject::create())
    pipeline->program = std::move(e.value());
  else
    return std::move(e.error());

  // for(int i = 0; i < FG_ShaderStage_COUNT; ++i)
  for(auto shader : std::span(state.Shaders))
  {
    if(shader != Backend::NULL_SHADER)
      RETURN_ERROR(pipeline->program.attach(ShaderObject(shader)));
  }
  RETURN_ERROR(pipeline->program.link());

  auto vlist = std::span(reinterpret_cast<std::pair<GLuint, GLsizei>*>(
                           ALLOCA(sizeof(std::pair<GLuint, GLsizei>) * vertexbuffers.size())),
                         vertexbuffers.size());

  for(size_t i = 0; i < vlist.size(); ++i)
  {
    // Have to be careful to use placement new here to avoid UB due to alloca
    new(&vlist[i]) std::pair{ Buffer(vertexbuffers[i]), strides[i] };
  }

  if(auto e = VertexArrayObject::create(pipeline->program, attributes, vlist, Buffer(indexbuffer)))
  {
    pipeline->vao = std::move(e.value());
  }
  else
    return std::move(e.error());

  pipeline->Members      = state.Members;
  pipeline->rt         = FrameBuffer(rendertarget);
  pipeline->StencilRef = state.StencilRef;
  Context::ColorFloats(state.BlendFactor, pipeline->BlendFactor, false);

  if(state.DepthFunc >= ArraySize(PrimitiveMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");

  pipeline->Primitive = PrimitiveMapping[state.Primitive];

  pipeline->Flags            = state.Flags;
  pipeline->SampleMask       = state.SampleMask;
  pipeline->StencilReadMask  = state.StencilReadMask;
  pipeline->StencilWriteMask = state.StencilWriteMask;

  if(state.DepthFunc >= ArraySize(ComparisonMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");

  if(state.StencilFunc >= ArraySize(ComparisonMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilFunc not valid comparison function");

  if(state.StencilFailOp >= ArraySize(StencilOpMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilFailOp not valid stencil operation");

  if(state.StencilDepthFailOp >= ArraySize(StencilOpMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilDepthFailOp not valid stencil operation");

  if(state.StencilPassOp >= ArraySize(StencilOpMapping))
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilPassOp not valid stencil operation");

  pipeline->StencilFailOp      = StencilOpMapping[state.StencilFailOp];
  pipeline->StencilDepthFailOp = StencilOpMapping[state.StencilDepthFailOp];
  pipeline->StencilPassOp      = StencilOpMapping[state.StencilPassOp];
  pipeline->StencilFunc        = ComparisonMapping[state.StencilFunc];
  pipeline->DepthFunc          = ComparisonMapping[state.DepthFunc];

  switch(indexstride)
  {
  case 1: pipeline->IndexType = GL_UNSIGNED_BYTE; break;
  case 2: pipeline->IndexType = GL_UNSIGNED_SHORT; break;
  case 4: pipeline->IndexType = GL_UNSIGNED_INT; break;
  case 0:
    if(!indexbuffer)
      break; // an indexstride of 0 is allowed if there is no indexbuffer
  default: return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Index stride must be 1, 2, or 4 bytes.");
  }
  pipeline->FillMode             = state.FillMode;
  pipeline->CullMode             = state.CullMode;
  pipeline->DepthBias            = state.DepthBias;
  pipeline->SlopeScaledDepthBias = state.SlopeScaledDepthBias;
  // pipeline->NodeMask          = state.NodeMask; // OpenGL does not support multi-GPU rendering without extensions
  pipeline->blend = blend;

  // glViewport(0, 0, 800, 600);
  return pipeline;
}

GLExpected<std::string> PipelineState::log() const noexcept { return program.log(); }

GLExpected<void> PipelineState::apply(Context* ctx) noexcept
{
  RETURN_ERROR(ctx->ApplyProgram(program));
  RETURN_ERROR(vao.bind());

  if(auto e = rt.bind(GL_FRAMEBUFFER))
    e.value().release();
  else
    return std::move(e.error());

  RETURN_ERROR(ctx->ApplyBlend(blend, BlendFactor, false));
  RETURN_ERROR(ctx->ApplyFlags(Flags, CullMode, FillMode));
  ctx->ApplyPrimitiveIndex(Primitive, IndexType);

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

GLExpected<ComputePipelineState*> ComputePipelineState::create(FG_Shader computeshader, FG_Vec3i workgroup,
                                                               uint32_t flags) noexcept
{
  if(!computeshader)
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Cannot create compute pipeline without compute shader!");

  auto pipeline = new ComputePipelineState{ COMPUTE_PIPELINE_FLAG, workgroup };

  if(auto e = ProgramObject::create())
    pipeline->program = std::move(e.value());
  else
    return std::move(e.error());

  RETURN_ERROR(pipeline->program.attach(ShaderObject(computeshader)));
  RETURN_ERROR(pipeline->program.link());

  return pipeline;
}

GLExpected<std::string> ComputePipelineState::log() const noexcept { return program.log(); }

GLExpected<void> ComputePipelineState::apply(Context* ctx) noexcept
{
  ctx->ApplyWorkGroup(workgroup);
  RETURN_ERROR(ctx->ApplyProgram(program));

  return {};
}