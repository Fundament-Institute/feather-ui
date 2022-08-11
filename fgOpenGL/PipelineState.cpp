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
  auto test               = Buffer(vertexbuffers[0]);

  PipelineState* pipeline = new PipelineState{};

  if(auto e = ProgramObject::create())
    pipeline->program = std::move(e.value());
  else
    return std::move(e.error());

  // for(int i = 0; i < FG_ShaderStage_COUNT; ++i)
  for(auto shader : std::span(state.shaders))
  {
    if(shader != Backend::NULL_SHADER)
      RETURN_ERROR(pipeline->program.attach(ShaderObject(shader)));
  }
  RETURN_ERROR(pipeline->program.link());

  auto vlist = std::span(reinterpret_cast<std::pair<GLuint, GLsizei>*>(
                           malloc(sizeof(std::pair<GLuint, GLsizei>) * vertexbuffers.size())),
                         vertexbuffers.size());

  for(size_t i = 0; i < vlist.size(); ++i)
  {
    // Have to be careful to use placement new here to avoid UB due to alloca
    new(&vlist[i]) std::pair<GLuint, GLsizei>{ Buffer(vertexbuffers[i]), strides[i] };
  }

  if(auto e = VertexArrayObject::create(pipeline->program, attributes, vlist, Buffer(indexbuffer)))
  {
    pipeline->vao = std::move(e.value());
  }
  else
    return std::move(e.error());

  pipeline->Members = state.members;
  pipeline->rt      = Framebuffer(rendertarget);

  if(state.members & FG_Pipeline_Member_Primitive)
  {
    if(state.primitive >= ArraySize(PrimitiveMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");
    pipeline->Primitive = PrimitiveMapping[state.primitive];
  }

  if(state.members & FG_Pipeline_Member_Flags)
    pipeline->Flags = state.flags;
  if(state.members & FG_Pipeline_Member_Sample_Mask)
    pipeline->SampleMask = state.sampleMask;
  if(state.members & FG_Pipeline_Member_Stencil_Read_Mask)
    pipeline->StencilReadMask = state.stencilReadMask;
  if(state.members & FG_Pipeline_Member_Stencil_Write_Mask)
    pipeline->StencilWriteMask = state.stencilWriteMask;
  if(state.members & FG_Pipeline_Member_Stencil_Ref)
    pipeline->StencilRef = state.stencilRef;
  if(state.members & FG_Pipeline_Member_Blend_Factor)
    Context::ColorFloats(state.blendFactor, pipeline->BlendFactor, false);

  if(state.members & FG_Pipeline_Member_Depth_Func)
  {
    if(state.depthFunc >= ArraySize(ComparisonMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "DepthFunc not valid comparison function");
    pipeline->DepthFunc = ComparisonMapping[state.depthFunc];
  }

  if(state.members & FG_Pipeline_Member_Stencil_Func)
  {
    if(state.stencilFunc >= ArraySize(ComparisonMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilFunc not valid comparison function");
    pipeline->StencilFunc = ComparisonMapping[state.stencilFunc];
  }

  if(state.members & FG_Pipeline_Member_Stencil_OP)
  {
    if(state.stencilFailOp >= ArraySize(StencilOpMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilFailOp not valid stencil operation");
    pipeline->StencilFailOp = StencilOpMapping[state.stencilFailOp];

    if(state.stencilDepthFailOp >= ArraySize(StencilOpMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilDepthFailOp not valid stencil operation");
    pipeline->StencilDepthFailOp = StencilOpMapping[state.stencilDepthFailOp];

    if(state.stencilPassOp >= ArraySize(StencilOpMapping))
      return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "StencilPassOp not valid stencil operation");
    pipeline->StencilPassOp = StencilOpMapping[state.stencilPassOp];
  }

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

  if(state.members & FG_Pipeline_Member_Fill)
    pipeline->FillMode = state.fillMode;
  if(state.members & FG_Pipeline_Member_Cull)
    pipeline->CullMode = state.cullMode;
  if(state.members & FG_Pipeline_Member_Depth_Slope_Bias)
  {
    pipeline->DepthBias            = state.depthBias;
    pipeline->SlopeScaledDepthBias = state.slopeScaledDepthBias;
  }
  // pipeline->NodeMask          = state.NodeMask; // OpenGL does not support multi-GPU rendering without extensions
  pipeline->blend = blend;

  // glViewport(0, 0, 800, 600);
  return pipeline;
}

GLExpected<std::string> PipelineState::log() const noexcept { return program.log(); }

GLExpected<void> PipelineState::apply(Context* ctx) noexcept
{
  if(program.is_valid())
  {
    RETURN_ERROR(ctx->ApplyProgram(program));
  }
  RETURN_ERROR(vao.bind());

  if(auto e = rt.bind(GL_FRAMEBUFFER))
    e.value().release();
  else
    return std::move(e.error());

  if(Members & FG_Pipeline_Member_Blend_Factor)
  {
    RETURN_ERROR(ctx->ApplyBlendFactor(BlendFactor));
  }

  RETURN_ERROR(ctx->ApplyBlend(blend, false));

  if(Members & FG_Pipeline_Member_Flags)
  {
    RETURN_ERROR(ctx->ApplyFlags(Flags));
  }
  if(Members & FG_Pipeline_Member_Cull)
  {
    RETURN_ERROR(ctx->ApplyCull(CullMode));
  }
  if(Members & FG_Pipeline_Member_Fill)
  {
    RETURN_ERROR(ctx->ApplyFill(FillMode));
  }

  ctx->ApplyIndextype(IndexType);

  if(Members & FG_Pipeline_Member_Primitive)
    ctx->ApplyPrimitive(Primitive);

  if(Members & FG_Pipeline_Member_Depth_Func)
  {
    glDepthFunc(DepthFunc);
    GL_ERROR("glDepthFunc");
  }
  if(Members & FG_Pipeline_Member_Stencil_OP)
  {
    glStencilOp(StencilFailOp, StencilDepthFailOp, StencilPassOp);
    GL_ERROR("glStencilOp");
  }
  if(Members & FG_Pipeline_Member_Stencil_Write_Mask)
  {
    glStencilMask(StencilWriteMask);
    GL_ERROR("glStencilMask");
  }
  if(Members & (FG_Pipeline_Member_Stencil_Read_Mask || FG_Pipeline_Member_Stencil_Func || FG_Pipeline_Member_Stencil_Ref))
  {
    GLint func = StencilFunc;
    GLint ref  = StencilRef;
    GLint mask = StencilReadMask;

    if(!(Members & FG_Pipeline_Member_Stencil_Read_Mask))
      glGetIntegerv(GL_STENCIL_VALUE_MASK, &mask);
    if(!(Members & FG_Pipeline_Member_Stencil_Func))
      glGetIntegerv(GL_STENCIL_FUNC, &func);
    if(!(Members & FG_Pipeline_Member_Stencil_Ref))
      glGetIntegerv(GL_STENCIL_REF, &ref);
    GL_ERROR("glGetIntegerv");

    glStencilFunc(func, ref, mask);
    GL_ERROR("glStencilFunc");
  }

  if(Members & FG_Pipeline_Member_Depth_Slope_Bias)
  {
    glPolygonOffset(SlopeScaledDepthBias, DepthBias);
    GL_ERROR("glPolygonOffset");
  }
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