// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "ProviderGL.hpp"

#include "ProviderGL.hpp"
#include "feather/compiler.h"
#include <cfloat>
#include "ShaderObject.hpp"
#include "ProgramObject.hpp"
#include "Renderbuffer.hpp"
#include "PipelineState.hpp"
#include "EnumMapping.hpp"
#include <cstring>

using GL::Provider;

FG_Caps Provider::GetCaps(FG_GraphicsInterface* self)
{
  FG_Caps caps         = { 0 };
  caps.openGL.features = FG_Feature_API_OpenGL | FG_Feature_Immediate_Mode | FG_Feature_Background_Opacity |
                         FG_Feature_Lines_Alpha;
  caps.openGL.version = 20;
  caps.openGL.glsl    = 110;
  if(!glGetIntegerv)
    return caps;

  if(GLAD_GL_EXT_texture_sRGB && GLAD_GL_ARB_framebuffer_sRGB)
    caps.openGL.features |= FG_Feature_Blend_Gamma;

  if(GLAD_GL_ARB_blend_func_extended)
    caps.openGL.features |= FG_Feature_Blend_EX;

  if(GLAD_GL_ARB_texture_filter_anisotropic)
    caps.openGL.features |= FG_Feature_Anisotropic_Filter;

  if(GLAD_GL_ARB_half_float_pixel)
    caps.openGL.features |= FG_Feature_Half_Float;

  if(GLAD_GL_ARB_shader_atomic_counters)
    caps.openGL.features |= FG_Feature_Atomic_Counters;

  if(GLAD_GL_ARB_shader_storage_buffer_object)
    caps.openGL.features |= FG_Feature_StorageBuffer;

  if(GLAD_GL_ARB_sync)
    caps.openGL.features |= FG_Feature_Sync;

  if(GLAD_GL_ARB_draw_instanced)
    caps.openGL.features |= FG_Feature_Instancing;

  if(GLAD_GL_ARB_compute_shader)
    caps.openGL.features |= FG_Feature_Compute_Shader;

  if(GLAD_GL_ARB_tessellation_shader)
    caps.openGL.features |= FG_Feature_Tesselation_Shader;

  if(GLAD_GL_NV_mesh_shader)
    caps.openGL.features |= FG_Feature_Mesh_Shader;

  constexpr auto GetVec3i = [](GLenum e, FG_Vec3i& out) {
    glGetIntegeri_v(e, 0, &out.x);
    glGetIntegeri_v(e, 1, &out.x);
    glGetIntegeri_v(e, 2, &out.x);
  };

  glGetIntegerv(GL_MAX_TEXTURE_SIZE, &caps.openGL.max_texture_size);
  glGetIntegerv(GL_MAX_VIEWPORT_DIMS, &caps.openGL.max_viewport_size.x);
  glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &caps.openGL.max_3d_texture_size);
  glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &caps.openGL.max_vertex_buffer_size);
  glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &caps.openGL.max_index_buffer_size);
  glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &caps.openGL.max_cube_texture_size);
  glGetFloatv(GL_MAX_TEXTURE_LOD_BIAS, &caps.openGL.max_texture_bias);
  glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &caps.openGL.max_vertex_attributes);
  glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &caps.openGL.max_texture_samplers);
  glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS, &caps.openGL.max_fragment_uniforms);
  glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS, &caps.openGL.max_vertex_uniforms);
  glGetIntegerv(GL_MAX_VARYING_FLOATS, &caps.openGL.max_varying_outputs);
  glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, &caps.openGL.max_vertex_texture_samplers);
  glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &caps.openGL.max_samplers);
  glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &caps.openGL.max_texture_levels);
  glGetIntegerv(GL_MAX_PROGRAM_TEXEL_OFFSET, &caps.openGL.max_texel_offset);
  glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &caps.openGL.max_renderbuffer_size);
  glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &caps.openGL.max_rendertargets);
  glGetIntegerv(GL_MAX_SAMPLES, &caps.openGL.max_multisampling);
  glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &caps.openGL.max_texturebuffer_size);
  glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS, &caps.openGL.max_vertex_uniform_blocks);
  glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, &caps.openGL.max_geometry_uniform_blocks);
  glGetIntegerv(GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS, &caps.openGL.max_geometry_texture_samplers);
  glGetIntegerv(GL_MAX_TEXTURE_COORDS, &caps.openGL.max_texture_coordinates);
  glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_COMPONENTS, &caps.openGL.max_geometry_uniforms);
  glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &caps.openGL.max_geometry_output_vertices);
  glGetIntegerv(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS, &caps.openGL.max_geometry_total_output);
  glGetIntegerv(GL_MAX_VERTEX_OUTPUT_COMPONENTS, &caps.openGL.max_vertex_output_components);
  glGetIntegerv(GL_MAX_GEOMETRY_INPUT_COMPONENTS, &caps.openGL.max_geometry_input_components);
  glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_COMPONENTS, &caps.openGL.max_geometry_output_components);
  glGetIntegerv(GL_MAX_FRAGMENT_INPUT_COMPONENTS, &caps.openGL.max_fragment_input_components);
  glGetIntegerv(GL_MAX_SAMPLE_MASK_WORDS, &caps.openGL.max_multisample_mask_words);
  glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &caps.openGL.max_multisample_color_samples);
  glGetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES, &caps.openGL.max_multisample_depth_samples);
  glGetIntegerv(GL_MAX_DUAL_SOURCE_DRAW_BUFFERS, &caps.openGL.max_dual_source_rendertargets);
  glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &caps.openGL.max_compute_uniform_blocks);
  glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, &caps.openGL.max_compute_samplers);
  glGetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, &caps.openGL.max_compute_image_uniforms);
  glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &caps.openGL.max_compute_shared_memory);
  glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_COMPONENTS, &caps.openGL.max_compute_uniforms);
  glGetIntegerv(GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS, &caps.openGL.max_compute_atomic_counter_buffers);
  glGetIntegerv(GL_MAX_COMPUTE_ATOMIC_COUNTERS, &caps.openGL.max_compute_atomic_counters);
  glGetIntegerv(GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS, &caps.openGL.max_compute_components);
  glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &caps.openGL.max_work_group_invocations);
  GetVec3i(GL_MAX_COMPUTE_WORK_GROUP_COUNT, caps.openGL.max_work_group_count);
  GetVec3i(GL_MAX_COMPUTE_WORK_GROUP_SIZE, caps.openGL.max_work_group_size);
  glGetIntegerv(GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS, &caps.openGL.max_atomic_counter_buffers);
  glGetIntegerv(GL_MAX_COMBINED_ATOMIC_COUNTERS, &caps.openGL.max_atomic_counters);
  glGetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, &caps.openGL.max_image_uniforms);
  glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &caps.openGL.max_storage_blocks);
  glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &caps.openGL.max_storage_buffers);
  glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &caps.openGL.max_storage_block_size);
  glGetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES, &caps.openGL.max_shader_output_resources);
  glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &caps.openGL.max_texture_anisotropy);
  glGetIntegerv(GL_MAX_PATCH_VERTICES, &caps.openGL.max_patch_vertices);
  glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &caps.openGL.max_tessellation_level);
  glGetIntegerv(GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS, &caps.openGL.max_tess_control_uniforms);
  glGetIntegerv(GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS, &caps.openGL.max_tess_evaluation_uniforms);
  glGetIntegerv(GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS, &caps.openGL.max_tess_control_components);
  glGetIntegerv(GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS, &caps.openGL.max_tess_evaluation_components);
  glGetIntegerv(GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS, &caps.openGL.max_tess_control_samplers);
  glGetIntegerv(GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS, &caps.openGL.max_tess_evaluation_samplers);
  glGetIntegerv(GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS, &caps.openGL.max_tess_control_output_components);
  glGetIntegerv(GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS, &caps.openGL.max_tess_evaluation_output_components);
  glGetIntegerv(GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS, &caps.openGL.max_control_outputs);
  glGetIntegerv(GL_MAX_TESS_PATCH_COMPONENTS, &caps.openGL.max_patch_components);
  glGetIntegerv(GL_MAX_TESS_CONTROL_INPUT_COMPONENTS, &caps.openGL.max_tess_control_inputs);
  glGetIntegerv(GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS, &caps.openGL.max_tess_evaluation_inputs);
  glGetIntegerv(GL_MAX_IMAGE_UNITS, &caps.openGL.max_images);
  glGetIntegerv(GL_MAX_MESH_TEXTURE_IMAGE_UNITS_NV, &caps.openGL.max_mesh_samplers);
  glGetIntegerv(GL_MAX_MESH_UNIFORM_COMPONENTS_NV, &caps.openGL.max_mesh_uniforms);
  glGetIntegerv(GL_MAX_MESH_IMAGE_UNIFORMS_NV, &caps.openGL.max_mesh_image_uniforms);
  glGetIntegerv(GL_MAX_MESH_ATOMIC_COUNTER_BUFFERS_NV, &caps.openGL.max_mesh_atomic_counter_buffers);
  glGetIntegerv(GL_MAX_MESH_ATOMIC_COUNTERS_NV, &caps.openGL.max_mesh_atomic_counters);
  glGetIntegerv(GL_MAX_COMBINED_MESH_UNIFORM_COMPONENTS_NV, &caps.openGL.max_mesh_components);
  glGetIntegerv(GL_MAX_TASK_TEXTURE_IMAGE_UNITS_NV, &caps.openGL.max_task_samplers);
  glGetIntegerv(GL_MAX_TASK_IMAGE_UNIFORMS_NV, &caps.openGL.max_task_image_uniforms);
  glGetIntegerv(GL_MAX_TASK_UNIFORM_COMPONENTS_NV, &caps.openGL.max_task_uniform_components);
  glGetIntegerv(GL_MAX_TASK_ATOMIC_COUNTER_BUFFERS_NV, &caps.openGL.max_task_atomic_counter_buffers);
  glGetIntegerv(GL_MAX_TASK_ATOMIC_COUNTERS_NV, &caps.openGL.max_task_atomic_counters);
  glGetIntegerv(GL_MAX_COMBINED_TASK_UNIFORM_COMPONENTS_NV, &caps.openGL.max_task_components);
  glGetIntegerv(GL_MAX_MESH_WORK_GROUP_INVOCATIONS_NV, &caps.openGL.max_mesh_work_group_invocations);
  glGetIntegerv(GL_MAX_TASK_WORK_GROUP_INVOCATIONS_NV, &caps.openGL.max_task_work_group_invocations);
  glGetIntegerv(GL_MAX_MESH_TOTAL_MEMORY_SIZE_NV, &caps.openGL.max_mesh_memory);
  glGetIntegerv(GL_MAX_TASK_TOTAL_MEMORY_SIZE_NV, &caps.openGL.max_task_memory);
  glGetIntegerv(GL_MAX_MESH_OUTPUT_VERTICES_NV, &caps.openGL.max_mesh_output_vertices);
  glGetIntegerv(GL_MAX_MESH_OUTPUT_PRIMITIVES_NV, &caps.openGL.max_mesh_output_primitives);
  glGetIntegerv(GL_MAX_TASK_OUTPUT_COUNT_NV, &caps.openGL.max_task_output);
  glGetIntegerv(GL_MAX_DRAW_MESH_TASKS_COUNT_NV, &caps.openGL.max_mesh_draw_tasks);
  glGetIntegerv(GL_MAX_MESH_VIEWS_NV, &caps.openGL.max_mesh_views);
  GetVec3i(GL_MAX_MESH_WORK_GROUP_SIZE_NV, caps.openGL.max_mesh_work_group_size);
  GetVec3i(GL_MAX_TASK_WORK_GROUP_SIZE_NV, caps.openGL.max_task_work_group_size);

  return caps;
}

FG_Context* Provider::CreateContext(FG_GraphicsInterface* self, FG_Vec2i size, enum FG_PixelFormat backbuffer)
{
  return nullptr;
}
int Provider::ResizeContext(FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size)
{
  auto backend = static_cast<Provider*>(self);
  auto ctx     = static_cast<Context*>(context);
  LOG_ERROR(backend, ctx->Resize(FG_Vec2{ static_cast<float>(size.x), static_cast<float>(size.y) }));
  return ERR_SUCCESS;
}

FG_Shader Provider::CompileShader(FG_GraphicsInterface* self, FG_Context* context, enum FG_ShaderStage stage,
                                  const char* source)
{
  auto backend = static_cast<Provider*>(self);

  if(stage >= ArraySize(ShaderStageMapping))
    CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Unsupported shader stage").log(backend);
  else
  {
    // Note: Can't use LOG_ERROR here because we return a value
    if(auto r = ShaderObject::create(source, ShaderStageMapping[stage], backend))
      return std::move(r.value()).release();
    else
      r.error().log(backend);
  }
  return NULL_SHADER;
}
int Provider::DestroyShader(FG_GraphicsInterface* self, FG_Context* context, FG_Shader shader)
{
  auto backend = static_cast<Provider*>(self);

  // Take ownership and delete
  Owned<ShaderObject> s(shader);
  if(!s.is_valid())
  {
    backend->LOG(FG_Level_Error, "Invalid shader!", s.release());
    return ERR_INVALID_PARAMETER;
  }
  return 0;
}

void* Provider::CreateCommandList(FG_GraphicsInterface* self, FG_Context* context, bool bundle)
{
  auto backend = static_cast<Provider*>(self);
  if(backend->_insidelist)
  {
    backend->LOG(FG_Level_Error,
                 "This backend can't do multiple command lists at the same time! Did you forget to free the old list?");
    return NULL_COMMANDLIST;
  }
  backend->_insidelist = true;
  return context;
}

int Provider::DestroyCommandList(FG_GraphicsInterface* self, void* commands)
{
  auto backend = static_cast<Provider*>(self);
  if(!commands)
  {
    backend->LOG(FG_Level_Error, "Expected a non-null command list but got NULL instead!");
    return ERR_INVALID_PARAMETER;
  }
  if(!backend->_insidelist)
  {
    backend->LOG(FG_Level_Error, "Mismatched CreateCommandList / DestroyCommandList pair !");
    return ERR_INVALID_CALL;
  }

  backend->_insidelist = false;
  return ERR_SUCCESS;
}

int Provider::Clear(FG_GraphicsInterface* self, void* commands, uint8_t clearbits, FG_Color16 RGBA, uint8_t stencil,
                    float depth, uint32_t num_rects, FG_Rect* rects)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->Clear(clearbits, RGBA, stencil, depth, std::span(rects, num_rects)));
  return ERR_SUCCESS;
}
int Provider::CopyResource(FG_GraphicsInterface* self, void* commands, FG_Resource src, FG_Resource dest, FG_Vec3i size,
                           int level)
{
  auto backend = static_cast<Provider*>(self);

  if(Buffer::validate(src) && Buffer::validate(dest))
  {
    return CopySubresource(self, commands, src, dest, 0, 0, size.x);
  }
  else if((Texture::validate(src) && Texture::validate(dest)) ||
          (Renderbuffer::validate(src) && Renderbuffer::validate(dest)))
  {
    return CopyResourceRegion(self, commands, src, dest, level, FG_Vec3i{ 0, 0, 0 }, FG_Vec3i{ 0, 0, 0 },
                              FG_Vec3i{ size.x, size.y, size.z });
  }
  backend->LOG(FG_Level_Error, "Mismatched src / dest resources");
  return ERR_INVALID_PARAMETER;
}
int Provider::CopySubresource(FG_GraphicsInterface* self, void* commands, FG_Resource src, FG_Resource dest,
                              unsigned long srcoffset, unsigned long destoffset, unsigned long bytes)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->CopySubresource(src, dest, srcoffset, destoffset, bytes));
  return ERR_SUCCESS;
}

int Provider::CopyResourceRegion(FG_GraphicsInterface* self, void* commands, FG_Resource src, FG_Resource dest, int level,
                                 FG_Vec3i srcoffset, FG_Vec3i destoffset, FG_Vec3i size)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->CopyResourceRegion(src, dest, level, srcoffset, destoffset, size));
  return ERR_SUCCESS;
}
int Provider::DrawGL(FG_GraphicsInterface* self, void* commands, uint32_t vertexcount, uint32_t instancecount,
                     uint32_t startvertex, uint32_t startinstance)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->DrawArrays(vertexcount, instancecount, startvertex, startinstance));
  return 0;
}
int Provider::DrawIndexed(FG_GraphicsInterface* self, void* commands, uint32_t indexcount, uint32_t instancecount,
                          uint32_t startindex, int startvertex, uint32_t startinstance)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->DrawIndexed(indexcount, instancecount, startindex, startvertex, startinstance));
  return 0;
}
int Provider::DrawMesh(FG_GraphicsInterface* self, void* commands, uint32_t first, uint32_t count)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->DrawMesh(first, count));
  return 0;
}
int Provider::Dispatch(FG_GraphicsInterface* self, void* commands)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->Dispatch());
  return 0;
}

int Provider::SyncPoint(FG_GraphicsInterface* self, void* commands, uint32_t barrier_flags)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  GLbitfield flags = 0;
  if(barrier_flags & FG_BarrierFlag_Vertex)
    flags |= GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Element)
    flags |= GL_ELEMENT_ARRAY_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Uniform)
    flags |= GL_UNIFORM_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Texture_Fetch)
    flags |= GL_TEXTURE_FETCH_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Texture_Update)
    flags |= GL_TEXTURE_UPDATE_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Image_Access)
    flags |= GL_SHADER_IMAGE_ACCESS_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Command)
    flags |= GL_COMMAND_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Pixel)
    flags |= GL_PIXEL_BUFFER_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Buffer)
    flags |= GL_BUFFER_UPDATE_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_RenderTarget)
    flags |= GL_FRAMEBUFFER_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Storage_Buffer)
    flags |= GL_SHADER_STORAGE_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Transform_Feedback)
    flags |= GL_TRANSFORM_FEEDBACK_BARRIER_BIT;
  if(barrier_flags & FG_BarrierFlag_Atomic_Counter)
    flags |= GL_ATOMIC_COUNTER_BARRIER_BIT;

  LOG_ERROR(backend, context->Barrier(flags));
  return 0;
}

int Provider::SetPipelineState(FG_GraphicsInterface* self, void* commands, uintptr_t state)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;

  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  if(!state)
  {
    context->_program = nullptr;
    return ERR_INVALID_PARAMETER;
  }

  if(reinterpret_cast<PipelineState*>(state)->Members & COMPUTE_PIPELINE_FLAG)
  {
    LOG_ERROR(backend, reinterpret_cast<ComputePipelineState*>(state)->apply(context));
  }
  else
  {
    LOG_ERROR(backend, reinterpret_cast<PipelineState*>(state)->apply(context));
  }

  return 0;
}

int Provider::SetViewports(FG_GraphicsInterface* self, void* commands, FG_Viewport* viewports, uint32_t count)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  if(count > 1)
    return ERR_NOT_IMPLEMENTED;
  LOG_ERROR(backend, context->SetViewports({ viewports, count }));
  return ERR_SUCCESS;
}

int Provider::SetScissors(FG_GraphicsInterface* self, void* commands, FG_Rect* rects, uint32_t count)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  if(count > 1)
    return ERR_NOT_IMPLEMENTED;
  LOG_ERROR(backend, context->SetScissors({ rects, count }));
  return ERR_SUCCESS;
}

int Provider::SetShaderConstants(FG_GraphicsInterface* self, void* commands, const FG_ShaderParameter* uniforms,
                                 const FG_ShaderValue* values, uint32_t count)
{
  auto backend = static_cast<Provider*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  LOG_ERROR(backend, context->SetShaderUniforms(uniforms, values, count));
  return ERR_SUCCESS;
}

int Provider::Execute(FG_GraphicsInterface* self, FG_Context* context, void* commands) { return ERR_SUCCESS; }

uintptr_t Provider::CreatePipelineState(FG_GraphicsInterface* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                        FG_Resource rendertarget, FG_Blend* blends, FG_Resource* vertexbuffer,
                                        GLsizei* strides, uint32_t n_buffers, FG_VertexParameter* attributes,
                                        uint32_t n_attributes, FG_Resource indexbuffer, uint8_t indexstride)
{
  if(!pipelinestate || !context || !blends)
    return NULL_PIPELINE;

  auto backend = static_cast<Provider*>(self);
  auto ctx     = reinterpret_cast<Context*>(context);

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = PipelineState::create(*pipelinestate, rendertarget, *blends, std::span(vertexbuffer, n_buffers), strides,
                                    std::span(attributes, n_attributes), indexbuffer, indexstride))
    return reinterpret_cast<uintptr_t>(e.value());
  else
    e.log(backend);
  return NULL_PIPELINE;
}
uintptr_t Provider::CreateComputePipeline(FG_GraphicsInterface* self, FG_Context* context, FG_Shader computeshader,
                                          FG_Vec3i workgroup, uint32_t flags)
{
  if(!context)
    return NULL_PIPELINE;

  auto backend = static_cast<Provider*>(self);
  auto ctx     = reinterpret_cast<Context*>(context);

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = ComputePipelineState::create(computeshader, workgroup, flags))
    return reinterpret_cast<uintptr_t>(e.value());
  else
    e.log(backend);
  return NULL_PIPELINE;
}

int Provider::DestroyPipelineState(FG_GraphicsInterface* self, FG_Context* context, uintptr_t state)
{
  if(!state)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Provider*>(self);
  if(reinterpret_cast<PipelineState*>(state)->Members & COMPUTE_PIPELINE_FLAG)
    delete reinterpret_cast<ComputePipelineState*>(state);
  else
    delete reinterpret_cast<PipelineState*>(state);
  return 0;
}

FG_Resource Provider::CreateBuffer(FG_GraphicsInterface* self, FG_Context* context, void* data, uint32_t bytes,
                                   enum FG_Usage usage)
{
  auto backend = static_cast<Provider*>(self);
  if(usage >= ArraySize(UsageMapping))
    return NULL_RESOURCE;

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = Buffer::create(UsageMapping[usage], data, bytes))
    return std::move(e.value()).release();
  else
    e.log(backend);
  return NULL_RESOURCE;
}
FG_Resource Provider::CreateTexture(FG_GraphicsInterface* self, FG_Context* context, FG_Vec2i size, enum FG_Usage usage,
                                    enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount)
{
  auto backend = static_cast<Provider*>(self);
  if(usage >= ArraySize(UsageMapping) || !sampler)
    return NULL_RESOURCE;

  // Can't use LOG_ERROR here because we return a pointer.
  if(usage == FG_Usage_Renderbuffer)
  {
    if(auto e = Renderbuffer::create(UsageMapping[usage], Format::Create(format, false), size, MultiSampleCount))
      return std::move(e.value()).release();
    else
      e.log(backend);
  }
  else
  {
    if(auto e =
         Texture::create2D(UsageMapping[usage], Format::Create(format, false), size, *sampler, data, MultiSampleCount))
      return std::move(e.value()).release();
    else
      e.log(backend);
  }

  return NULL_RESOURCE;
}
FG_Resource Provider::CreateRenderTarget(FG_GraphicsInterface* self, FG_Context* context, FG_Resource depthstencil,
                                         FG_Resource* textures, uint32_t n_textures, int attachments)
{
  auto backend = static_cast<Provider*>(self);

  if(auto e = Framebuffer::create(GL_FRAMEBUFFER, GL_TEXTURE_2D, 0, 0, textures, n_textures))
  {
    if(auto flags = (attachments & (FG_ClearFlag_Depth | FG_ClearFlag_Stencil)))
    {
      switch(flags)
      {
      case FG_ClearFlag_Depth | FG_ClearFlag_Stencil: flags = GL_DEPTH_STENCIL_ATTACHMENT; break;
      case FG_ClearFlag_Depth: flags = GL_DEPTH_ATTACHMENT; break;
      case FG_ClearFlag_Stencil: flags = GL_STENCIL_ATTACHMENT; break;
      }

      if(auto __e = e.value().attach2D(GL_FRAMEBUFFER, flags, GL_TEXTURE_2D, depthstencil, 0); __e.has_error())
      {
        e.log(backend);
        return NULL_RESOURCE;
      }
    }
    return std::move(e.value()).release();
  }
  else
    e.log(backend);
  return NULL_RESOURCE;
}
int Provider::DestroyResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource)
{
  auto backend = static_cast<Provider*>(self);
  if(Framebuffer::validate(resource))
    Owned<Framebuffer> b(resource);
  else if(Texture::validate(resource))
    Owned<Texture> t(resource);
  else if(Renderbuffer::validate(resource))
    Owned<Renderbuffer> t(resource);
  else if(Buffer::validate(resource))
    Owned<Buffer> fb(resource);
  else
    return ERR_INVALID_PARAMETER;

  if(GLError e{ "glDelete", __FILE__, __LINE__ }; e.has_error())
    return e.log(backend);

  return 0;
}

GLenum getOpenGLAccessFlags(uint32_t flags)
{
  GLenum v = 0;
  if(flags & FG_AccessFlag_Read)
    v |= GL_MAP_READ_BIT;
  if(flags & FG_AccessFlag_Write)
    v |= GL_MAP_WRITE_BIT;
  // if(flags & FG_AccessFlag_PERSISTENT)
  //   v |= GL_MAP_PERSISTENT_BIT;
  if(flags & FG_AccessFlag_Invalidate_Range)
    v |= GL_MAP_INVALIDATE_RANGE_BIT;
  if(flags & FG_AccessFlag_Invalidate_Buffer)
    v |= GL_MAP_INVALIDATE_BUFFER_BIT;
  if(flags & FG_AccessFlag_Unsynchronized)
    v |= GL_MAP_UNSYNCHRONIZED_BIT;
  return v;
}

GLenum getOpenGLAccessEnum(uint32_t flags)
{
  switch(flags)
  {
  case FG_AccessFlag_Read | FG_AccessFlag_Write: return GL_READ_WRITE;
  case FG_AccessFlag_Read: return GL_READ_ONLY;
  case FG_AccessFlag_Write: return GL_WRITE_ONLY;
  }
  return 0;
}

void* Provider::MapResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, uint32_t offset,
                            uint32_t length, enum FG_Usage usage, uint32_t access)
{
  auto backend = static_cast<Provider*>(self);

  if(usage >= ArraySize(UsageMapping) || !context)
    return nullptr;

  BindRef bind;
  if(Buffer::validate(resource))
  {
    if(auto b = Buffer(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      b.error().log(backend);
  }
  if(Texture::validate(resource))
  {
    if(auto b = Texture(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      b.error().log(backend);
  }
  if(Framebuffer::validate(resource))
  {
    if(auto b = Framebuffer(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      b.error().log(backend);
  }
  if(!bind)
    return nullptr;

  void* v = nullptr;
  if(!offset && !length)
    v = glMapBuffer(UsageMapping[usage], getOpenGLAccessEnum(access));
  else
    v = glMapBufferRange(UsageMapping[usage], offset, length, getOpenGLAccessFlags(access));

  if(GLError e{ "glMapBuffer", __FILE__, __LINE__ }; e.has_error())
  {
    e.log(backend);
    return nullptr;
  }

  return v;
}
int Provider::UnmapResource(FG_GraphicsInterface* self, FG_Context* context, FG_Resource resource, enum FG_Usage usage)
{
  auto backend = static_cast<Provider*>(self);

  if(usage >= ArraySize(UsageMapping) || !context)
    return ERR_INVALID_PARAMETER;

  BindRef bind;
  if(Buffer::validate(resource))
  {
    if(auto b = Buffer(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(Texture::validate(resource))
  {
    if(auto b = Texture(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(Framebuffer::validate(resource))
  {
    if(auto b = Framebuffer(resource).bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(!bind)
    return ERR_INVALID_PARAMETER;

  glUnmapBuffer(UsageMapping[usage]);

  if(GLError e{ "glUnmapBuffer", __FILE__, __LINE__ }; e.has_error())
    return e.log(backend);

  return 0;
}

int Provider::BeginDraw(FG_GraphicsInterface* self, FG_Context* context, FG_Rect* area)
{
  if(!context)
    return ERR_MISSING_PARAMETER;
  LOG_ERROR(static_cast<Provider*>(self), reinterpret_cast<Context*>(context)->BeginDraw(area));
  return ERR_SUCCESS;
}
int Provider::EndDraw(FG_GraphicsInterface* self, FG_Context* context)
{
  LOG_ERROR(static_cast<Provider*>(self), reinterpret_cast<Context*>(context)->EndDraw());
  return ERR_SUCCESS;
}

int Provider::DestroyGL(FG_GraphicsInterface* self)
{
  if(!self)
    return ERR_INVALID_PARAMETER;
  delete static_cast<Provider*>(self);
  return ERR_SUCCESS;
}

int Provider::LoadGL(GLADloadproc loader)
{
  if(!gladLoadGLLoader(loader))
    LOG(FG_Level_Error, "gladLoadGL failed");
  else
    return ERR_SUCCESS;

  return ERR_UNKNOWN;
}

Provider::Provider(void* log_context, FG_Log log) : _logctx(log_context), _log(log), _insidelist(false)
{
  getCaps               = &GetCaps;
  createContext         = &CreateContext;
  resizeContext         = &ResizeContext;
  compileShader         = &CompileShader;
  destroyShader         = &DestroyShader;
  createCommandList     = &CreateCommandList;
  destroyCommandList    = &DestroyCommandList;
  clear                 = &Clear;
  copyResource          = &CopyResource;
  copySubresource       = &CopySubresource;
  copyResourceRegion    = &CopyResourceRegion;
  draw                  = &DrawGL;
  drawIndexed           = &DrawIndexed;
  drawMesh              = &DrawMesh;
  dispatch              = &Dispatch;
  syncPoint             = &SyncPoint;
  setPipelineState      = &SetPipelineState;
  setViewports          = &SetViewports;
  setScissors           = &SetScissors;
  setShaderConstants    = &SetShaderConstants;
  execute               = &Execute;
  createPipelineState   = &CreatePipelineState;
  createComputePipeline = &CreateComputePipeline;
  destroyPipelineState  = &DestroyPipelineState;
  createBuffer          = &CreateBuffer;
  createTexture         = &CreateTexture;
  createRenderTarget    = &CreateRenderTarget;
  destroyResource       = &DestroyResource;
  mapResource           = &MapResource;
  unmapResource         = &UnmapResource;
  destroy               = &DestroyGL;

  this->LOG(FG_Level_Notice, "Initializing fgOpenGL...");
}

Provider::~Provider() {}

extern "C" FG_COMPILER_DLLEXPORT FG_GraphicsInterface* fgOpenGL(void* log_context, FG_Log log)
{
  static_assert(std::is_same<FG_InitGraphics, decltype(&fgOpenGL)>::value,
                "fgOpenGL must match InitBackend function pointer");

  return new Provider(log_context, log);
}
