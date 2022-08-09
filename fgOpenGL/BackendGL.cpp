// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "platform.hpp"
#include "BackendGL.hpp"
#include <cfloat>
#include "ShaderObject.hpp"
#include "ProgramObject.hpp"
#include "Renderbuffer.hpp"
#include "PipelineState.hpp"
#include "EnumMapping.hpp"
#include <cstring>

#ifdef FG_PLATFORM_WIN32
  #include <dwrite_1.h>
#else
  #include <dlfcn.h>
#endif

using GL::Backend;

int Backend::_lasterr            = 0;
int Backend::_refcount           = 0;
char Backend::_lasterrdesc[1024] = {};
int Backend::_maxjoy             = 0;
Backend* Backend::_singleton     = nullptr;
const float Backend::BASE_DPI    = 96.0f;
void* Backend::_library          = 0;

FG_Result Backend::Behavior(Context* w, const FG_Msg& msg)
{
  return (*_behavior)(w->element, w, _root, const_cast<FG_Msg*>(&msg));
}

FG_Caps Backend::GetCaps(FG_Backend* self)
{
  FG_Caps caps;
  caps.openGL.features = FG_Feature_API_OpenGL | FG_Feature_Immediate_Mode | FG_Feature_Background_Opacity |
                         FG_Feature_Lines_Alpha | FG_Feature_Blend_Gamma | FG_Feature_Instancing |
                         FG_Feature_Independent_Blend;
  caps.openGL.version           = 20;
  caps.openGL.glsl              = 110;
  caps.openGL.max_rendertargets = 8;
  return caps;
}

FG_Shader Backend::CompileShader(FG_Backend* self, FG_Context* context, enum FG_ShaderStage stage, const char* source)
{
  auto backend = static_cast<Backend*>(self);

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
int Backend::DestroyShader(FG_Backend* self, FG_Context* context, FG_Shader shader)
{
  auto backend = static_cast<Backend*>(self);

  // Take ownership and delete
  Owned<ShaderObject> s(shader);
  if(!s.is_valid())
  {
    backend->LOG(FG_Level_Error, "Invalid shader!", s.release());
    return ERR_INVALID_PARAMETER;
  }
  return 0;
}

void* Backend::CreateCommandList(FG_Backend* self, FG_Context* context, bool bundle)
{
  auto backend = static_cast<Backend*>(self);
  if(backend->_insidelist)
  {
    backend->LOG(FG_Level_Error,
                 "This backend can't do multiple command lists at the same time! Did you forget to free the old list?");
    return NULL_COMMANDLIST;
  }
  backend->_insidelist = true;
  return context;
}

int Backend::DestroyCommandList(FG_Backend* self, void* commands)
{
  auto backend = static_cast<Backend*>(self);
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

int Backend::Clear(FG_Backend* self, void* commands, uint8_t clearbits, FG_Color RGBA, uint8_t stencil, float depth,
                   uint32_t num_rects, FG_Rect* rects)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  glClearDepth(depth);
  if(auto e = glGetError())
    return e;
  glClearStencil(stencil);
  if(auto e = glGetError())
    return e;
  std::array<float, 4> colors;
  Context::ColorFloats(RGBA, colors, false);
  glClearColor(colors[0], colors[1], colors[2], colors[3]);
  if(auto e = glGetError())
    return e;

  GLbitfield flags = 0;
  if(clearbits & FG_ClearFlag_Color)
    flags |= GL_COLOR_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Depth)
    flags |= GL_DEPTH_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Stencil)
    flags |= GL_STENCIL_BUFFER_BIT;
  if(clearbits & FG_ClearFlag_Accumulator)
    flags |= GL_ACCUM_BUFFER_BIT;

  if(!num_rects)
  {
    glClear(flags);
    return glGetError();
  }
  // TODO: iterate through rects and set scissor rects for each one and clear.

  return ERR_NOT_IMPLEMENTED;
}
int Backend::CopyResource(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest)
{
  auto backend = static_cast<Backend*>(self);

  GLuint source = src;
  GLuint destination = dest;

  if (IsBuffer(source) && IsBuffer(dest)) 
  {
    return CopySubresource(self, commands, src, dest, 0, 0, 0);
  }
  else if((IsTexture(source) && IsTexture(destination)) || (IsRenderbuffer(source) && IsRenderbuffer(destination)))
  {
    return CopyResourceRegion(self, commands, src, dest, FG_Vec3i{ 0, 0, 0 }, FG_Vec3i{ 0, 0, 0 }, FG_Vec3i{ 800, 600, 1 });
  }
  else
    return 0;

  return 1;
}
int Backend::CopySubresource(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest, unsigned long srcoffset,
                             unsigned long destoffset, unsigned long bytes)
{
  auto backend = static_cast<Backend*>(self);

  LOG_ERROR(backend, backend->CopySubresourceHelper(src, dest, srcoffset, destoffset, bytes));

  return 1;
}
GL::GLExpected<void> Backend::CopySubresourceHelper(FG_Resource src, FG_Resource dest, unsigned long srcoffset,
                                                    unsigned long destoffset, unsigned long bytes)
{
  GLuint source      = src;
  GLuint destination = dest;

  if (IsBuffer(source) && IsBuffer(destination)) 
  {
    if(auto e = Buffer(source).bind(GL_COPY_READ_BUFFER))
    {
      if(auto b = Buffer(destination).bind(GL_COPY_WRITE_BUFFER))
      {
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, srcoffset, destoffset, bytes);
        GL_ERROR("glCopyBufferSubData");
      }
      else
        return std::move(b.error());
    }
    else
      return std::move(e.error());
  }
  else
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Incompatible source and/or destination resources");

  return {};
}

int Backend::CopyResourceRegion(FG_Backend* self, void* commands, FG_Resource src, FG_Resource dest, FG_Vec3i srcoffset,
                                FG_Vec3i destoffset, FG_Vec3i size)
{
  auto backend = static_cast<Backend*>(self);
  
  LOG_ERROR(backend, backend->CopyResourceRegionHelper(GL_TEXTURE_2D, src, dest, srcoffset, destoffset, size));

  return 1;
}
GL::GLExpected<void> Backend::CopyResourceRegionHelper(GLenum type, FG_Resource src, FG_Resource dest, FG_Vec3i srcoffset,
                                                       FG_Vec3i destoffset, FG_Vec3i size)
{
  GLuint source      = src;
  GLuint destination = dest;
  
  if(IsTexture(source) && IsTexture(destination))
  {
    switch(type)
    {
    case GL_TEXTURE_1D:
      glCopyImageSubData(source, GL_TEXTURE_1D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_1D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z);
      break;
    case GL_TEXTURE_3D:
      glCopyImageSubData(source, GL_TEXTURE_3D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_3D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z);
      break;
    default:
      glCopyImageSubData(source, GL_TEXTURE_2D, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_TEXTURE_2D, 0,
                         destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z);
      break;
    }
    
    GL_ERROR("glCopyImageSubData");
  }
  else if(IsRenderbuffer(source) && IsRenderbuffer(destination))
  {
    glCopyImageSubData(source, GL_RENDERBUFFER, 0, srcoffset.x, srcoffset.y, srcoffset.z, destination, GL_RENDERBUFFER, 0,
                       destoffset.x, destoffset.y, destoffset.z, size.x, size.y, size.z);
    GL_ERROR("glCopyImageSubData");
  }
  else
    return CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Incompatible source and/or destination resources");

  return {};
}

int Backend::DrawGL(FG_Backend* self, void* commands, uint32_t vertexcount, uint32_t instancecount, uint32_t startvertex,
                    uint32_t startinstance)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->DrawArrays(vertexcount, instancecount, startvertex, startinstance));
  return 0;
}
int Backend::DrawIndexed(FG_Backend* self, void* commands, uint32_t indexcount, uint32_t instancecount, uint32_t startindex,
                         int startvertex, uint32_t startinstance)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->DrawIndexed(indexcount, instancecount, startindex, startvertex, startinstance));
  return 0;
}

int Backend::Dispatch(FG_Backend* self, void* commands)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, context->Dispatch());
  return 0;
}

int Backend::SyncPoint(FG_Backend* self, void* commands, uint32_t barrier_flags)
{
  if(!commands)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Backend*>(self);
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

int Backend::SetPipelineState(FG_Backend* self, void* commands, uintptr_t state)
{
  if(!commands || !state)
    return ERR_INVALID_PARAMETER;

  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);

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
int Backend::SetDepthStencil(FG_Backend* self, void* commands, bool Front, uint8_t StencilFailOp,
                             uint8_t StencilDepthFailOp, uint8_t StencilPassOp, uint8_t StencilFunc)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  return ERR_NOT_IMPLEMENTED;
}
int Backend::SetViewports(FG_Backend* self, void* commands, FG_Viewport* viewports, uint32_t count)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  return ERR_NOT_IMPLEMENTED;
}
int Backend::SetShaderConstants(FG_Backend* self, void* commands, const FG_ShaderParameter* uniforms,
                                const FG_ShaderValue* values, uint32_t count)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  LOG_ERROR(backend, context->SetShaderUniforms(uniforms, values, count));
  return 0;
}
int Backend::Execute(FG_Backend* self, FG_Context* context, void* commands) { return ERR_SUCCESS; }

uintptr_t Backend::CreatePipelineState(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                       FG_Resource rendertarget, FG_Blend* blends, FG_Resource* vertexbuffer,
                                       GLsizei* strides, uint32_t n_buffers, FG_VertexParameter* attributes,
                                       uint32_t n_attributes, FG_Resource indexbuffer, uint8_t indexstride)
{
  if(!pipelinestate || !context || !blends)
    return NULL_PIPELINE;

  auto backend = static_cast<Backend*>(self);
  auto ctx     = reinterpret_cast<Context*>(context);

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = PipelineState::create(*pipelinestate, rendertarget, *blends, std::span(vertexbuffer, n_buffers), strides,
                                    std::span(attributes, n_attributes), indexbuffer, indexstride))
    return reinterpret_cast<uintptr_t>(e.value());
  else
    e.log(backend);
  return NULL_PIPELINE;
}
uintptr_t Backend::CreateComputePipeline(FG_Backend* self, FG_Context* context, FG_Shader computeshader, FG_Vec3i workgroup,
                                         uint32_t flags)
{
  if(!context)
    return NULL_PIPELINE;

  auto backend = static_cast<Backend*>(self);
  auto ctx     = reinterpret_cast<Context*>(context);

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = ComputePipelineState::create(computeshader, workgroup, flags))
    return reinterpret_cast<uintptr_t>(e.value());
  else
    e.log(backend);
  return NULL_PIPELINE;
}

int Backend::DestroyPipelineState(FG_Backend* self, FG_Context* context, uintptr_t state)
{
  if(!state)
    return ERR_INVALID_PARAMETER;
  auto backend = static_cast<Backend*>(self);
  if(reinterpret_cast<PipelineState*>(state)->Members & COMPUTE_PIPELINE_FLAG)
    delete reinterpret_cast<ComputePipelineState*>(state);
  else
    delete reinterpret_cast<PipelineState*>(state);
  return 0;
}

FG_Resource Backend::CreateBuffer(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes, enum FG_Usage usage)
{
  auto backend = static_cast<Backend*>(self);
  if(usage >= ArraySize(UsageMapping))
    return NULL_RESOURCE;

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = Buffer::create(UsageMapping[usage], data, bytes))
    return std::move(e.value()).release();
  else
    e.log(backend);
  return NULL_RESOURCE;
}
FG_Resource Backend::CreateTexture(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_Usage usage,
                                   enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount)
{
  auto backend = static_cast<Backend*>(self);
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
FG_Resource Backend::CreateRenderTarget(FG_Backend* self, FG_Context* context, FG_Resource depthstencil,
                                        FG_Resource* textures, uint32_t n_textures, int attachments)
{
  auto backend = static_cast<Backend*>(self);

  if(auto e = FrameBuffer::create(GL_FRAMEBUFFER, GL_TEXTURE_2D, 0, 0, textures, n_textures))
  {
    if(auto flags = (attachments & (FG_ClearFlag_Depth | FG_ClearFlag_Stencil)))
    {
      switch(flags)
      {
      case FG_ClearFlag_Depth | FG_ClearFlag_Stencil: flags = GL_DEPTH_STENCIL_ATTACHMENT; break;
      case FG_ClearFlag_Depth: flags = GL_DEPTH_ATTACHMENT; break;
      case FG_ClearFlag_Stencil: flags = GL_STENCIL_ATTACHMENT; break;
      }

      if(auto __e = e.value().attach2D(GL_FRAMEBUFFER, flags, GL_TEXTURE_2D, depthstencil, 0)) {}
      else
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
int Backend::DestroyResource(FG_Backend* self, FG_Context* context, FG_Resource resource)
{
  auto backend = static_cast<Backend*>(self);
  GLuint i     = resource;
  if(glIsBuffer(i))
    Owned<Buffer> b(i);
  else if(glIsTexture(i))
    Owned<Texture> t(i);
  else if(glIsRenderbuffer(i))
    Owned<Renderbuffer> t(i);
  else if(glIsFramebuffer(i))
    Owned<FrameBuffer> fb(i);
  else
    return ERR_INVALID_PARAMETER;

  return ERR_SUCCESS;
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

void* Backend::MapResource(FG_Backend* self, FG_Context* context, FG_Resource resource, uint32_t offset, uint32_t length,
                           enum FG_Usage usage, uint32_t access)
{
  auto backend = static_cast<Backend*>(self);
  GLuint i     = resource;

  if(usage >= ArraySize(UsageMapping) || !context)
    return nullptr;

  BindRef bind;
  if(auto v = Buffer(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      b.error().log(backend);
  }
  if(auto v = Texture(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      b.error().log(backend);
  }
  if(auto v = FrameBuffer(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
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

  if(GLError e{ "glMapBuffer", __FILE__, __LINE__ })
  {
    e.log(backend);
    return nullptr;
  }

  return v;
}
int Backend::UnmapResource(FG_Backend* self, FG_Context* context, FG_Resource resource, enum FG_Usage usage)
{
  auto backend = static_cast<Backend*>(self);
  GLuint i     = resource;

  if(usage >= ArraySize(UsageMapping) || !context)
    return ERR_INVALID_PARAMETER;

  BindRef bind;
  if(auto v = Buffer(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(auto v = Texture(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(auto v = FrameBuffer(i))
  {
    if(auto b = v.bind(UsageMapping[usage]))
      bind = std::move(b.value());
    else
      return b.error().log(backend);
  }
  if(!bind)
    return ERR_INVALID_PARAMETER;

  glUnmapBuffer(UsageMapping[usage]);

  if(GLError e{ "glUnmapBuffer", __FILE__, __LINE__ })
    return e.log(backend);

  return 0;
}

FG_Window* Backend::CreateWindowGL(FG_Backend* self, FG_Element* element, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                                   const char* caption, uint64_t flags)
{
  auto backend = static_cast<Backend*>(self);
  _lasterr     = 0;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  Window* window = new Window(backend, reinterpret_cast<GLFWmonitor*>(display), element, pos, dim, flags, caption);

  if(!_lasterr)
  {
    window->_next     = backend->_windows;
    backend->_windows = window;
    if(window->_next)
      window->_next->_prev = window;

    return window;
  }

  delete window;
  return nullptr;
}

int Backend::SetWindow(FG_Backend* self, FG_Window* window, FG_Element* element, FG_Display* display, FG_Vec2* pos,
                       FG_Vec2* dim, const char* caption, uint64_t flags)
{
  if(!window)
    return ERR_MISSING_PARAMETER;
  auto w     = static_cast<Window*>(window);
  w->element = element;

  auto glwindow = w->GetWindow();
  if(!glwindow)
  {
    if(dim)
      w->ApplyDim(*dim);
    return ERR_SUCCESS;
  }
  if(caption)
    glfwSetWindowTitle(glwindow, caption);

  if((w->_flags ^ flags) & FG_WindowFlag_Fullscreen) // If we toggled fullscreen we need a different code path
  {
    int posx, posy;
    int dimx, dimy;
    glfwGetWindowPos(glwindow, &posx, &posy);
    glfwGetWindowSize(glwindow, &dimx, &dimy);
    if(flags & FG_WindowFlag_Fullscreen)
      glfwSetWindowMonitor(glwindow, (GLFWmonitor*)display, 0, 0, !dim ? dimx : static_cast<int>(ceilf(dim->x)),
                           !dim ? dimy : static_cast<int>(ceilf(dim->y)), GLFW_DONT_CARE);
    else
      glfwSetWindowMonitor(glwindow, NULL, !pos ? posx : static_cast<int>(ceilf(pos->x)),
                           !pos ? posy : static_cast<int>(ceilf(pos->y)), !dim ? dimx : static_cast<int>(ceilf(dim->x)),
                           !dim ? dimy : static_cast<int>(ceilf(dim->y)), GLFW_DONT_CARE);
  }
  else
  {
    if(pos)
      glfwSetWindowPos(glwindow, static_cast<int>(ceilf(pos->x)), static_cast<int>(ceilf(pos->y)));
    if(dim)
      glfwSetWindowSize(glwindow, static_cast<int>(ceilf(dim->x)), static_cast<int>(ceilf(dim->y)));
  }

  if(flags & FG_WindowFlag_Maximized)
    glfwMaximizeWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  if(flags & FG_WindowFlag_Minimized)
    glfwIconifyWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  w->_flags = flags;
  return ERR_SUCCESS;
}

int Backend::DestroyWindow(FG_Backend* self, FG_Window* window)
{
  if(!self || !window)
    return ERR_MISSING_PARAMETER;

  _lasterr = 0; // We set this to capture GLFW errors, which are seperate from OpenGL errors (for right now)
  delete static_cast<Window*>(window);
  return _lasterr;
}
int Backend::BeginDraw(FG_Backend* self, FG_Context* context, FG_Rect* area)
{
  if(!context)
    return ERR_MISSING_PARAMETER;
  _lasterr = 0; // We set this to capture GLFW errors, which are seperate from OpenGL errors (for right now)
  LOG_ERROR(static_cast<Backend*>(self), reinterpret_cast<Context*>(context)->BeginDraw(area));
  return _lasterr;
}
int Backend::EndDraw(FG_Backend* self, FG_Context* context)
{
  _lasterr = 0; // We set this to capture GLFW errors, which are seperate from OpenGL errors (for right now)
  LOG_ERROR(static_cast<Backend*>(self), reinterpret_cast<Context*>(context)->EndDraw());
  return _lasterr;
}

int Backend::PutClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, const char* data, uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_CLIPBOARD_FAILURE;
  if(data != 0 && count > 0 && EmptyClipboard())
  {
    if(kind == FG_Clipboard_Text)
    {
      size_t unilen  = MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(count), 0, 0);
      HGLOBAL unimem = GlobalAlloc(GMEM_MOVEABLE, unilen * sizeof(wchar_t));
      if(unimem)
      {
        wchar_t* uni = reinterpret_cast<wchar_t*>(GlobalLock(unimem));
        size_t sz    = MultiByteToWideChar(CP_UTF8, 0, data, static_cast<int>(count), uni, static_cast<int>(unilen));
        if(sz < unilen) // ensure we have a null terminator
          uni[sz] = 0;
        GlobalUnlock(unimem);
        SetClipboardData(CF_UNICODETEXT, unimem);
      }
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count + 1);
      if(gmem)
      {
        char* mem = reinterpret_cast<char*>(GlobalLock(gmem));
        MEMCPY(mem, count + 1, data, count);
        mem[count] = 0;
        GlobalUnlock(gmem);
        SetClipboardData(CF_TEXT, gmem);
      }
    }
    else
    {
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count);
      if(gmem)
      {
        void* mem = GlobalLock(gmem);
        MEMCPY(mem, count, data, count);
        GlobalUnlock(gmem);
        UINT format = CF_PRIVATEFIRST;
        switch(kind)
        {
        case FG_Clipboard_Wave: format = CF_WAVE; break;
        case FG_Clipboard_Bitmap: format = CF_BITMAP; break;
        }
        SetClipboardData(format, gmem);
      }
    }
  }
  CloseClipboard();
  return ERR_SUCCESS;
#else
  if(kind != FG_Clipboard_Text)
    return ERR_INVALID_KIND;

  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), data);
  return _lasterr;
#endif
}

uint32_t Backend::GetClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, void* target, uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_SUCCESS;
  UINT format = CF_PRIVATEFIRST;
  switch(kind)
  {
  case FG_Clipboard_Text:
    if(IsClipboardFormatAvailable(CF_UNICODETEXT))
    {
      HANDLE gdata = GetClipboardData(CF_UNICODETEXT);
      SIZE_T len   = 0;
      if(gdata)
      {
        SIZE_T size        = GlobalSize(gdata) / 2;
        const wchar_t* str = (const wchar_t*)GlobalLock(gdata);
        if(str)
        {
          len = WideCharToMultiByte(CP_UTF8, 0, str, static_cast<int>(size), 0, 0, NULL, NULL);

          if(target && count >= len)
            len = WideCharToMultiByte(CP_UTF8, 0, str, static_cast<int>(size), (char*)target, static_cast<int>(count), NULL,
                                      NULL);

          GlobalUnlock(gdata);
        }
      }
      CloseClipboard();
      return (uint32_t)len;
    }
    format = CF_TEXT;
    break;
  case FG_Clipboard_Wave: format = CF_WAVE; break;
  case FG_Clipboard_Bitmap: format = CF_BITMAP; break;
  }
  HANDLE gdata = GetClipboardData(format);
  SIZE_T size  = 0;
  if(gdata)
  {
    size = GlobalSize(gdata);
    if(target && count >= size)
    {
      void* data = GlobalLock(gdata);
      if(data)
      {
        MEMCPY(target, count, data, size);
        GlobalUnlock(gdata);
      }
      else
        size = 0;
    }
  }
  CloseClipboard();
  return (uint32_t)size;
#else
  if(kind != FG_Clipboard_Text)
    return ERR_SUCCESS;

  auto str = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(target)
    strncpy(reinterpret_cast<char*>(target), str, count);
  return strlen(str) + 1;
#endif
}

bool Backend::CheckClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind)
{
  // TODO: handle GLFW errors
#ifdef FG_PLATFORM_WIN32
  switch(kind)
  {
  case FG_Clipboard_Text: return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT);
  case FG_Clipboard_Wave: return IsClipboardFormatAvailable(CF_WAVE);
  case FG_Clipboard_Bitmap: return IsClipboardFormatAvailable(CF_BITMAP);
  case FG_Clipboard_Custom: return IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  case FG_Clipboard_All:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT) |
           IsClipboardFormatAvailable(CF_WAVE) | IsClipboardFormatAvailable(CF_BITMAP) |
           IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  }
  return false;
#else
  if(kind != FG_Clipboard_Text && kind != FG_Clipboard_All)
    return false;

  auto p = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(!p)
    return false;
  return p[0] != 0;
#endif
}

int Backend::ClearClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return ERR_CLIPBOARD_FAILURE;
  if(!EmptyClipboard())
    return ERR_CLIPBOARD_FAILURE;
  CloseClipboard();
  return ERR_SUCCESS;
#else
  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), "");
  return _lasterr;
#endif
}

int Backend::ProcessMessages(FG_Backend* self, FG_Window* window)
{
  // TODO: handle GLFW errors
  auto backend = static_cast<Backend*>(self);
  glfwPollEvents();

  // TODO: Process all joystick events

  return backend->_windows != nullptr;
}
int Backend::GetMessageSyncObject(FG_Backend* self, FG_Window* window)
{
  auto backend = static_cast<Backend*>(self);
  return ERR_NOT_IMPLEMENTED;
}

int Backend::SetCursorGL(FG_Backend* self, FG_Window* window, FG_Cursor cursor)
{
  // TODO: handle GLFW errors
  static GLFWcursor* arrow   = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  static GLFWcursor* ibeam   = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
  static GLFWcursor* cross   = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  static GLFWcursor* hand    = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
  static GLFWcursor* hresize = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
  static GLFWcursor* vresize = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);

  auto glwindow = static_cast<Window*>(window)->GetWindow();
  switch(cursor)
  {
  case FG_Cursor_Arrow: glfwSetCursor(glwindow, arrow); return _lasterr;
  case FG_Cursor_IBeam: glfwSetCursor(glwindow, ibeam); return _lasterr;
  case FG_Cursor_Cross: glfwSetCursor(glwindow, cross); return _lasterr;
  case FG_Cursor_Hand: glfwSetCursor(glwindow, hand); return _lasterr;
  case FG_Cursor_ResizeWE: glfwSetCursor(glwindow, hresize); return _lasterr;
  case FG_Cursor_ResizeNS: glfwSetCursor(glwindow, vresize); return _lasterr;
  }

  return ERR_INVALID_CURSOR;
}

int Backend::GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out)
{
  int count;
  auto monitors = glfwGetMonitors(&count);
  if(index >= static_cast<unsigned int>(count))
    return ERR_INVALID_DISPLAY;
  return GetDisplay(self, reinterpret_cast<uintptr_t>(monitors[index]), out);
}

int Backend::GetDisplay(FG_Backend* self, uintptr_t handle, FG_Display* out)
{
  if(!handle || !out)
    return ERR_MISSING_PARAMETER;
  out->handle = handle;
  // TODO: handle GLFW errors
  out->primary = glfwGetPrimaryMonitor() == reinterpret_cast<GLFWmonitor*>(handle);
  glfwGetMonitorWorkarea(reinterpret_cast<GLFWmonitor*>(handle), &out->offset.x, &out->offset.y, &out->size.x,
                         &out->size.y);
  glfwGetMonitorContentScale(reinterpret_cast<GLFWmonitor*>(handle), &out->dpi.x, &out->dpi.y);
  out->dpi.x *= BASE_DPI;
  out->dpi.y *= BASE_DPI;
  out->scale = 1.0f; // GLFW doesn't know what text scaling is
  return ERR_SUCCESS;
}

int Backend::GetDisplayWindow(FG_Backend* self, FG_Window* window, FG_Display* out)
{
  return GetDisplay(self, reinterpret_cast<uintptr_t>(glfwGetWindowMonitor(static_cast<Window*>(window)->GetWindow())),
                    out);
}

int Backend::CreateSystemControl(FG_Backend* self, FG_Context* context, const char* id, FG_Rect* area, ...)
{
  auto backend = static_cast<Backend*>(self);
  return ERR_NOT_IMPLEMENTED;
}
int Backend::SetSystemControl(FG_Backend* self, FG_Context* context, void* control, FG_Rect* area, ...)
{
  return ERR_NOT_IMPLEMENTED;
}
int Backend::DestroySystemControl(FG_Backend* self, FG_Context* context, void* control) { return ERR_NOT_IMPLEMENTED; }

GLFWglproc glGetProcAddress(const char* procname)
{
#ifdef FG_PLATFORM_WIN32
  return (GLFWglproc)GetProcAddress((HMODULE)Backend::_library, procname);
#else
  return (GLFWglproc)dlsym(Backend::_library, procname);
#endif
}

// This allows a game with a pre-existing openGL context to define a "region" for feather to draw in
/*
FG_Window* Backend::CreateRegionGL(FG_Backend* self, FG_Element* element, FG_Window desc, FG_Vec3 pos, FG_Vec3 dim)
{
  auto backend                      = reinterpret_cast<Backend*>(self);
  FG_Vec d                          = { dim.x, dim.y };
  Context* context                  = new Context(backend, element, &d);
  *static_cast<FG_Window*>(context) = desc;

  // glfwMakeContextCurrent(_window); // TODO: Make the context current
  if(!gladLoadGLLoader((GLADloadproc)glGetProcAddress))
    (*backend->_log)(backend->_root, FG_Level_Error, "gladLoadGL failed");
  backend->LogError("gladLoadGL");
  context->CreateResources();

  return context;
}
*/

int Backend::DestroyGL(FG_Backend* self)
{
  if(!self)
    return ERR_INVALID_PARAMETER;
  delete static_cast<Backend*>(self);

  if(--Backend::_refcount == 0)
  {
    Backend::_maxjoy  = 0;
    Backend::_lasterr = 0;
    glfwTerminate();
#ifdef FG_PLATFORM_WIN32
    if(Backend::_library)
      FreeLibrary((HMODULE)Backend::_library);
#endif
#ifdef FG_PLATFORM_POSIX
    if(Backend::_library)
      dlclose(Backend::_library);
#endif
  }

  return ERR_SUCCESS;
}

extern "C" FG_COMPILER_DLLEXPORT FG_Backend* fgOpenGL(void* root, FG_Log log, FG_Behavior behavior)
{
  static_assert(std::is_same<FG_InitBackend, decltype(&fgOpenGL)>::value,
                "fgOpenGL must match InitBackend function pointer");

#ifdef FG_PLATFORM_WIN32
  typedef BOOL(WINAPI * tGetPolicy)(LPDWORD lpFlags);
  typedef BOOL(WINAPI * tSetPolicy)(DWORD dwFlags);
  const DWORD EXCEPTION_SWALLOWING = 0x1;
  DWORD dwFlags;

  HMODULE kernel32 = LoadLibraryA("kernel32.dll");
  fgassert(kernel32 != 0);
  tGetPolicy pGetPolicy = (tGetPolicy)GetProcAddress(kernel32, "GetProcessUserModeExceptionPolicy");
  tSetPolicy pSetPolicy = (tSetPolicy)GetProcAddress(kernel32, "SetProcessUserModeExceptionPolicy");
  if(pGetPolicy && pSetPolicy && pGetPolicy(&dwFlags))
    pSetPolicy(dwFlags & ~EXCEPTION_SWALLOWING); // Turn off the filter
#endif

  if(++Backend::_refcount == 1)
  {
#ifdef FG_PLATFORM_WIN32
    Backend::_library = LoadLibraryA("opengl32.dll");
#endif
#ifdef FG_PLATFORM_POSIX
  #ifdef __CYGWIN__
    Backend::_library = dlopen("libGL-1.so", 0);
  #endif
    if(!Backend::_library)
      Backend::_library = dlopen("libGL.so.1", 0);
    if(!Backend::_library)
      Backend::_library = dlopen("libGL.so", 0);
#endif
    glfwSetErrorCallback(&Backend::ErrorCallback);

    if(!glfwInit())
    {
      FG_LogValue logvalues[2] = { { .type = FG_LogType_I32, .i32 = Backend::_lasterr },
                                   { .type = FG_LogType_String, .string = Backend::_lasterrdesc } };
      (*log)(root, FG_Level_Error, __FILE__, __LINE__, "glfwInit() failed!", logvalues, 2, &Backend::FreeGL);
      --Backend::_refcount;
      return nullptr;
    }

    glfwSetJoystickCallback(&Backend::JoystickCallback);
  }

  return new Backend(root, log, behavior);
}

#ifdef FG_PLATFORM_WIN32
long long GetRegistryValueW(HKEY__* hKeyRoot, const wchar_t* szKey, const wchar_t* szValue, unsigned char* data,
                            unsigned long sz)
{
  HKEY__* hKey;
  LRESULT e = RegOpenKeyExW(hKeyRoot, szKey, 0, KEY_READ, &hKey);
  if(!hKey)
    return -2;
  LSTATUS r = RegQueryValueExW(hKey, szValue, 0, 0, data, &sz);
  RegCloseKey(hKey);
  if(r == ERROR_SUCCESS)
    return sz;
  return (r == ERROR_MORE_DATA) ? sz : -1;
}
#endif

#define _STRINGIFY(x) x
#define TXT(x)        _STRINGIFY(#x)

Backend::Backend(void* root, FG_Log log, FG_Behavior behavior) :
  _root(root), _log(log), _behavior(behavior), _windows(nullptr), _insidelist(false)
{
  getCaps               = &GetCaps;
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
  dispatch              = &Dispatch;
  syncPoint             = &SyncPoint;
  setDepthStencil       = &SetDepthStencil;
  setPipelineState      = &SetPipelineState;
  setViewports          = &SetViewports;
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
  createWindow          = &CreateWindowGL;
  setWindow             = &SetWindow;
  destroyWindow         = &DestroyWindow;
  beginDraw             = &BeginDraw;
  endDraw               = &EndDraw;
  putClipboard          = &PutClipboard;
  getClipboard          = &GetClipboard;
  checkClipboard        = &CheckClipboard;
  clearClipboard        = &ClearClipboard;
  processMessages       = &ProcessMessages;
  getMessageSyncObject  = &GetMessageSyncObject;
  setCursor             = &SetCursorGL;
  getDisplayIndex       = &GetDisplayIndex;
  getDisplay            = &GetDisplay;
  getDisplayWindow      = &GetDisplayWindow;
  createSystemControl   = &CreateSystemControl;
  setSystemControl      = &SetSystemControl;
  destroySystemControl  = &DestroySystemControl;
  destroy               = &DestroyGL;

  this->LOG(FG_Level_Notice, "Initializing fgOpenGL...");

#ifdef FG_PLATFORM_WIN32
  #ifdef FG_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
  #endif

  long long sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate",
                           reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      cursorblink = _wtoi(buf.data());
  }
  else
    this->LOG(FG_Level_Warning, "Couldn't get user's cursor blink rate.");

  sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime", 0, 0);
  if(sz > 0)
  {
    std::wstring buf;
    buf.resize(sz / 2);
    sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Mouse", L"MouseHoverTime",
                           reinterpret_cast<unsigned char*>(buf.data()), (unsigned long)sz);
    if(sz > 0)
      tooltipdelay = _wtoi(buf.data());
  }
  else
    this->LOG(FG_Level_Warning, "Couldn't get user's mouse hover time.");

#else
  cursorblink  = 530;
  tooltipdelay = 500;
#endif
  _singleton = this;
}

Backend::~Backend()
{
  while(_windows)
  {
    auto p = _windows->_next;
    delete _windows;
    _windows = p;
  }

  if(_singleton == this)
    _singleton = nullptr;

#ifdef FG_PLATFORM_WIN32
  PostQuitMessage(0);
#endif
}
void Backend::ErrorCallback(int error, const char* description)
{
  _lasterr = error;
#ifdef _WIN32
  strncpy_s(_lasterrdesc, description, 1024);
#else
  strncpy(_lasterrdesc, description, 1024);
#endif

  if(_singleton)
    _singleton->LOG(FG_Level_Error, description);
}

void Backend::JoystickCallback(int id, int connected)
{
  // This only keeps an approximate count of the highest joy ID because internally, GLFW has it's own array of joysticks
  // that it derives the ID from anyway, with a maximum of 16.
  if(connected && id > _maxjoy)
    _maxjoy = id;
  if(!connected && id == _maxjoy)
    --_maxjoy;
}
