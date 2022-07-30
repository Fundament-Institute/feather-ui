// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "BackendGL.hpp"

#include "platform.hpp"
#include "BackendGL.hpp"
#include <cfloat>
#include "ShaderObject.hpp"
#include "ProgramObject.hpp"
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
  caps.OpenGL.features = FG_Feature_API_OPENGL | FG_Feature_IMMEDIATE_MODE | FG_Feature_BACKGROUND_OPACITY |
                         FG_Feature_LINES_ALPHA | FG_Feature_BLEND_GAMMA | FG_Feature_INSTANCING |
                         FG_Feature_INDEPENDENT_BLEND;
  caps.OpenGL.version           = 20;
  caps.OpenGL.glsl              = 110;
  caps.OpenGL.max_rendertargets = 8;
  return caps;
}

void* Backend::CompileShader(FG_Backend* self, FG_Context* context, enum FG_ShaderStage stage, const char* source)
{
  if(stage >= ArraySize(ShaderStageMapping))
    CUSTOM_ERROR(ERR_INVALID_PARAMETER, "Unsupported shader stage").log(static_cast<Backend*>(self));
  else
  {
    // Note: Can't use LOG_ERROR here because 
    if(auto r = ShaderObject::create(source, ShaderStageMapping[stage]))
      return pack_ptr(std::move(r.value()).release());
    else
      r.error().log(static_cast<Backend*>(self));
  }
  return nullptr;
}
int Backend::DestroyShader(FG_Backend* self, FG_Context* context, void* shader)
{
  auto backend = static_cast<Backend*>(self);

  // ShaderObject takes ownership of this and deletes it once it's destructed.
  ShaderObject s(shader);
  if(!s.is_valid())
  {
    backend->LOG(FG_Level_ERROR, "Invalid shader!", s.release());
    return ERR_INVALID_PARAMETER;
  }
  return 0;
}

void* Backend::CreateCommandList(FG_Backend* self, FG_Context* context, bool bundle)
{
  auto backend = static_cast<Backend*>(self);
  if(backend->_insidelist)
  {
    backend->LOG(FG_Level_ERROR,
                 "This backend can't do multiple command lists at the same time! Did you forget to free the old list?");
    return nullptr;
  }
  backend->_insidelist = true;
  return context;
}

int Backend::DestroyCommandList(FG_Backend* self, void* commands)
{
  auto backend = static_cast<Backend*>(self);
  if(!commands)
  {
    backend->LOG(FG_Level_ERROR, "Expected a non-null command list but got NULL instead!");
    return ERR_INVALID_PARAMETER;
  }
  if(!backend->_insidelist)
  {
    backend->LOG(FG_Level_ERROR, "Mismatched CreateCommandList / DestroyCommandList pair !");
    return ERR_INVALID_CALL;
  }

  backend->_insidelist = false;
  return ERR_SUCCESS;
}

int Backend::ClearDepthStencil(FG_Backend* self, void* commands, FG_Resource* depthstencil, char clear, uint8_t stencil,
                               float depth, uint32_t num_rects, FG_Rect* rects)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  // TODO: bind depthstencil only if necessary
  glClearDepth(depth);
  if(auto e = glGetError())
    return e;
  glClearStencil(stencil);
  if(auto e = glGetError())
    return e;

  // TODO: iterate through rects and set scissor rects for each one and clear.

  // clear 0 clears both, clear 1 is just depth, clear -1 is just stencil
  switch(clear)
  {
  case 0: glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  case 1: glClear(GL_DEPTH_BUFFER_BIT);
  case -1: glClear(GL_STENCIL_BUFFER_BIT);
  }

  return glGetError();
}
int Backend::ClearRenderTarget(FG_Backend* self, void* commands, FG_Resource* rendertarget, FG_Color RGBA,
                               uint32_t num_rects, FG_Rect* rects)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);

  // TODO: bind rendertarget only if necessary
  std::array<float, 4> colors;
  Context::ColorFloats(RGBA, colors, false);
  glClearColor(colors[0], colors[1], colors[2], colors[3]);
  if(auto e = glGetError())
    return e;

  // TODO: iterate through num_rects and apply scissor rect
  glClear(GL_COLOR_BUFFER_BIT);
  return glGetError();
}
int Backend::CopyResource(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest)
{
  auto backend = static_cast<Backend*>(self);
  return ERR_NOT_IMPLEMENTED;
}
int Backend::CopySubresource(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest, unsigned long srcoffset,
                             unsigned long destoffset, unsigned long bytes)
{
  auto backend = static_cast<Backend*>(self);
  return ERR_NOT_IMPLEMENTED;
}
int Backend::CopyResourceRegion(FG_Backend* self, void* commands, FG_Resource* src, FG_Resource* dest, FG_Vec3i srcoffset,
                                FG_Vec3i destoffset, FG_Vec3i size)
{
  auto backend = static_cast<Backend*>(self);
  return ERR_NOT_IMPLEMENTED;
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
int Backend::SetPipelineState(FG_Backend* self, void* commands, void* state)
{
  if(!commands || !state)
    return ERR_INVALID_PARAMETER;

  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(commands);
  LOG_ERROR(backend, reinterpret_cast<PipelineState*>(state)->apply(context));
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

void* Backend::CreatePipelineState(FG_Backend* self, FG_Context* context, FG_PipelineState* pipelinestate,
                                   FG_Resource** rendertargets, uint32_t n_targets, FG_Blend* blends,
                                   FG_Resource** vertexbuffer, GLsizei* strides, uint32_t n_buffers,
                                   FG_ShaderParameter* attributes, uint32_t n_attributes, FG_Resource* indexbuffer,
                                   uint8_t indexstride)
{
  if(!pipelinestate || !context || !blends)
    return nullptr;

  auto backend = static_cast<Backend*>(self);
  auto ctx     = reinterpret_cast<Context*>(context);
  
  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = PipelineState::create(*pipelinestate, std::span(rendertargets, n_targets), *blends,
                                    std::span(vertexbuffer, n_buffers), strides, std::span(attributes, n_attributes),
                                    indexbuffer, indexstride))
    return e.value();
  else
    e.log(backend);
  return nullptr;
}

int Backend::DestroyPipelineState(FG_Backend* self, FG_Context* context, void* state)
{
  auto backend = static_cast<Backend*>(self);
  delete reinterpret_cast<FG_PipelineState*>(state);
  return ERR_NOT_IMPLEMENTED;
}

FG_Resource* Backend::CreateBuffer(FG_Backend* self, FG_Context* context, void* data, uint32_t bytes, enum FG_Type type)
{
  auto backend = static_cast<Backend*>(self);
  if(type >= ArraySize(TypeMapping))
    return nullptr;

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = Buffer::create(TypeMapping[type], data, bytes))
    return pack_ptr(std::move(e.value()).release());
  else
    e.log(backend);
  return nullptr;
}
FG_Resource* Backend::CreateTexture(FG_Backend* self, FG_Context* context, FG_Vec2i size, enum FG_Type type,
                                    enum FG_PixelFormat format, FG_Sampler* sampler, void* data, int MultiSampleCount)
{
  auto backend = static_cast<Backend*>(self);
  if(type >= ArraySize(TypeMapping) || !sampler)
    return nullptr;

  // Can't use LOG_ERROR here because we return a pointer.
  if(auto e = Texture::create2D(TypeMapping[type], Format::Create(format, false), size, *sampler, data, MultiSampleCount))
    return pack_ptr(std::move(e.value()).release());
  else
    e.log(backend);
  return nullptr;
}
FG_Resource* Backend::CreateRenderTarget(FG_Backend* self, FG_Context* context, FG_Resource** textures, uint32_t n_textures)
{
  auto backend = static_cast<Backend*>(self);
  
  if(auto e = FrameBuffer::create(GL_FRAMEBUFFER, GL_TEXTURE_2D, 0, 0, textures, n_textures))
    return pack_ptr(std::move(e.value()).release());
  else
    e.log(backend);
  return nullptr;
}
int Backend::DestroyResource(FG_Backend* self, FG_Context* context, FG_Resource* resource)
{
  auto backend = static_cast<Backend*>(self);
  auto i       = unpack_ptr<GLuint>(resource);
  if(glIsBuffer(i))
    Buffer b(i);
  else if(glIsTexture(i))
    Texture t(i);
  else if(glIsFramebuffer(i))
    FrameBuffer fb(i);
  else
    return ERR_INVALID_PARAMETER;

  return ERR_SUCCESS;
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

  if((w->_flags ^ flags) & FG_WindowFlag_FULLSCREEN) // If we toggled fullscreen we need a different code path
  {
    int posx, posy;
    int dimx, dimy;
    glfwGetWindowPos(glwindow, &posx, &posy);
    glfwGetWindowSize(glwindow, &dimx, &dimy);
    if(flags & FG_WindowFlag_FULLSCREEN)
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

  if(flags & FG_WindowFlag_MAXIMIZED)
    glfwMaximizeWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  if(flags & FG_WindowFlag_MINIMIZED)
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
    if(kind == FG_Clipboard_TEXT)
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
        case FG_Clipboard_WAVE: format = CF_WAVE; break;
        case FG_Clipboard_BITMAP: format = CF_BITMAP; break;
        }
        SetClipboardData(format, gmem);
      }
    }
  }
  CloseClipboard();
  return ERR_SUCCESS;
#else
  if(kind != FG_Clipboard_TEXT)
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
  case FG_Clipboard_TEXT:
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
  case FG_Clipboard_WAVE: format = CF_WAVE; break;
  case FG_Clipboard_BITMAP: format = CF_BITMAP; break;
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
  if(kind != FG_Clipboard_TEXT)
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
  case FG_Clipboard_TEXT: return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT);
  case FG_Clipboard_WAVE: return IsClipboardFormatAvailable(CF_WAVE);
  case FG_Clipboard_BITMAP: return IsClipboardFormatAvailable(CF_BITMAP);
  case FG_Clipboard_CUSTOM: return IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  case FG_Clipboard_ALL:
    return IsClipboardFormatAvailable(CF_TEXT) | IsClipboardFormatAvailable(CF_UNICODETEXT) |
           IsClipboardFormatAvailable(CF_WAVE) | IsClipboardFormatAvailable(CF_BITMAP) |
           IsClipboardFormatAvailable(CF_PRIVATEFIRST);
  }
  return false;
#else
  if(kind != FG_Clipboard_TEXT && kind != FG_Clipboard_ALL)
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
  case FG_Cursor_ARROW: glfwSetCursor(glwindow, arrow); return _lasterr;
  case FG_Cursor_IBEAM: glfwSetCursor(glwindow, ibeam); return _lasterr;
  case FG_Cursor_CROSS: glfwSetCursor(glwindow, cross); return _lasterr;
  case FG_Cursor_HAND: glfwSetCursor(glwindow, hand); return _lasterr;
  case FG_Cursor_RESIZEWE: glfwSetCursor(glwindow, hresize); return _lasterr;
  case FG_Cursor_RESIZENS: glfwSetCursor(glwindow, vresize); return _lasterr;
  }

  return ERR_INVALID_CURSOR;
}

int Backend::GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out)
{
  int count;
  auto monitors = glfwGetMonitors(&count);
  if(index >= static_cast<unsigned int>(count))
    return ERR_INVALID_DISPLAY;
  return GetDisplay(self, monitors[index], out);
}

int Backend::GetDisplay(FG_Backend* self, void* handle, FG_Display* out)
{
  if(!handle || !out)
    return ERR_MISSING_PARAMETER;
  out->handle  = handle;
  // TODO: handle GLFW errors
  out->primary = glfwGetPrimaryMonitor() == handle;
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
  return GetDisplay(self, glfwGetWindowMonitor(static_cast<Window*>(window)->GetWindow()), out);
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
    (*backend->_log)(backend->_root, FG_Level_ERROR, "gladLoadGL failed");
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
      (*log)(root, FG_Level_ERROR, __FILE__, __LINE__, "glfwInit() failed!", logvalues, 2, &Backend::FreeGL);
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
  getCaps              = &GetCaps;
  compileShader        = &CompileShader;
  destroyShader        = &DestroyShader;
  createCommandList    = &CreateCommandList;
  destroyCommandList   = &DestroyCommandList;
  clearDepthStencil    = &ClearDepthStencil;
  clearRenderTarget    = &ClearRenderTarget;
  copyResource         = &CopyResource;
  copySubresource      = &CopySubresource;
  copyResourceRegion   = &CopyResourceRegion;
  draw                 = &DrawGL;
  drawIndexed          = &DrawIndexed;
  setDepthStencil      = &SetDepthStencil;
  setPipelineState     = &SetPipelineState;
  setViewports         = &SetViewports;
  setShaderConstants   = &SetShaderConstants;
  execute              = &Execute;
  createPipelineState  = &CreatePipelineState;
  destroyPipelineState = &DestroyPipelineState;
  createBuffer         = &CreateBuffer;
  createTexture        = &CreateTexture;
  createRenderTarget   = &CreateRenderTarget;
  destroyResource      = &DestroyResource;
  createWindow         = &CreateWindowGL;
  setWindow            = &SetWindow;
  destroyWindow        = &DestroyWindow;
  beginDraw            = &BeginDraw;
  endDraw              = &EndDraw;
  putClipboard         = &PutClipboard;
  getClipboard         = &GetClipboard;
  checkClipboard       = &CheckClipboard;
  clearClipboard       = &ClearClipboard;
  processMessages      = &ProcessMessages;
  getMessageSyncObject = &GetMessageSyncObject;
  setCursor            = &SetCursorGL;
  getDisplayIndex      = &GetDisplayIndex;
  getDisplay           = &GetDisplay;
  getDisplayWindow     = &GetDisplayWindow;
  createSystemControl  = &CreateSystemControl;
  setSystemControl     = &SetSystemControl;
  destroySystemControl = &DestroySystemControl;
  destroy              = &DestroyGL;

  this->LOG(FG_Level_NONE, "Initializing fgOpenGL...");

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
    this->LOG(FG_Level_WARNING, "Couldn't get user's cursor blink rate.");

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
    this->LOG(FG_Level_WARNING, "Couldn't get user's mouse hover time.");

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
    _singleton->LOG(FG_Level_ERROR, description);
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
