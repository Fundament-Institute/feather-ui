// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "platform.h"
#include "Backend.h"
#include <filesystem>

using namespace GL;
namespace fs = std::filesystem;

namespace GL {
  __KHASH_IMPL(assets, , const FG_Asset*, char, 0, kh_ptr_hash_func, kh_int_hash_equal);
}

int Backend::_lasterr            = 0;
int Backend::_refcount           = 0;
char Backend::_lasterrdesc[1024] = {};
int Backend::_maxjoy             = 0;

void Backend::DrawBoundBuffer(Shader* shader, GLuint instance, size_t stride, GLsizei count, const Attribute* data,
                              size_t n_data, int primitive)
{
  glUseProgram(instance);
  LogError("glUseProgram");

  shader->SetVertices(this, instance, stride);
  for(size_t i = 0; i < n_data; ++i)
    Shader::SetUniform(instance, data[i]);
   
  glDrawArrays(primitive, 0, count);
  LogError("glDrawArrays");
}

FG_Err Backend::DrawTextGL(FG_Backend* self, void* window, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color color,
                           float lineHeight, float letterSpacing, float blur, FG_AntiAliasing aa)
{
  return -1;
}

FG_Err Backend::DrawAsset(FG_Backend* self, void* window, FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color,
                          float time)
{
  return -1;
}

FG_Err Backend::DrawRect(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor, float border,
                         FG_Color borderColor, float blur, FG_Asset* asset)
{
  return -1;
}

FG_Err Backend::DrawCircle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* arcs, FG_Color fillColor, float border,
                           FG_Color borderColor, float blur, FG_Asset* asset)
{
  return -1;
}
FG_Err Backend::DrawTriangle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                             float border, FG_Color borderColor, float blur, FG_Asset* asset)
{
  return -1;
}

FG_Err Backend::DrawLines(FG_Backend* self, void* window, FG_Vec* points, uint32_t count, FG_Color color) { return -1; }

FG_Err Backend::DrawCurve(FG_Backend* self, void* window, FG_Vec* anchors, uint32_t count, FG_Color fillColor, float stroke,
                          FG_Color strokeColor)
{
  auto instance = static_cast<Backend*>(self);

  return -1;
}

// FG_Err DrawShader(FG_Backend* self, fgShader);
FG_Err Backend::PushLayer(FG_Backend* self, void* window, FG_Rect* area, float* transform, float opacity, void* cache)
{
  if(!area)
    return -1;
  reinterpret_cast<Context*>(window)->PushLayer(*area, transform, opacity, reinterpret_cast<Layer*>(cache));
  return 0;
}

void* Backend::PopLayer(FG_Backend* self, void* window)
{
  return !window ? 0 : reinterpret_cast<Context*>(window)->PopLayer();
}
FG_Err Backend::DestroyLayer(FG_Backend* self, void* window, void* layer)
{
  if(!layer)
    return -1;

  delete reinterpret_cast<Layer*>(layer);
  return 0;
}

FG_Err Backend::PushClip(FG_Backend* self, void* window, FG_Rect* area)
{
  if(!area)
    return -1;
  reinterpret_cast<Context*>(window)->PushClip(*area);
  return 0;
}

FG_Err Backend::PopClip(FG_Backend* self, void* window)
{
  reinterpret_cast<Context*>(window)->PopClip();
  return 0;
}

FG_Err Backend::DirtyRect(FG_Backend* self, void* window, void* layer, FG_Rect* area) { return -1; }

FG_Font* Backend::CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                               FG_Vec dpi)
{
  return nullptr;
}

FG_Err Backend::DestroyFont(FG_Backend* self, FG_Font* font) { return -1; }
FG_Err Backend::DestroyLayout(FG_Backend* self, void* layout) { return -1; }

void* Backend::FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                          float letterSpacing, void* prev, FG_Vec dpi)
{
  return nullptr;
}

uint32_t Backend::FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, float lineHeight,
                            float letterSpacing, FG_Vec pos, FG_Vec* cursor, FG_Vec dpi)
{
  return ~0;
}

FG_Vec Backend::FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, float lineHeight,
                        float letterSpacing, uint32_t index, FG_Vec dpi)
{
  return FG_Vec{};
}

FG_Asset* Backend::CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format)
{
  auto backend     = static_cast<Backend*>(self);
  Asset* asset     = reinterpret_cast<Asset*>(malloc(count + sizeof(Asset)));
  asset->data.data = asset + 1;
  MEMCPY(asset->data.data, count, data, count);
  asset->count  = count;
  asset->format = format;
  int r;
  kh_put_assets(backend->_assethash, asset, &r);
  return asset;
}

FG_Err Backend::DestroyAsset(FG_Backend* self, FG_Asset* fgasset)
{
  auto backend = static_cast<Backend*>(self);

  // Once we remove the asset from the hash, contexts who reach a reference count of 0 for an asset will destroy it.
  khiter_t iter = kh_get_assets(backend->_assethash, fgasset);
  if(iter < kh_end(backend->_assethash))
    kh_del_assets(backend->_assethash, iter);

  free(static_cast<Asset*>(fgasset));
  return 0;
}

FG_Err Backend::PutClipboard(FG_Backend* self, FG_Clipboard kind, const char* data, uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return -1;
  if(data != 0 && count > 0 && EmptyClipboard())
  {
    if(kind == FG_Clipboard_TEXT)
    {
      size_t unilen  = MultiByteToWideChar(CP_UTF8, 0, data, (int)count, 0, 0);
      HGLOBAL unimem = GlobalAlloc(GMEM_MOVEABLE, unilen * sizeof(wchar_t));
      if(unimem)
      {
        wchar_t* uni = (wchar_t*)GlobalLock(unimem);
        size_t sz    = MultiByteToWideChar(CP_UTF8, 0, data, (int)count, uni, (int)unilen);
        if(sz < unilen) // ensure we have a null terminator
          uni[sz] = 0;
        GlobalUnlock(unimem);
        SetClipboardData(CF_UNICODETEXT, unimem);
      }
      HGLOBAL gmem = GlobalAlloc(GMEM_MOVEABLE, count + 1);
      if(gmem)
      {
        char* mem = (char*)GlobalLock(gmem);
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
  return 0;
#else
  return -1;
#endif
}

uint32_t Backend::GetClipboard(FG_Backend* self, FG_Clipboard kind, void* target, uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return 0;
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
          len = WideCharToMultiByte(CP_UTF8, 0, str, (int)size, 0, 0, NULL, NULL);

          if(target && count >= len)
            len = WideCharToMultiByte(CP_UTF8, 0, str, (int)size, (char*)target, (int)count, NULL, NULL);

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
  glfwGetClipboardString() return -1;
#endif
}

bool Backend::CheckClipboard(FG_Backend* self, FG_Clipboard kind)
{
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
  return glfwGetClipboardString() != nullptr;
#endif
}

FG_Err Backend::ClearClipboard(FG_Backend* self, FG_Clipboard kind)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return -1;
  if(!EmptyClipboard())
    return -1;
  CloseClipboard();
  return 0;
#else
  glfwSetClipboardString(nullptr, "");
  return 0;
#endif
}

FG_Err Backend::ProcessMessages(FG_Backend* self)
{
  auto backend = static_cast<Backend*>(self);
  glfwPollEvents();

  // TODO: Process all joystick events

  return 1; // GLFW eats WM_QUIT and just closes all windows, so the application will have to detect this
}

FG_Err Backend::SetCursorGL(FG_Backend* self, void* window, FG_Cursor cursor)
{
  static GLFWcursor* arrow   = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  static GLFWcursor* ibeam   = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
  static GLFWcursor* cross   = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  static GLFWcursor* hand    = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
  static GLFWcursor* hresize = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
  static GLFWcursor* vresize = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);

  auto glwindow = reinterpret_cast<Context*>(window)->GetWindow();
  switch(cursor)
  {
  case FG_Cursor_ARROW: glfwSetCursor(glwindow, arrow); return _lasterr;
  case FG_Cursor_IBEAM: glfwSetCursor(glwindow, ibeam); return _lasterr;
  case FG_Cursor_CROSS: glfwSetCursor(glwindow, cross); return _lasterr;
  case FG_Cursor_HAND: glfwSetCursor(glwindow, hand); return _lasterr;
  case FG_Cursor_RESIZEWE: glfwSetCursor(glwindow, hresize); return _lasterr;
  case FG_Cursor_RESIZENS: glfwSetCursor(glwindow, vresize); return _lasterr;
  }

  return -1;
}

FG_Err Backend::RequestAnimationFrame(FG_Backend* self, void* window, unsigned long long microdelay)
{
  if(!window)
    return -1;
  reinterpret_cast<Context*>(window)->Draw(nullptr);
  return 0;
}

FG_Err Backend::GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out)
{
  int count;
  auto monitors = glfwGetMonitors(&count);
  if(index >= static_cast<unsigned int>(count))
    return -1;
  return GetDisplay(self, monitors[index], out);
}

FG_Err Backend::GetDisplay(FG_Backend* self, void* handle, FG_Display* out)
{
  if(!handle || !out)
    return -1;
  out->handle  = handle;
  out->primary = glfwGetPrimaryMonitor() == handle;
  glfwGetMonitorWorkarea(reinterpret_cast<GLFWmonitor*>(handle), &out->offset.x, &out->offset.y, &out->size.x,
                         &out->size.y);
  glfwGetMonitorContentScale(reinterpret_cast<GLFWmonitor*>(handle), &out->dpi.x, &out->dpi.y);
  out->dpi.x *= 96.0f;
  out->dpi.y *= 96.0f;
  out->scale = 1.0f; // GLFW doesn't know what text scaling is
  return 0;
}

FG_Err Backend::GetDisplayWindow(FG_Backend* self, void* window, FG_Display* out)
{
  return GetDisplay(self, glfwGetWindowMonitor(reinterpret_cast<Context*>(window)->GetWindow()), out);
}

void* Backend::CreateWindowGL(FG_Backend* self, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                              const char* caption, uint64_t flags, void* context)
{
  auto backend = reinterpret_cast<Backend*>(self);
  _lasterr     = 0;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  Window* window = new Window(static_cast<Backend*>(self), reinterpret_cast<GLFWmonitor*>(display), element, pos, dim,
                              flags, caption, context);

  if(!_lasterr)
  {
    window->_next     = backend->_windows;
    backend->_windows = window;
    if(window->_next)
      window->_next->_prev = window;

    return static_cast<Context*>(window);
  }

  delete window;
  return nullptr;
}

FG_Err Backend::SetWindowGL(FG_Backend* self, void* window, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                            const char* caption, uint64_t flags)
{
  if(!window)
    return -1;
  auto context      = reinterpret_cast<Context*>(window);
  context->_element = element;

  auto glwindow = context->GetWindow();
  if(!glwindow)
    return 0;
  auto w = static_cast<Window*>(context);
  if(caption)
    glfwSetWindowTitle(glwindow, caption);

  if((w->_flags ^ flags) & FG_Window_FULLSCREEN) // If we toggled fullscreen we need a different code path
  {
    int posx, posy;
    int dimx, dimy;
    glfwGetWindowPos(glwindow, &posx, &posy);
    glfwGetWindowSize(glwindow, &dimx, &dimy);
    if(flags & FG_Window_FULLSCREEN)
      glfwSetWindowMonitor(glwindow, (GLFWmonitor*)display, 0, 0, !dim ? dimx : (int)dim->x, !dim ? dimy : (int)dim->y,
                           GLFW_DONT_CARE);
    else
      glfwSetWindowMonitor(glwindow, NULL, !pos ? posx : (int)pos->x, !pos ? posy : (int)pos->y, !dim ? dimx : (int)dim->x,
                           !dim ? dimy : (int)dim->y, GLFW_DONT_CARE);
  }
  else
  {
    if(pos)
      glfwSetWindowPos(glwindow, (int)pos->x, (int)pos->y);
    if(dim)
      glfwSetWindowSize(glwindow, (int)dim->x, (int)dim->y);
  }

  if(flags & FG_Window_MAXIMIZED)
    glfwMaximizeWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  if(flags & FG_Window_MINIMIZED)
    glfwIconifyWindow(glwindow);
  else
    glfwRestoreWindow(glwindow);

  w->_flags = flags;
  return 0;
}

FG_Err Backend::DestroyWindow(FG_Backend* self, void* window)
{
  if(!window)
    return -1;
  _lasterr = 0;
  delete reinterpret_cast<Context*>(window);
  return _lasterr;
}
FG_Err Backend::BeginDraw(FG_Backend* self, void* window, FG_Rect* area, bool clear)
{
  if(!window)
    return -1;
  _lasterr = 0;
  reinterpret_cast<Context*>(window)->BeginDraw(area, clear);
  return _lasterr;
}
FG_Err Backend::EndDraw(FG_Backend* self, void* window)
{
  _lasterr = 0;
  reinterpret_cast<Context*>(window)->EndDraw();
  return _lasterr;
}

void DestroyGL(FG_Backend* self)
{
  auto gl = static_cast<Backend*>(self);
  if(!gl)
    return;

  gl->~Backend();
  free(gl);

  if(--Backend::_refcount == 0)
  {
    Backend::_maxjoy  = 0;
    Backend::_lasterr = 0;
    glfwTerminate();
  }
}

bool Backend::LogError(const char* call)
{
  int err = glGetError();
  if(err != GL_NO_ERROR)
  {
    (*_log)(_root, FG_Level_ERROR, "%s: 0x%04X", call, err);
    return true;
  }

  return false;
}

#ifdef FG_DEBUG
  #define FG_MAIN_FUNCTION fgOpenGL_d
#else
  #define FG_MAIN_FUNCTION fgOpenGL
#endif

extern "C" FG_COMPILER_DLLEXPORT FG_Backend* FG_MAIN_FUNCTION(void* root, FG_Log log, FG_Behavior behavior)
{
  static_assert(std::is_same<FG_InitBackend, decltype(&(FG_MAIN_FUNCTION))>::value,
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
    if(!glfwInit())
    {
      (*log)(root, FG_Level_ERROR, "glfwInit() failed!");
      --Backend::_refcount;
      return nullptr;
    }

    glfwSetErrorCallback(&Backend::ErrorCallback);
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

Backend::Backend(void* root, FG_Log log, FG_Behavior behavior) :
  _root(root), _log(log), _behavior(behavior), _assethash(kh_init_assets()), _windows(nullptr)
{
  drawText     = &DrawTextGL;
  drawAsset    = &DrawAsset;
  drawRect     = &DrawRect;
  drawCircle   = &DrawCircle;
  drawTriangle = &DrawTriangle;
  drawLines    = &DrawLines;
  drawCurve    = &DrawCurve;
  // drawShader =&DrawShader;
  pushLayer             = &PushLayer;
  popLayer              = &PopLayer;
  pushClip              = &PushClip;
  popClip               = &PopClip;
  dirtyRect             = &DirtyRect;
  beginDraw             = &BeginDraw;
  endDraw               = &EndDraw;
  createFont            = &CreateFontGL;
  destroyFont           = &DestroyFont;
  fontLayout            = &FontLayout;
  destroyLayout         = &DestroyLayout;
  fontIndex             = &FontIndex;
  fontPos               = &FontPos;
  createAsset           = &CreateAsset;
  destroyAsset          = &DestroyAsset;
  putClipboard          = &PutClipboard;
  getClipboard          = &GetClipboard;
  checkClipboard        = &CheckClipboard;
  clearClipboard        = &ClearClipboard;
  processMessages       = &ProcessMessages;
  setCursor             = &SetCursorGL;
  requestAnimationFrame = &RequestAnimationFrame;
  getDisplayIndex       = &GetDisplayIndex;
  getDisplay            = &GetDisplay;
  getDisplayWindow      = &GetDisplayWindow;
  createWindow          = &CreateWindowGL;
  setWindow             = &SetWindowGL;
  destroyWindow         = &DestroyWindow;
  destroy               = &DestroyGL;

  (*_log)(_root, FG_Level_NONE, "Initializing fgOpenGL...");

  static const char* vertex_shader_text = "#version 110\n"
                                          "uniform mat4 MVP;\n"
                                          "attribute vec3 vCol;\n"
                                          "attribute vec2 vPos;\n"
                                          "varying vec3 color;\n"
                                          "void main()\n"
                                          "{\n"
                                          "    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
                                          "    color = vCol;\n"
                                          "}\n";

  static const char* fragment_shader_text = "#version 110\n"
                                            "varying vec3 color;\n"
                                            "void main()\n"
                                            "{\n"
                                            "    gl_FragColor = vec4(color, 1.0);\n"
                                            "}\n";

  _imageshader = Shader(fragment_shader_text, vertex_shader_text, 0,
                        Shader::Data{ Shader::VERTEX, GL_FLOAT_VEC3, "vCol", 2 * sizeof(float) },
                        Shader::Data{ Shader::VERTEX, GL_FLOAT_VEC2, "vPos", 0 },
                        Shader::Data{ Shader::GLOBAL, GL_FLOAT_MAT4, "MVP" });

#ifdef FG_PLATFORM_WIN32
  #ifdef FG_DEBUG
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);
  #endif

  int64_t sz = GetRegistryValueW(HKEY_CURRENT_USER, L"Control Panel\\Desktop", L"CursorBlinkRate", 0, 0);
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
    (*_log)(_root, FG_Level_WARNING, "Couldn't get user's cursor blink rate.");

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
    (*_log)(_root, FG_Level_WARNING, "Couldn't get user's mouse hover time.");
#else
  cursorblink  = 530;
  tooltipdelay = 500;
#endif
}

Backend::~Backend()
{
  while(_windows)
  {
    auto p = _windows->_next;
    delete _windows;
    _windows = p;
  }

#ifdef FG_PLATFORM_WIN32
  PostQuitMessage(0);
#endif

  kh_destroy_assets(_assethash);
}
void Backend::ErrorCallback(int error, const char* description)
{
  _lasterr = error;
  strncpy(_lasterrdesc, description, 1024);
}

FG_Result Backend::Behavior(Context* w, const FG_Msg& msg)
{
  return (*_behavior)(w->_element, w, _root, const_cast<FG_Msg*>(&msg));
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
