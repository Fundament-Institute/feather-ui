// Copyright (c)2022 Fundament Software
// For conditions of distribution and use, see copyright notice in "Bridge.hpp"

#include "platform.hpp"
#include "Bridge.hpp"
#include <cfloat>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using FG::Bridge;

int Bridge::_refcount      = 0;
Bridge* Bridge::_singleton = nullptr;
void* Bridge::_library     = nullptr;

decltype(Bridge::_wglGetProcAddress) Bridge::_wglGetProcAddress = nullptr;

FG_Window* Bridge::EmplaceContext(struct FG_GraphicsDesktopBridge* self, FG_Window* window, enum FG_PixelFormat backbuffer)
{
  // This basically does nothing right now because GLFW actually initializes the openGL context. When we change to proper
  // window initialization, we will actually be creating and attaching the openGL context in here.
  auto b = static_cast<Bridge*>(self);
  auto w = static_cast<GLFW::Window*>(window);
  w->MakeCurrent();
  b->_provider->LoadGL(wglLoadProc);
  auto dim        = w->GetSize();
  window->context = new GL::Context(FG_Vec2{ static_cast<float>(dim.x), static_cast<float>(dim.y) });
  return w;
}
FG_Window* Bridge::AttachContext(struct FG_GraphicsDesktopBridge* self, FG_Context* context, FG_Window* window)
{
  return nullptr;
}
int Bridge::BeginDraw(struct FG_GraphicsDesktopBridge* self, FG_Window* window, FG_Rect* area)
{
  auto b = static_cast<Bridge*>(self);
  auto w = static_cast<GLFW::Window*>(window);
  w->MakeCurrent();
  if(auto e = static_cast<GL::Context*>(window->context)->BeginDraw(area); !e)
  {
    e.log(b->_provider);
  }
  return GL::ERR_SUCCESS;
}
int Bridge::EndDraw(struct FG_GraphicsDesktopBridge* self, FG_Window* window)
{
  auto b = static_cast<Bridge*>(self);
  auto w = static_cast<GLFW::Window*>(window);
  w->SwapBuffers();
  if(auto e = static_cast<GL::Context*>(window->context)->EndDraw(); !e)
  {
    e.log(b->_provider);
  }
  return GL::ERR_SUCCESS;
}
int Bridge::DestroyImpl(struct FG_GraphicsDesktopBridge* self)
{
  if(!self)
    return GL::ERR_INVALID_PARAMETER;
  delete static_cast<Bridge*>(self);

  if(--Bridge::_refcount == 0)
  {
#ifdef FG_PLATFORM_WIN32
    if(Bridge::_library)
      FreeLibrary((HMODULE)Bridge::_library);
#endif
#ifdef FG_PLATFORM_POSIX
    if(Bridge::_library)
      dlclose(Bridge::_library);
#endif
    Bridge::_wglGetProcAddress = nullptr;
  }

  return GL::ERR_SUCCESS;
}

void* Bridge::wglLoadProc(const char* name)
{
  auto proc = Bridge::_wglGetProcAddress(name);
  if(proc)
    return proc;

  return LoadProc(Bridge::_library, name);
}

void* Bridge::LoadProc(void* library, const char* name)
{
#ifdef FG_PLATFORM_WIN32
  return GetProcAddress(static_cast<HMODULE>(library), name);
#endif
#ifdef FG_PLATFORM_POSIX
  return dlsym(library, name);
#endif
}

extern "C" FG_COMPILER_DLLEXPORT FG_GraphicsDesktopBridge* fgOpenGLDesktopBridge(FG_GraphicsInterface* graphics,
                                                                                 void* log_context, FG_Log log)
{
  static_assert(std::is_same<FG_InitGraphicsDesktopBridge, decltype(&fgOpenGLDesktopBridge)>::value,
                "fgOpenGLDesktopBridge must match InitGraphicsDesktopBridge function pointer");

  if(++Bridge::_refcount == 1)
  {
#ifdef FG_PLATFORM_WIN32
    Bridge::_library = LoadLibraryA("opengl32.dll");
#endif
#ifdef FG_PLATFORM_POSIX
  #ifdef __CYGWIN__
    Bridge::_library = dlopen("libGL-1.so", 0);
  #endif
    if(!Bridge::_library)
      Bridge::_library = dlopen("libGL.so.1", 0);
    if(!Provider::_library)
      Bridge::_library = dlopen("libGL.so", 0);
#endif

    Bridge::_wglGetProcAddress =
      reinterpret_cast<decltype(Bridge::_wglGetProcAddress)>(Bridge::LoadProc(Bridge::_library, "wglGetProcAddress"));
  }

  return new Bridge(static_cast<GL::Provider*>(graphics), log_context, log);
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

Bridge::Bridge(GL::Provider* provider, void* logctx, FG_Log log) : _logctx(logctx), _log(log), _provider(provider)
{
  emplaceContext = &EmplaceContext;
  attachContext  = &AttachContext;
  beginDraw      = &BeginDraw;
  endDraw        = &EndDraw;
  destroy        = &DestroyImpl;

  this->LOG(FG_Level_Notice, "Initializing fgOpenGL Desktop Bridge...");
  _singleton = this;
}

Bridge::~Bridge()
{
  if(_singleton == this)
    _singleton = nullptr;
}