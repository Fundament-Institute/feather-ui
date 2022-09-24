/* fgOpenGLDesktopBridge - OpenGL Desktop OS Bridge for Feather GUI
Copyright (c)2022 Fundament Software

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef FG__OPENGL_DESKTOP_BRIDGE_H
#define FG__OPENGL_DESKTOP_BRIDGE_H

namespace FG {
  class Bridge;
}

#include "feather/graphics_desktop_bridge.h"
#define AUTO_FRIEND FG::Bridge
#include "ProviderGL.hpp"
#include "ProviderGLFW.hpp"
#include <vector>

#ifdef FG_COMPILER_MSC
  #define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg, __VA_ARGS__)
#else
  #define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg __VA_OPT__(, ) __VA_ARGS__)
#endif

namespace FG {
  class Bridge : public FG_GraphicsDesktopBridge
  {
  public:
    Bridge(GL::Provider* provider, void* logctx, FG_Log log);
    ~Bridge();
    FG_Result Behavior(Bridge* data, const FG_Msg& msg);
    template<typename... Args> void Log(FG_Level level, const char* file, int line, const char* msg, Args&&... args)
    {
      if constexpr(sizeof...(Args) == 0)
        _log(_logctx, level, file, line, msg, nullptr, 0, &FreeImpl);
      else
        fgLogImpl<&FreeImpl>(_log, _logctx, level, file, line, msg, std::tuple<Args...>(std::forward<Args>(args)...),
                             std::index_sequence_for<Args...>{});
    }

    static void FreeImpl(char* p) { free(p); }

    static FG_Window* EmplaceContext(struct FG_GraphicsDesktopBridge* self, FG_Window* window,
                                     enum FG_PixelFormat backbuffer);
    static FG_Window* AttachContext(struct FG_GraphicsDesktopBridge* self, FG_Context* context, FG_Window* window);
    static int BeginDraw(struct FG_GraphicsDesktopBridge* self, FG_Window* window, FG_Rect* area);
    static int EndDraw(struct FG_GraphicsDesktopBridge* self, FG_Window* window);
    static int DestroyImpl(struct FG_GraphicsDesktopBridge* self);
    static void* ctxLoadProc(const char* name);
    static void* LoadProc(void* library, const char* name); 

    void* _logctx;

    static Bridge* _singleton;
    static void* _library;
    static int _refcount;

#ifdef FG_PLATFORM_WIN32
    static PROC(WINAPI* _wglGetProcAddress)(LPCSTR);
#else
    typedef void (*__GLXextproc)(void);
    static __GLXextproc(* _glxGetProcAddress)(const GLubyte *procName);
    static __GLXextproc(* _glxGetProcAddressARB)(const GLubyte *procName);
#endif

  protected:
    FG_Log _log;
    GL::Provider* _provider;
  };
}

#endif
