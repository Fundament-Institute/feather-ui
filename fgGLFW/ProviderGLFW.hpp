/* fgOpenGL - OpenGL Provider for Feather GUI
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

#ifndef FG__GLFW_H
#define FG__GLFW_H

#include "Window.hpp"
#include <vector>
#include <memory>
#include <string>

#define LOG(level, msg, ...) Log(level, __FILE__, __LINE__, msg __VA_OPT__(, ) __VA_ARGS__)

namespace GLFW {
  enum GLFW_Err
  {
    ERR_SUCCESS = 0,
    ERR_UNKNOWN = 0x10000,
    ERR_NOT_IMPLEMENTED,
    ERR_MISSING_PARAMETER,
    ERR_INVALID_KIND,
    ERR_CLIPBOARD_FAILURE,
    ERR_INVALID_CURSOR,
    ERR_INVALID_DISPLAY,
    ERR_INVALID_PARAMETER,
    ERR_INVALID_CALL,
    ERR_NULL,
  };

  class Provider : public FG_DesktopInterface
  {
  public:
    Provider(void* root, FG_Log log, FG_Behavior behavior);
    ~Provider();
    FG_Result Behavior(WindowGL* data, const FG_Msg& msg);
    template<typename... Args> void Log(FG_Level level, const char* file, int line, const char* msg, Args&&... args)
    {
      if constexpr(sizeof...(Args) == 0)
        _log(_logctx, level, file, line, msg, nullptr, 0, &FreeImpl);
      else
        fgLogImpl<&FreeImpl>(_log, _logctx, level, file, line, msg, std::tuple<Args...>(std::forward<Args>(args)...),
                             std::index_sequence_for<Args...>{});
    }

    static void FreeImpl(char* p) { free(p); }
    static FG_Window* CreateWindowImpl(FG_DesktopInterface* self, uintptr_t window_id, FG_Display* display, FG_Vec2* pos,
                                       FG_Vec2* dim, const char* caption, uint64_t flags);
    static int SetWindow(FG_DesktopInterface* self, FG_Window* window, FG_Display* display, FG_Vec2* pos, FG_Vec2* dim,
                         const char* caption, uint64_t flags);
    static int DestroyWindow(FG_DesktopInterface* self, FG_Window* window);
    static int InvalidateWindow(FG_DesktopInterface* self, FG_Window* window, FG_Rect* optarea);
    static int PutClipboard(FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind, const char* data,
                            uint32_t count);
    static uint32_t GetClipboard(FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind, void* target,
                                 uint32_t count);
    static bool CheckClipboard(FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind);
    static int ClearClipboard(FG_DesktopInterface* self, FG_Window* window, enum FG_Clipboard kind);
    static int ProcessMessages(FG_DesktopInterface* self, FG_Window* window, void* ui_state);
    static int GetMessageSyncObject(FG_DesktopInterface* self, FG_Window* window);
    static int SetCursorImpl(FG_DesktopInterface* self, FG_Window* window, enum FG_Cursor cursor);
    static int GetDisplayIndex(FG_DesktopInterface* self, unsigned int index, FG_Display* out);
    static int GetDisplay(FG_DesktopInterface* self, uintptr_t handle, FG_Display* out);
    static int GetDisplayWindow(FG_DesktopInterface* self, FG_Window* window, FG_Display* out);
    static int CreateSystemControl(FG_DesktopInterface* self, FG_Window* context, const char* id, FG_Rect* area, ...);
    static int SetSystemControl(FG_DesktopInterface* self, FG_Window* context, void* control, FG_Rect* area, ...);
    static int DestroySystemControl(FG_DesktopInterface* self, FG_Window* context, void* control);
    static int DestroyImpl(FG_DesktopInterface* self);

    static void ErrorCallback(int error, const char* description);
    static void JoystickCallback(int id, int connected);

    void* _uictx;
    void* _logctx;
    WindowGL* _windows;

    static int _lasterr;
    static int _refcount;
    static char _lasterrdesc[1024]; // 1024 is from GLFW internals
    static int _maxjoy;
    static const float BASE_DPI;
    static Provider* _singleton;

  protected:
    FG_Log _log;
    FG_Behavior _behavior;
  };
}

#endif
