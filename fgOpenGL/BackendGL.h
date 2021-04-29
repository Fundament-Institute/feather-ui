/* fgOpenGL - OpenGL Backend for Feather GUI
Copyright (c)2021 Fundament Software

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

#ifndef FG__OPENGL_H
#define FG__OPENGL_H

#include "Window.h"
#include <vector>

struct FT_LibraryRec_;
struct IDWriteFactory1;

namespace GL {
  KHASH_DECLARE(assets, const FG_Asset*, char);

  class Backend : public FG_Backend
  {
  public:
    Backend(void* root, FG_Log log, FG_Behavior behavior);
    ~Backend();
    FG_Result Behavior(Context* data, const FG_Msg& msg);
    bool LogError(const char* call);

    static FG_Err DrawGL(FG_Backend* self, FG_Window* window, FG_Command* commandlist, unsigned int n_commands,
                         FG_BlendState* blend);
    static bool Clear(FG_Backend* self, FG_Window* window, FG_Color color);
    static FG_Err PushLayer(FG_Backend* self, FG_Window* window, FG_Asset* layer, float* transform, float opacity,
                            FG_BlendState* blend);
    static FG_Err PopLayer(FG_Backend* self, FG_Window* window);
    static FG_Err SetRenderTarget(FG_Backend* self, FG_Window* window, FG_Asset* target);
    static FG_Err PushClip(FG_Backend* self, FG_Window* window, FG_Rect* area);
    static FG_Err PopClip(FG_Backend* self, FG_Window* window);
    static FG_Err DirtyRect(FG_Backend* self, FG_Window* window, FG_Rect* area);
    static FG_Shader* CreateShader(FG_Backend* self, const char* ps, const char* vs, const char* gs, const char* cs,
                                   const char* ds, const char* hs, FG_ShaderParameter* parameters, uint32_t n_parameters);
    static FG_Err DestroyShader(FG_Backend* self, FG_Shader* shader);
    static FG_Err GetProjection(FG_Backend* self, FG_Window* window, FG_Asset* layer, float* proj4x4);
    static FG_Font* CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                                 FG_Vec dpi, FG_AntiAliasing aa);
    static FG_Err DestroyFont(FG_Backend* self, FG_Font* font);
    static void* FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                            float letterSpacing, FG_BreakStyle breakStyle, void* prev);
    static FG_Err DestroyLayout(FG_Backend* self, void* layout);
    static uint32_t FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Vec pos, FG_Vec* cursor);
    static FG_Vec FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, uint32_t index);
    static FG_Asset* CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format, int flags);
    static FG_Asset* CreateBuffer(FG_Backend* self, void* data, uint32_t bytes, uint8_t primitive,
                                  FG_ShaderParameter* parameters, uint32_t n_parameters);
    static FG_Asset* CreateLayer(FG_Backend* self, FG_Window* window, FG_Vec* size, int flags);
    static FG_Err DestroyAsset(FG_Backend* self, FG_Asset* asset);
    static FG_Err PutClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, const char* data, uint32_t count);
    static uint32_t GetClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, void* target, uint32_t count);
    static bool CheckClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind);
    static FG_Err ClearClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind);
    static FG_Err ProcessMessages(FG_Backend* self);
    static FG_Err SetCursorGL(FG_Backend* self, FG_Window* window, FG_Cursor cursor);
    static FG_Err GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out);
    static FG_Err GetDisplay(FG_Backend* self, void* handle, FG_Display* out);
    static FG_Err GetDisplayWindow(FG_Backend* self, FG_Window* window, FG_Display* out);
    static FG_Window* CreateRegionGL(FG_Backend* self, FG_MsgReceiver* element, FG_Window desc, FG_Vec3 pos, FG_Vec3 dim);
    static FG_Window* CreateWindowGL(FG_Backend* self, FG_MsgReceiver* element, void* display, FG_Vec* pos, FG_Vec* dim,
                                     const char* caption, uint64_t flags);
    static FG_Err SetWindowGL(FG_Backend* self, FG_Window* window, FG_MsgReceiver* element, void* display, FG_Vec* pos,
                              FG_Vec* dim, const char* caption, uint64_t flags);
    static FG_Err DestroyWindow(FG_Backend* self, FG_Window* window);
    static FG_Err BeginDraw(FG_Backend* self, FG_Window* window, FG_Rect* area);
    static FG_Err EndDraw(FG_Backend* self, FG_Window* window);
    static void* CreateSystemControl(FG_Backend* self, FG_Window* window, const char* id, FG_Rect* area, ...);
    static FG_Err SetSystemControl(FG_Backend* self, FG_Window* window, void* control, FG_Rect* area, ...);
    static FG_Err DestroySystemControl(FG_Backend* self, FG_Window* window, void* control);
    static void ErrorCallback(int error, const char* description);
    static void JoystickCallback(int id, int connected);

    FG_Log _log;
    void* _root;
    Window* _windows;
    Shader _imageshader; // used for text
    Shader _rectshader;
    Shader _circleshader;
    Shader _arcshader;
    Shader _trishader;
    Shader _lineshader;
    struct FT_LibraryRec_* _ftlib;

    static int _lasterr;
    static int _refcount;
    static char _lasterrdesc[1024]; // 1024 is from GLFW internals
    static int _maxjoy;
    static const float BASE_DPI;
    static Backend* _singleton;
    static const float PI;
    static void* _library;

    #ifdef FG_PLATFORM_WIN32
    IDWriteFactory1* _writefactory = 0;
    #endif

  protected:
    FG_Behavior _behavior;
    kh_assets_t* _assethash;
  };
}

#endif
