/* fgOpenGL - OpenGL Backend for Feather GUI
Copyright (c)2020 Fundament Software

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

typedef int FG_Err;

namespace GL {
  struct Asset : FG_Asset
  {
    uint32_t count;
  };

  KHASH_DECLARE(assets, const FG_Asset*, char);

  class Backend : public FG_Backend
  {
  public:
    Backend(void* root, FG_Log log, FG_Behavior behavior);
    ~Backend();
    bool AssetExists(const FG_Asset*);
    FG_Result Behavior(Window* data, const FG_Msg& msg);

    static FG_Err DrawTextGL(FG_Backend* self, void* window, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color color,
                             float lineHeight, float letterSpacing, float blur, FG_AntiAliasing aa);
    static FG_Err DrawAsset(FG_Backend* self, void* window, FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color,
                            float time);
    static FG_Err DrawRect(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                           float border, FG_Color borderColor, float blur, FG_Asset* asset);
    static FG_Err DrawCircle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* arcs, FG_Color fillColor, float border,
                             FG_Color borderColor, float blur, FG_Asset* asset);
    static FG_Err DrawTriangle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                               float border, FG_Color borderColor, float blur, FG_Asset* asset);
    static FG_Err DrawLines(FG_Backend* self, void* window, FG_Vec* points, uint32_t count, FG_Color color);
    static FG_Err DrawCurve(FG_Backend* self, void* window, FG_Vec* anchors, uint32_t count, FG_Color fillColor,
                            float stroke, FG_Color strokeColor);
    static FG_Err PushLayer(FG_Backend* self, void* window, FG_Rect* area, float* transform, float opacity);
    static FG_Err PopLayer(FG_Backend* self, void* window);
    static FG_Err PushClip(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Err PopClip(FG_Backend* self, void* window);
    static FG_Err DirtyRect(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Font* CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                                 FG_Vec dpi);
    static FG_Err DestroyFont(FG_Backend* self, FG_Font* font);
    static void* FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                            float letterSpacing, void* prev, FG_Vec dpi);
    static FG_Err DestroyLayout(FG_Backend* self, void* layout);
    static uint32_t FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, float lineHeight,
                              float letterSpacing, FG_Vec pos, FG_Vec* cursor, FG_Vec dpi);
    static FG_Vec FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, float lineHeight,
                          float letterSpacing, uint32_t index, FG_Vec dpi);
    static FG_Asset* CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format);
    static FG_Err DestroyAsset(FG_Backend* self, FG_Asset* asset);
    static FG_Err PutClipboard(FG_Backend* self, FG_Clipboard kind, const char* data, uint32_t count);
    static uint32_t GetClipboard(FG_Backend* self, FG_Clipboard kind, void* target, uint32_t count);
    static bool CheckClipboard(FG_Backend* self, FG_Clipboard kind);
    static FG_Err ClearClipboard(FG_Backend* self, FG_Clipboard kind);
    static FG_Err ProcessMessages(FG_Backend* self);
    static FG_Err SetCursorGL(FG_Backend* self, void* window, FG_Cursor cursor);
    static FG_Err RequestAnimationFrame(FG_Backend* self, void* window, unsigned long long microdelay);
    static FG_Err GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out);
    static FG_Err GetDisplay(FG_Backend* self, void* handle, FG_Display* out);
    static FG_Err GetDisplayWindow(FG_Backend* self, void* window, FG_Display* out);
    static void* CreateWindowGL(FG_Backend* self, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                                const char* caption, uint64_t flags);
    static FG_Err SetWindowGL(FG_Backend* self, void* window, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                              const char* caption, uint64_t flags);
    static FG_Err DestroyWindow(FG_Backend* self, void* window);
    static FG_Err BeginDraw(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Err EndDraw(FG_Backend* self, void* window);
    static void ErrorCallback(int error, const char* description);

    FG_Log _log;
    void* _root;

    static int _lasterr;
    static int _refcount;
    static char _lasterrdesc[1024]; // 1024 is from GLFW internals

  protected:
    FG_Behavior _behavior;
    kh_assets_t* _assethash;
  };
}

#endif
