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
struct FT_LibraryRec_;

namespace GL {
  KHASH_DECLARE(assets, const FG_Asset*, char);

  class Backend : public FG_Backend
  {
  public:
    Backend(void* root, FG_Log log, FG_Behavior behavior);
    ~Backend();
    FG_Result Behavior(Context* data, const FG_Msg& msg);
    bool LogError(const char* call);
    FG_Err DrawTextureQuad(Context* context, GLuint tex, ImageVertex* v, FG_Color color, float* transform,
                           FG_BlendState* blend);

    static FG_Err DrawTextGL(FG_Backend* self, void* window, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color color,
                             float blur, float rotate, float z, FG_BlendState* blend);
    static FG_Err DrawAsset(FG_Backend* self, void* window, FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color,
                            float time, float rotate, float z, FG_BlendState* blend);
    static FG_Err DrawRect(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                           float border, FG_Color borderColor, float blur, FG_Asset* asset, float rotate, float z,
                           FG_BlendState* blend);
    static FG_Err DrawCircle(FG_Backend* self, void* window, FG_Rect* area, FG_Vec* angles, FG_Color fillColor,
                             float border, FG_Color borderColor, float blur, float innerRadius, float innerBorder,
                             FG_Asset* asset, float rotate, FG_BlendState* blend);
    static FG_Err DrawTriangle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                               float border, FG_Color borderColor, float blur, FG_Asset* asset, float rotate, float z,
                               FG_BlendState* blend);
    static FG_Err DrawLines(FG_Backend* self, void* window, FG_Vec* points, uint32_t count, FG_Color color,
                            FG_BlendState* blend);
    static FG_Err DrawCurve(FG_Backend* self, void* window, FG_Vec* anchors, uint32_t count, FG_Color fillColor,
                            float stroke, FG_Color strokeColor, FG_BlendState* blend);
    static FG_Err DrawShader(FG_Backend* self, void* window, FG_Shader* shader, FG_Asset* vertices, FG_Asset* indices,
                             FG_BlendState* blend, ...);
    static bool Clear(FG_Backend* self, void* window, FG_Color color);
    static FG_Err PushLayer(FG_Backend* self, void* window, FG_Asset* layer, float* transform, float opacity,
                            FG_BlendState* blend);
    static FG_Err PopLayer(FG_Backend* self, void* window);
    static FG_Err SetRenderTarget(FG_Backend* self, void* window, FG_Asset* target);
    static FG_Err PushClip(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Err PopClip(FG_Backend* self, void* window);
    static FG_Err DirtyRect(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Shader* CreateShader(FG_Backend* self, const char* ps, const char* vs, const char* gs, const char* cs,
                                   const char* ds, const char* hs, FG_ShaderParameter* parameters, uint32_t n_parameters);
    static FG_Err DestroyShader(FG_Backend* self, FG_Shader* shader);
    static FG_Err GetProjection(FG_Backend* self, void* window, void* layer, float* proj4x4);
    static FG_Font* CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                                 FG_Vec dpi, FG_AntiAliasing aa);
    static FG_Err DestroyFont(FG_Backend* self, FG_Font* font);
    static void* FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                            float letterSpacing, FG_BreakStyle breakStyle, void* prev);
    static FG_Err DestroyLayout(FG_Backend* self, void* layout);
    static uint32_t FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Vec pos, FG_Vec* cursor);
    static FG_Vec FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, uint32_t index);
    static FG_Asset* CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format);
    static FG_Asset* CreateBuffer(FG_Backend* self, void* data, uint32_t bytes, uint8_t primitive,
                                  FG_ShaderParameter* parameters, uint32_t n_parameters);
    static FG_Asset* CreateLayer(FG_Backend* self, void* window, FG_Vec* size, bool cache);
    static FG_Err DestroyAsset(FG_Backend* self, FG_Asset* asset);
    static FG_Err PutClipboard(FG_Backend* self, void* window, FG_Clipboard kind, const char* data, uint32_t count);
    static uint32_t GetClipboard(FG_Backend* self, void* window, FG_Clipboard kind, void* target, uint32_t count);
    static bool CheckClipboard(FG_Backend* self, void* window, FG_Clipboard kind);
    static FG_Err ClearClipboard(FG_Backend* self, void* window, FG_Clipboard kind);
    static FG_Err ProcessMessages(FG_Backend* self);
    static FG_Err SetCursorGL(FG_Backend* self, void* window, FG_Cursor cursor);
    static FG_Err GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out);
    static FG_Err GetDisplay(FG_Backend* self, void* handle, FG_Display* out);
    static FG_Err GetDisplayWindow(FG_Backend* self, void* window, FG_Display* out);
    static void* CreateWindowGL(FG_Backend* self, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                                const char* caption, uint64_t flags, void* context);
    static FG_Err SetWindowGL(FG_Backend* self, void* window, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                              const char* caption, uint64_t flags);
    static FG_Err DestroyWindow(FG_Backend* self, void* window);
    static FG_Err BeginDraw(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Err EndDraw(FG_Backend* self, void* window);
    static void* CreateSystemControl(FG_Backend* self, void* window, const char* id, FG_Rect* area, ...);
    static FG_Err SetSystemControl(FG_Backend* self, void* window, void* control, FG_Rect* area, ...);
    static FG_Err DestroySystemControl(FG_Backend* self, void* window, void* control);
    static void ErrorCallback(int error, const char* description);
    static void JoystickCallback(int id, int connected);
    static void ColorFloats(const FG_Color& c, float (&colors)[4]);
    static void GenTransform(float (&target)[4][4], const FG_Rect& area, float rotate, float z);

    FG_Log _log;
    void* _root;
    Window* _windows;
    Shader _imageshader; // used for text
    Shader _rectshader;
    Shader _circleshader;
    Shader _trishader;
    Shader _lineshader;
    struct FT_LibraryRec_* _ftlib;

    static int _lasterr;
    static int _refcount;
    static char _lasterrdesc[1024]; // 1024 is from GLFW internals
    static int _maxjoy;
    static const float BASE_DPI;
    static Backend* _singleton;

    template<class T> inline static void _buildQuad(T (&v)[4], const FG_Rect& area)
    {
      // Due to counter-clockwise windings, we have to go topleft, bottomleft, topright, which also requires flipping the UV
      // x/y coordinates
      v[0].posUV[0] = area.left;
      v[0].posUV[1] = area.top;
      v[0].posUV[2] = 0;
      v[0].posUV[3] = 0;

      v[1].posUV[0] = area.left;
      v[1].posUV[1] = area.bottom;
      v[1].posUV[2] = 1.0f;
      v[1].posUV[3] = 0;

      v[2].posUV[0] = area.right;
      v[2].posUV[1] = area.top;
      v[2].posUV[2] = 0;
      v[2].posUV[3] = 1.0f;

      v[3].posUV[0] = area.right;
      v[3].posUV[1] = area.bottom;
      v[3].posUV[2] = 1.0f;
      v[3].posUV[3] = 1.0f;
    }

  protected:
    template<class T> inline static void _buildPosUV(T (&v)[4], const FG_Rect& area, const FG_Rect& uv, float x, float y)
    {
      // Due to counter-clockwise windings, we have to go topleft, bottomleft, topright, which also requires flipping the UV
      // x/y coordinates
      v[0].posUV[0] = area.left;
      v[0].posUV[1] = area.top;
      v[0].posUV[2] = uv.top / y;
      v[0].posUV[3] = uv.left / x;

      v[1].posUV[0] = area.left;
      v[1].posUV[1] = area.bottom;
      v[1].posUV[2] = uv.bottom / y;
      v[1].posUV[3] = uv.left / x;

      v[2].posUV[0] = area.right;
      v[2].posUV[1] = area.top;
      v[2].posUV[2] = uv.top / y;
      v[2].posUV[3] = uv.right / x;

      v[3].posUV[0] = area.right;
      v[3].posUV[1] = area.bottom;
      v[3].posUV[2] = uv.bottom / y;
      v[3].posUV[3] = uv.right / x;
    }
    void _drawStandard(GLuint shader, GLuint vao, float (&proj)[4][4], const FG_Rect& area, const FG_Rect& corners,
                       FG_Color fillColor, float border, FG_Color borderColor, float blur, float rotate, float z);
    static void _flushbatchdraw(Backend* backend, Context* context, Font* font);

    FG_Behavior _behavior;
    kh_assets_t* _assethash;
  };
}

#endif
