/* fgDirect2D - Direct2D Backend for Feather GUI
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

#ifndef FG__DIRECT2D_H
#define FG__DIRECT2D_H

#include "Window.h"
#include <vector>

struct IDWriteFactory1;
struct _DWM_BLURBEHIND;

typedef long(__stdcall* DWMCOMPENABLE)(int*);
typedef long(__stdcall* DWMBLURBEHIND)(struct HWND__*, const struct _DWM_BLURBEHIND*);
typedef int FG_Err;

namespace D2D {
  class Backend : public FG_Backend
  {
  public:
    Backend(void* root, FG_Log log, FG_Behavior behavior, ID2D1Factory1* factory, IWICImagingFactory* wicfactory,
            IDWriteFactory1* writefactory);
    ~Backend();
    void RefreshMonitors();
    Asset* LoadAsset(const char* data, size_t count);
    FG_Result Behavior(Window* data, const FG_Msg& msg);
    long CreateHWNDTarget(const D2D1_RENDER_TARGET_PROPERTIES& rtprop, const D2D1_HWND_RENDER_TARGET_PROPERTIES& hprop,
                          ID2D1HwndRenderTarget** target);
    uint16_t GetTouchIndex(unsigned long index, bool up);

    static FG_Err DrawTextD2D(FG_Backend* self, void* window, FG_Font* font, void* fontlayout, FG_Rect* area,
                              FG_Color color, float blur, float rotate, float z);
    static FG_Err DrawAsset(FG_Backend* self, void* window, FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color,
                            float time, float rotate, float z);
    static FG_Err DrawRect(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                           float border, FG_Color borderColor, float blur, FG_Asset* asset, float rotate, float z);
    static FG_Err DrawCircle(FG_Backend* self, void* window, FG_Rect* area, FG_Vec* angles, FG_Color fillColor,
                             float border, FG_Color borderColor, float blur, float innerRadius, float innerBorder,
                             FG_Asset* asset, float rotate);
    static FG_Err DrawTriangle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                               float border, FG_Color borderColor, float blur, FG_Asset* asset, float rotate, float z);
    static FG_Err DrawLines(FG_Backend* self, void* window, FG_Vec* points, uint32_t count, FG_Color color);
    static FG_Err DrawCurve(FG_Backend* self, void* window, FG_Vec* anchors, uint32_t count, FG_Color fillColor,
                            float stroke, FG_Color strokeColor);
    static FG_Err PushLayer(FG_Backend* self, void* window, FG_Rect* area, float* transform, float opacity, void* cache);
    static void* PopLayer(FG_Backend* self, void* window);
    static FG_Err DestroyLayer(FG_Backend* self, void* window, void* layer);
    static FG_Err PushClip(FG_Backend* self, void* window, FG_Rect* area);
    static FG_Err PopClip(FG_Backend* self, void* window);
    static FG_Err DirtyRect(FG_Backend* self, void* window, void* layer, FG_Rect* area);
    static FG_Font* CreateFontD2D(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                                  FG_Vec dpi, FG_AntiAliasing aa);
    static FG_Err DestroyFont(FG_Backend* self, FG_Font* font);
    static void* FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                            float letterSpacing, FG_BreakStyle breakStyle, void* prev);
    static FG_Err DestroyLayout(FG_Backend* self, void* layout);
    static uint32_t FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Vec pos, FG_Vec* cursor);
    static FG_Vec FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, uint32_t index);
    static FG_Asset* CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format);
    static FG_Err DestroyAsset(FG_Backend* self, FG_Asset* asset);
    static FG_Err PutClipboard(FG_Backend* self, void* window, FG_Clipboard kind, const char* data, uint32_t count);
    static uint32_t GetClipboard(FG_Backend* self, void* window, FG_Clipboard kind, void* target, uint32_t count);
    static bool CheckClipboard(FG_Backend* self, void* window, FG_Clipboard kind);
    static FG_Err ClearClipboard(FG_Backend* self, void* window, FG_Clipboard kind);
    static FG_Err ProcessMessages(FG_Backend* self);
    static FG_Err SetCursorD2D(FG_Backend* self, void* window, FG_Cursor cursor);
    static FG_Err RequestAnimationFrame(FG_Backend* self, void* window, unsigned long long microdelay);
    static FG_Err GetDisplayIndex(FG_Backend* self, unsigned int index, FG_Display* out);
    static FG_Err GetDisplay(FG_Backend* self, void* handle, FG_Display* out);
    static FG_Err GetDisplayWindow(FG_Backend* self, void* window, FG_Display* out);
    static void* CreateWindowD2D(FG_Backend* self, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                                 const char* caption, uint64_t flags, void* context);
    static FG_Err SetWindowD2D(FG_Backend* self, void* window, FG_Element* element, void* display, FG_Vec* pos, FG_Vec* dim,
                               const char* caption, uint64_t flags);
    static FG_Err DestroyWindow(FG_Backend* self, void* window);
    static FG_Err BeginDraw(FG_Backend* self, void* window, FG_Rect* area, bool clear);
    static FG_Err EndDraw(FG_Backend* self, void* window);
    static void PushRotate(D2D::Window* context, float rotate, const FG_Rect& area);
    static void PopRotate(D2D::Window* context, float rotate);
    DWMBLURBEHIND dwmblurbehind;
    FG_Log _log;
    void* _root;
    CompactArray<uint64_t, 2, uint16_t> _touchid;

    static constexpr wchar_t WindowClass[] = L"WindowD2D";

  protected:
    static int __stdcall EnumerateMonitorsProc(struct HMONITOR__* monitor, struct HDC__* hdc, struct tagRECT*,
                                               longptr_t lparam);
    template<int N, typename Arg, typename... Args>
    static inline FG_Err DrawEffect(Window* ctx, ID2D1Effect* effect, const FG_Rect& area, float rotate,
                                    const Arg arg, const Args&... args);
    static Window* FromHWND(void* p);

    ID2D1Factory1* _factory;
    IWICImagingFactory* _wicfactory;
    IDWriteFactory1* _writefactory;
    CompactArray<FG_Display> _displays;
    FG_Behavior _behavior;
    long(__stdcall* getDpiForMonitor)(struct HMONITOR__*, int, unsigned int*, unsigned int*);
    long(__stdcall* getScaleFactorForMonitor)(struct HMONITOR__*, int*);

    static const unsigned char VALID_DISPLAY = 0x40;
  };
}

#endif
