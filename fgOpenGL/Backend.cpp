// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "platform.h"
#include "Backend.h"
#include "Font.h"
#include "linmath.h"
#include "utf.h"
#include <filesystem>
#include "freetype/freetype.h"

using namespace GL;
namespace fs = std::filesystem;

namespace GL {
  __KHASH_IMPL(assets, , const FG_Asset*, char, 0, kh_ptr_hash_func, kh_int_hash_equal);
}

int Backend::_lasterr            = 0;
int Backend::_refcount           = 0;
char Backend::_lasterrdesc[1024] = {};
int Backend::_maxjoy             = 0;
const float Backend::BASE_DPI    = 96.0f;

void Backend::ColorFloats(const FG_Color& c, float (&colors)[4])
{
  colors[0] = c.r / 255.0f;
  colors[1] = c.g / 255.0f;
  colors[2] = c.b / 255.0f;
  colors[3] = c.a / 255.0f;
}

FG_Err Backend::DrawTextGL(FG_Backend* self, void* window, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Color color,
                           float lineHeight, float letterSpacing, float blur)
{
  return -1;
}

FG_Err Backend::DrawAsset(FG_Backend* self, void* window, FG_Asset* asset, FG_Rect* area, FG_Rect* source, FG_Color color,
                          float time)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(window);
  GLuint tex   = context->LoadAsset(static_cast<Asset*>(asset));

  FG_Rect full = { 0, 0, (float)asset->size.x, (float)asset->size.y };
  if(!source)
    source = &full;
  ImageVertex v[4];
  _buildPosUV(v, *area, *source, (float)asset->size.x, (float)asset->size.y);

  for(int i = 0; i < 4; ++i)
    ColorFloats(color, v[i].color);

  Attribute MVP("MVP", GL_FLOAT_MAT4, (float*)context->proj);

  glUseProgram(context->_imageshader);
  backend->LogError("glUseProgram");
  glBindVertexArray(context->_imageobject);
  backend->LogError("glBindVertexArray");

  glEnable(GL_TEXTURE_2D);
  backend->LogError("glEnable");
  glActiveTexture(GL_TEXTURE0);
  backend->LogError("glActiveTexture");
  glBindTexture(GL_TEXTURE_2D, tex);
  backend->LogError("glBindTexture");

  glBindBuffer(GL_ARRAY_BUFFER, context->_imagebuffer);
  backend->LogError("glBindBuffer");
  context->AppendBatch(v, sizeof(v), 4);
  Shader::SetUniform(backend, context->_imageshader, MVP);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, context->FlushBatch());
  backend->LogError("glDrawArrays");
  glBindVertexArray(0);
  backend->LogError("glBindVertexArray");

  glDisable(GL_TEXTURE_2D);
  backend->LogError("glDisable");
  return 0;
}

void Backend::GenTransform(float (&target)[4][4], const FG_Rect& area)
{
  // mat4x4_translate(v, area->left, area->top, 0);
  // mat4x4_scale_aniso(mv, v, area->right - area->left, area->bottom - area->top, 1);
  memset(target, 0, sizeof(float) * 4 * 4);
  target[0][0] = area.right - area.left;
  target[1][1] = area.bottom - area.top;
  target[2][2] = 1.0f;
  target[3][3] = 1.0f;
  target[3][0] = area.left;
  target[3][1] = area.top;
}

void Backend::_drawStandard(GLuint shader, GLuint vao, float (&proj)[4][4], const FG_Rect& area, const FG_Rect& corners,
                            FG_Color fillColor, float border, FG_Color borderColor, float blur)
{
  mat4x4 mvp;
  GenTransform(mvp, area);
  mat4x4_mul(mvp, proj, mvp);
  Attribute MVP("MVP", GL_FLOAT_MAT4, (float*)mvp);
  Attribute DimBorderBlur("DimBorderBlur", GL_FLOAT_VEC4);
  DimBorderBlur.data[0] = area.right - area.left;
  DimBorderBlur.data[1] = area.bottom - area.top;
  DimBorderBlur.data[2] = border;
  DimBorderBlur.data[3] = blur;

  Attribute Corners("Corners", GL_FLOAT_VEC4, (float*)corners.ltrb);
  Attribute Fill("Fill", GL_FLOAT_VEC4);
  ColorFloats(fillColor, Fill.data);
  Attribute Outline("Outline", GL_FLOAT_VEC4);
  ColorFloats(borderColor, Outline.data);

  _drawVAO(shader, vao, GL_TRIANGLE_STRIP, 4, MVP, DimBorderBlur, Corners, Fill, Outline);
}

FG_Err Backend::DrawRect(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor, float border,
                         FG_Color borderColor, float blur, FG_Asset* asset)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(window);
  backend->_drawStandard(context->_rectshader, context->_quadobject, context->proj, *area, *corners, fillColor, border,
                         borderColor, blur);
  return 0;
}

FG_Err Backend::DrawCircle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* arcs, FG_Color fillColor, float border,
                           FG_Color borderColor, float blur, FG_Asset* asset)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(window);
  backend->_drawStandard(context->_circleshader, context->_quadobject, context->proj, *area, *arcs, fillColor, border,
                         borderColor, blur);
  return 0;
}
FG_Err Backend::DrawTriangle(FG_Backend* self, void* window, FG_Rect* area, FG_Rect* corners, FG_Color fillColor,
                             float border, FG_Color borderColor, float blur, FG_Asset* asset)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(window);
  backend->_drawStandard(context->_trishader, context->_quadobject, context->proj, *area, *corners, fillColor, border,
                         borderColor, blur);
  return 0;
}

FG_Err Backend::DrawLines(FG_Backend* self, void* window, FG_Vec* points, uint32_t count, FG_Color color)
{
  auto backend = static_cast<Backend*>(self);
  auto context = reinterpret_cast<Context*>(window);

  /*struct LineVertex
  {
    float x, y;
    float a, r, g, b;
  } vertex;

  context->AppendBatch(&vertex, sizeof(LineVertex), 0);
  for(uint32_t i = 1; i < count; ++i)
    context->AppendBatch(&vertex, sizeof(LineVertex), 1);
  context->FlushBatch(backend->_lineshader, context->_lineshader, GL_LINE_STRIP, 0, 0);*/

  return glGetError();
}

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

FG_Err Backend::DirtyRect(FG_Backend* self, void* window, void* layer, FG_Rect* area)
{
  if(!self || !window)
    return -1;
  if(layer)
    reinterpret_cast<Layer*>(layer)->Dirty(area);
  return 0; // TODO: Handle dirty backbuffers
}

FG_Font* Backend::CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                               FG_Vec dpi, FG_AntiAliasing aa)
{
  return new Font(static_cast<Backend*>(self), family, pt, aa, dpi);
}

FG_Err Backend::DestroyFont(FG_Backend* self, FG_Font* font)
{
  if(!self || !font)
    return -1;
  delete static_cast<Font*>(font);
  return 0;
}
FG_Err Backend::DestroyLayout(FG_Backend* self, void* layout)
{
  if(!self || !layout)
    return -1;
  free(reinterpret_cast<TextLayout*>(layout)->text);
  free(layout);
  return 0;
}

void* Backend::FontLayout(FG_Backend* self, FG_Font* font, const char* text, FG_Rect* area, float lineHeight,
                          float letterSpacing, FG_BreakStyle breakStyle, void* prev)
{
  DestroyLayout(self, prev); // Figuring out if prev can be reused ends up being too costly to bother with

  // Simple layout to be replaced by Harfbuzz eventually
  std::vector<const char32_t*> breaks;
  Font* f        = static_cast<Font*>(font);
  float maxwidth = area->right - area->left;
  size_t len     = strlen(text) + 1;
  char32_t* utf  = (char32_t*)malloc(len * sizeof(char32_t)); // overallocate for UTF32
  UTF8toUTF32(text, len, utf, len);
  const char32_t* cur = utf;

  do
  {
    breaks.push_back(cur);
    f->GetLineWidth(cur, maxwidth, breakStyle, letterSpacing);
  } while(*cur);

  auto layout           = reinterpret_cast<TextLayout*>(malloc(sizeof(TextLayout) + (breaks.size() * sizeof(char32_t*))));
  layout->text          = utf;
  layout->lines         = reinterpret_cast<const char32_t**>(layout + 1);
  layout->n_lines       = breaks.size();
  layout->area          = *area;
  layout->lineheight    = lineHeight;
  layout->letterspacing = letterSpacing;
  layout->breakstyle    = breakStyle;
  for(size_t i = 0; i < breaks.size(); ++i)
    layout->lines[i] = breaks[i];

  return layout;
}
uint32_t Backend::FontIndex(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, FG_Vec pos, FG_Vec* cursor)
{
  if(!self || !font || !fontlayout || !area)
    return ~0U;
  auto layout = reinterpret_cast<TextLayout*>(fontlayout);
  Font* f     = static_cast<Font*>(font);
  auto r      = f->GetIndex(layout->text, area->right - area->left, layout->breakstyle, layout->lineheight, layout->letterspacing, pos);
  cursor->x   = r.second.x;
  cursor->y   = r.second.y;
  return r.first;
}

FG_Vec Backend::FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, uint32_t index)
{
  if(!self || !font || !fontlayout || !area)
    return { NAN, NAN };
  auto layout = reinterpret_cast<TextLayout*>(fontlayout);
  Font* f     = static_cast<Font*>(font);
  auto r      = f->GetPos(layout->text, area->right - area->left, layout->breakstyle, layout->lineheight, layout->letterspacing, index);
  FG_Vec c    = { r.second.x, r.second.y };
  return c;
}

FG_Asset* Backend::CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format)
{
  auto backend     = static_cast<Backend*>(self);
  uint32_t len     = !count ? strlen(data) + 1 : count;
  Asset* asset     = reinterpret_cast<Asset*>(malloc(len + sizeof(Asset)));
  asset->data.data = asset + 1;
  MEMCPY(asset->data.data, len, data, len);
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
  out->dpi.x *= BASE_DPI;
  out->dpi.y *= BASE_DPI;
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
  {
    if(dim)
      context->SetDim(*dim);
    return 0;
  }
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
#ifdef FG_PLATFORM_POSIX
    FcFini();
#endif
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
#ifdef FG_PLATFORM_POSIX
    FcInit();
#endif

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

#define _STRINGIFY(x) x
#define TXT(x)        _STRINGIFY(#x)

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
  FT_Init_FreeType(&_ftlib);

  const char* image_vs =
#include "Image.vs.glsl"
    ;

  const char* image_fs =
#include "Image.fs.glsl"
    ;

  _imageshader = Shader(image_fs, image_vs, 0,
                        { Shader::Data{ Shader::VERTEX, GL_FLOAT_VEC4, "vPosUV", 0 },
                          Shader::Data{ Shader::VERTEX, GL_FLOAT_VEC4, "vColor", 4 * sizeof(float) },
                          Shader::Data{ Shader::GLOBAL, GL_FLOAT_MAT4, "MVP" } });

  const char* standard_vs =
#include "standard.vs.glsl"
    ;

  const char* roundrect_fs =
#include "RoundRect.fs.glsl"
    ;

  const char* triangle_fs =
#include "Triangle.fs.glsl"
    ;

  const char* circle_fs =
#include "Circle.fs.glsl"
    ;

  std::initializer_list<Shader::Data> standardLayout = {
    Shader::Data{ Shader::VERTEX, GL_FLOAT_VEC2, "vPos", 0 },
    Shader::Data{ Shader::GLOBAL, GL_FLOAT_MAT4, "MVP" },
    Shader::Data{ Shader::GLOBAL, GL_FLOAT_VEC4, "DimBorderBlur" },
    Shader::Data{ Shader::GLOBAL, GL_FLOAT_VEC4, "Corners" },
    Shader::Data{ Shader::GLOBAL, GL_FLOAT_VEC4, "Fill" },
    Shader::Data{ Shader::GLOBAL, GL_FLOAT_VEC4, "Outline" },
  };

  _rectshader   = Shader(roundrect_fs, standard_vs, 0, standardLayout);
  _trishader    = Shader(triangle_fs, standard_vs, 0, standardLayout);
  _circleshader = Shader(circle_fs, standard_vs, 0, standardLayout);

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

  FT_Done_FreeType(_ftlib);
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
