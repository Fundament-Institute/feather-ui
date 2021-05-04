// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "Backend.h"

#include "platform.h"
#include "BackendGL.h"
#include "Font.h"
#include "linmath.h"
#include "utf.h"
#include <filesystem>
#include <float.h>
#include "SOIL.h"
#include "ft2build.h"
#include FT_FREETYPE_H
#include "freetype/freetype.h"

#ifdef FG_PLATFORM_WIN32
  #include <dwrite_1.h>
#else
  #include <dlfcn.h>
#endif

using namespace GL;
namespace fs = std::filesystem;

namespace GL {
  __KHASH_IMPL(assets, , const FG_Asset*, char, 0, kh_ptr_hash_func, kh_int_hash_equal);
}

int Backend::_lasterr            = 0;
int Backend::_refcount           = 0;
char Backend::_lasterrdesc[1024] = {};
int Backend::_maxjoy             = 0;
Backend* Backend::_singleton     = nullptr;
const float Backend::BASE_DPI    = 96.0f;
const float Backend::PI          = 3.14159265359f;
void* Backend::_library          = 0;

FG_Err Backend::DrawGL(FG_Backend* self, FG_Window* window, FG_Command* commandlist, unsigned int n_commands,
                       FG_BlendState* blend)
{
  if(!self || !window)
    return -1;

  auto backend = static_cast<Backend*>(self);
  auto context = static_cast<Context*>(window);

  auto dflags    = context->ApplyBlend(blend).flags;
  bool linearize = !(dflags & FG_DrawFlags_LINEAR);
  for(unsigned int i = 0; i < n_commands; ++i)
  {
    auto& c = commandlist[i];
    switch(c.category)
    {
    case FG_Category_ARC:
      context->DrawArc(*c.shape.area, c.shape.arc.angles, c.shape.fillColor, c.shape.border, c.shape.borderColor,
                       c.shape.blur, c.shape.arc.innerRadius, c.shape.asset, c.shape.z, linearize);
      break;
    case FG_Category_CIRCLE:
      context->DrawCircle(*c.shape.area, c.shape.fillColor, c.shape.border, c.shape.borderColor, c.shape.blur,
                          c.shape.circle.innerRadius, c.shape.circle.innerBorder, c.shape.asset, c.shape.z, linearize);
      break;
    case FG_Category_RECT:
      context->DrawRect(*c.shape.area, *c.shape.rect.corners, c.shape.fillColor, c.shape.border, c.shape.borderColor,
                        c.shape.blur, c.shape.asset, c.shape.rect.rotate, c.shape.z, linearize);
      break;
    case FG_Category_TRIANGLE:
      context->DrawTriangle(*c.shape.area, *c.shape.triangle.corners, c.shape.fillColor, c.shape.border,
                            c.shape.borderColor, c.shape.blur, c.shape.asset, c.shape.triangle.rotate, c.shape.z,
                            linearize);
      break;
    case FG_Category_TEXT:
      context->DrawTextGL(c.text.font, c.text.layout, c.text.area, c.text.color, c.text.blur, c.text.rotate, c.text.z,
                          linearize);
      break;
    case FG_Category_ASSET:
      context->DrawAsset(c.asset.asset, c.asset.area, c.asset.source, c.asset.color, c.asset.time, c.asset.rotate,
                         c.asset.z, linearize);
      break;
    case FG_Category_LINES: context->DrawLines(c.lines.points, c.lines.count, c.lines.color, linearize); break;
    case FG_Category_CURVE:
      context->DrawCurve(c.curve.points, c.curve.count, c.curve.fillColor, c.curve.stroke, c.curve.strokeColor, linearize);
      break;
    case FG_Category_SHADER:
      context->DrawShader(c.shader.shader, c.shader.vertices, c.shader.indices, c.shader.values);
      break;
    default: return -20;
    }
  }
  return 0;
}

bool Backend::Clear(FG_Backend* self, FG_Window* window, FG_Color color)
{
  auto backend = static_cast<Backend*>(self);

  glClearColor(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a / 255.0f);
  backend->LogError("glClearColor");
  glClear(GL_COLOR_BUFFER_BIT);
  backend->LogError("glClear");

  return true;
}

FG_Err Backend::PushLayer(FG_Backend* self, FG_Window* window, FG_Asset* layer, float* transform, float opacity,
                          FG_BlendState* blend)
{
  if(!self || !window)
    return -1;
  static_cast<Context*>(window)->PushLayer(static_cast<Layer*>(layer), transform, opacity, blend);
  return 0;
}

FG_Err Backend::PopLayer(FG_Backend* self, FG_Window* window)
{
  return !window ? -1 : static_cast<Context*>(window)->PopLayer();
}

FG_Err Backend::PushClip(FG_Backend* self, FG_Window* window, FG_Rect* area)
{
  if(!area)
    return -1;
  static_cast<Context*>(window)->PushClip(*area);
  return 0;
}

FG_Err Backend::PopClip(FG_Backend* self, FG_Window* window)
{
  static_cast<Context*>(window)->PopClip();
  return 0;
}

FG_Err Backend::DirtyRect(FG_Backend* self, FG_Window* window, FG_Rect* area)
{
  if(!self || !window)
    return -1;

  static_cast<Context*>(window)->DirtyRect(area);
  return 0;
}

FG_Shader* Backend::CreateShader(FG_Backend* self, const char* ps, const char* vs, const char* gs, const char* cs,
                                 const char* ds, const char* hs, FG_ShaderParameter* parameters, uint32_t n_parameters)
{
  return new Shader(ps, vs, gs, parameters, n_parameters);
}
FG_Err Backend::DestroyShader(FG_Backend* self, FG_Shader* shader)
{
  if(!self || !shader)
    return -1;
  delete static_cast<Shader*>(shader);
  return 0;
}

FG_Err Backend::GetProjection(FG_Backend* self, FG_Window* window, FG_Asset* layer, float* proj4x4)
{
  if(!self || !window || !proj4x4)
    return -1;

  if(!layer)
    memcpy(proj4x4, static_cast<Context*>(window)->proj, 4 * 4 * sizeof(float));
  else
    memcpy(proj4x4, static_cast<Layer*>(layer)->proj, 4 * 4 * sizeof(float));
  return 0;
}

FG_Font* Backend::CreateFontGL(FG_Backend* self, const char* family, unsigned short weight, bool italic, unsigned int pt,
                               FG_Vec dpi, FG_AntiAliasing aa)
{
  return new Font(static_cast<Backend*>(self), family, weight, italic, pt, aa, dpi);
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
  auto r =
    f->GetIndex(layout->text, area->right - area->left, layout->breakstyle, layout->lineheight, layout->letterspacing, pos);
  cursor->x = r.second.x;
  cursor->y = r.second.y;
  return r.first;
}

FG_Vec Backend::FontPos(FG_Backend* self, FG_Font* font, void* fontlayout, FG_Rect* area, uint32_t index)
{
  if(!self || !font || !fontlayout || !area)
    return { NAN, NAN };
  auto layout = reinterpret_cast<TextLayout*>(fontlayout);
  Font* f     = static_cast<Font*>(font);
  auto r =
    f->GetPos(layout->text, area->right - area->left, layout->breakstyle, layout->lineheight, layout->letterspacing, index);
  FG_Vec c = { r.second.x, r.second.y };
  return c;
}

FG_Asset* Backend::CreateAsset(FG_Backend* self, const char* data, uint32_t count, FG_Format format, int flags)
{
  auto backend = static_cast<Backend*>(self);
  size_t len   = !count ? strlen(data) + 1 : count;
  void* image;
  int width, height, channels;
  if(!count)
    image = SOIL_load_image(data, &width, &height, &channels, SOIL_LOAD_AUTO);
  else
    image = SOIL_load_image_from_memory(reinterpret_cast<const unsigned char*>(data), count, &width, &height, &channels,
                                        SOIL_LOAD_AUTO);

  if(!image)
  {
    (*backend->_log)(backend->_root, FG_Level_ERROR, "%s failed!",
                     !count ? "SOIL_load_image" : "SOIL_load_image_from_memory");
    return nullptr;
  }

  // We don't currently force channels, but if we do in the future this check needs to happen
  // if( (force_channels >= 1) && (force_channels <= 4) )
  //	channels = force_channels;

  Asset* asset     = reinterpret_cast<Asset*>(malloc(sizeof(Asset)));
  asset->data.data = image;
  asset->count     = count;
  asset->format    = format;
  asset->size.x    = width;
  asset->size.y    = height;
  asset->channels  = channels;
  asset->flags     = flags;
  int r;
  kh_put_assets(backend->_assethash, asset, &r);
  return asset;
}

FG_Asset* Backend::CreateBuffer(FG_Backend* self, void* data, uint32_t bytes, uint8_t primitive,
                                FG_ShaderParameter* parameters, uint32_t n_parameters)
{
  auto backend    = static_cast<Backend*>(self);
  uint32_t count  = 0;
  uint32_t stride = 0;

  for(uint32_t i = 0; i < n_parameters; ++i)
  {
    GLenum type = 0;
    switch(parameters[i].type)
    {
    case FG_ShaderType_FLOAT: type = GL_FLOAT; break;
    case FG_ShaderType_INT: type = GL_INT; break;
    case FG_ShaderType_UINT: type = GL_UNSIGNED_INT; break;
    }
    stride += Context::GetBytes(type) * Context::GetMultiCount(parameters[i].length, parameters[i].multi);
  }

  switch(primitive)
  {
  case FG_Primitive_INDEX_BYTE: stride = sizeof(char); break;
  case FG_Primitive_INDEX_SHORT: stride = sizeof(short); break;
  case FG_Primitive_INDEX_INT: stride = sizeof(int); break;
  }

  if((bytes % stride) > 0)
  {
    (*backend->_log)(backend->_root, FG_Level_ERROR,
                     "%u bytes can't be evenly divided by %u stride (remainder %u), required by primitive type %hhu!",
                     bytes, stride, bytes % stride, primitive);
    return 0;
  }

  count = bytes / stride;

  switch(primitive)
  {
  case FG_Primitive_TRIANGLE:
    if((count % 3) != 0)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "A list of triangles must have a multiple of 3 vertices, got %u!",
                       count);
      return 0;
    }
  case FG_Primitive_TRIANGLE_STRIP:
    if(count < 3)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "Can't have less than 3 vertices for triangles, got %u!", count);
      return 0;
    }
    break;
  case FG_Primitive_LINE:
    if((count % 2) != 0)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "A list of lines must have a multiple of 2 vertices, got %u!",
                       count);
      return 0;
    }
  case FG_Primitive_LINE_STRIP:
    if(count < 2)
    {
      (*backend->_log)(backend->_root, FG_Level_ERROR, "Can't have less than 2 vertices for lines, got %u!", count);
      return 0;
    }
    break;
  }

  Asset* asset     = reinterpret_cast<Asset*>(malloc(sizeof(Asset) + bytes + sizeof(FG_ShaderParameter) * n_parameters));
  asset->format    = FG_Format_BUFFER;
  asset->count     = count;
  asset->stride    = stride;
  asset->primitive = primitive;
  asset->data.data = asset + 1;
  memcpy(asset->data.data, data, bytes);
  asset->parameters   = reinterpret_cast<FG_ShaderParameter*>(reinterpret_cast<char*>(asset + 1) + bytes);
  asset->n_parameters = n_parameters;
  memcpy(asset->parameters, parameters, sizeof(FG_ShaderParameter) * n_parameters);

  int r;
  kh_put_assets(backend->_assethash, asset, &r);
  return asset;
}

FG_Asset* Backend::CreateLayer(FG_Backend* self, FG_Window* window, FG_Vec* size, int flags)
{
  return static_cast<Context*>(window)->CreateLayer(size, flags);
}

FG_Err Backend::DestroyAsset(FG_Backend* self, FG_Asset* fgasset)
{
  if(!self || !fgasset)
    return -1;

  if(fgasset->format == FG_Format_LAYER)
  {
    delete static_cast<Layer*>(fgasset);
    return 0;
  }

  auto backend = static_cast<Backend*>(self);

  // Once we remove the asset from the hash, contexts who reach a reference count of 0 for an asset will destroy it.
  khiter_t iter = kh_get_assets(backend->_assethash, fgasset);
  if(iter < kh_end(backend->_assethash))
    kh_del_assets(backend->_assethash, iter);

  free(static_cast<Asset*>(fgasset));
  return 0;
}

void* Backend::CreateSystemControl(FG_Backend* self, FG_Window* window, const char* id, FG_Rect* area, ...) { return 0; }
FG_Err Backend::SetSystemControl(FG_Backend* self, FG_Window* window, void* control, FG_Rect* area, ...) { return -1; }
FG_Err Backend::DestroySystemControl(FG_Backend* self, FG_Window* window, void* control) { return -1; }

FG_Err Backend::PutClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, const char* data, uint32_t count)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return -1;
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
  return 0;
#else
  if(kind != FG_Clipboard_TEXT)
    return -1;

  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), data);
  return _lasterr;
#endif
}

uint32_t Backend::GetClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind, void* target, uint32_t count)
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
    return 0;

  auto str = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(target)
    strncpy(reinterpret_cast<char*>(target), str, count);
  return strlen(str) + 1;
#endif
}

bool Backend::CheckClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind)
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
  if(kind != FG_Clipboard_TEXT && kind != FG_Clipboard_ALL)
    return false;

  auto p = glfwGetClipboardString(static_cast<Context*>(window)->GetWindow());
  if(!p)
    return false;
  return p[0] != 0;
#endif
}

FG_Err Backend::ClearClipboard(FG_Backend* self, FG_Window* window, FG_Clipboard kind)
{
#ifdef FG_PLATFORM_WIN32
  if(!OpenClipboard(GetActiveWindow()))
    return -1;
  if(!EmptyClipboard())
    return -1;
  CloseClipboard();
  return 0;
#else
  _lasterr = 0;
  glfwSetClipboardString(static_cast<Context*>(window)->GetWindow(), "");
  return _lasterr;
#endif
}

FG_Err Backend::ProcessMessages(FG_Backend* self)
{
  auto backend = static_cast<Backend*>(self);
  glfwPollEvents();

  // TODO: Process all joystick events

  return backend->_windows != nullptr;
}

FG_Err Backend::SetCursorGL(FG_Backend* self, FG_Window* window, FG_Cursor cursor)
{
  static GLFWcursor* arrow   = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
  static GLFWcursor* ibeam   = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
  static GLFWcursor* cross   = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
  static GLFWcursor* hand    = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
  static GLFWcursor* hresize = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
  static GLFWcursor* vresize = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);

  auto glwindow = static_cast<Context*>(window)->GetWindow();
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

FG_Err Backend::GetDisplayWindow(FG_Backend* self, FG_Window* window, FG_Display* out)
{
  return GetDisplay(self, glfwGetWindowMonitor(static_cast<Context*>(window)->GetWindow()), out);
}

GLFWglproc glGetProcAddress(const char* procname)
{
#ifdef FG_PLATFORM_WIN32
  return (GLFWglproc)GetProcAddress((HMODULE)Backend::_library, procname);
#else
  return (GLFWglproc)dlsym(Backend::_library, procname);
#endif
}

FG_Window* Backend::CreateRegionGL(FG_Backend* self, FG_MsgReceiver* element, FG_Window desc, FG_Vec3 pos, FG_Vec3 dim)
{
  auto backend                      = reinterpret_cast<Backend*>(self);
  FG_Vec d                          = { dim.x, dim.y };
  Context* context                  = new Context(backend, element, &d);
  *static_cast<FG_Window*>(context) = desc;

  // glfwMakeContextCurrent(_window); // TODO: Make the context current
  if(!gladLoadGL(glGetProcAddress))
    (*backend->_log)(backend->_root, FG_Level_ERROR, "gladLoadGL failed");
  backend->LogError("gladLoadGL");
  context->CreateResources();

  return context;
}

FG_Window* Backend::CreateWindowGL(FG_Backend* self, FG_MsgReceiver* element, void* display, FG_Vec* pos, FG_Vec* dim,
                                   const char* caption, uint64_t flags)
{
  auto backend = reinterpret_cast<Backend*>(self);
  _lasterr     = 0;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  Window* window =
    new Window(static_cast<Backend*>(self), reinterpret_cast<GLFWmonitor*>(display), element, pos, dim, flags, caption);

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

FG_Err Backend::SetWindowGL(FG_Backend* self, FG_Window* window, FG_MsgReceiver* element, void* display, FG_Vec* pos,
                            FG_Vec* dim, const char* caption, uint64_t flags)
{
  if(!window)
    return -1;
  auto context      = static_cast<Context*>(window);
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
  return 0;
}

FG_Err Backend::DestroyWindow(FG_Backend* self, FG_Window* window)
{
  if(!self || !window)
    return -1;

  _lasterr = 0;
  delete static_cast<Context*>(window);
  return _lasterr;
}
FG_Err Backend::BeginDraw(FG_Backend* self, FG_Window* window, FG_Rect* area)
{
  if(!window)
    return -1;
  _lasterr = 0;
  static_cast<Context*>(window)->BeginDraw(area);
  return _lasterr;
}
FG_Err Backend::EndDraw(FG_Backend* self, FG_Window* window)
{
  _lasterr = 0;
  static_cast<Context*>(window)->EndDraw();
  return _lasterr;
}

void DestroyGL(FG_Backend* self)
{
  if(!self)
    return;
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
    Backend::_library = dlopen("libGL-1.so");
  #endif
    if(!Backend::_library)
      Backend::_library = dlopen("libGL.so.1");
    if(!Backend::_library)
      Backend::_library = dlopen("libGL.so");

    FcInit();
#endif
    glfwSetErrorCallback(&Backend::ErrorCallback);

    if(!glfwInit())
    {
      (*log)(root, FG_Level_ERROR, "glfwInit() failed with %i! %s", Backend::_lasterr, Backend::_lasterrdesc);
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
  _root(root), _log(log), _behavior(behavior), _assethash(kh_init_assets()), _windows(nullptr)
{
  draw                 = &DrawGL;
  clear                = &Clear;
  pushLayer            = &PushLayer;
  popLayer             = &PopLayer;
  pushClip             = &PushClip;
  popClip              = &PopClip;
  dirtyRect            = &DirtyRect;
  beginDraw            = &BeginDraw;
  endDraw              = &EndDraw;
  createShader         = &CreateShader;
  destroyShader        = &DestroyShader;
  createFont           = &CreateFontGL;
  destroyFont          = &DestroyFont;
  fontLayout           = &FontLayout;
  destroyLayout        = &DestroyLayout;
  fontIndex            = &FontIndex;
  fontPos              = &FontPos;
  createAsset          = &CreateAsset;
  createBuffer         = &CreateBuffer;
  createLayer          = &CreateLayer;
  destroyAsset         = &DestroyAsset;
  getProjection        = &GetProjection;
  putClipboard         = &PutClipboard;
  getClipboard         = &GetClipboard;
  checkClipboard       = &CheckClipboard;
  clearClipboard       = &ClearClipboard;
  processMessages      = &ProcessMessages;
  setCursor            = &SetCursorGL;
  getDisplayIndex      = &GetDisplayIndex;
  getDisplay           = &GetDisplay;
  getDisplayWindow     = &GetDisplayWindow;
  createRegion         = &CreateRegionGL;
  createWindow         = &CreateWindowGL;
  setWindow            = &SetWindowGL;
  destroyWindow        = &DestroyWindow;
  destroy              = &DestroyGL;
  createSystemControl  = &CreateSystemControl;
  setSystemControl     = &SetSystemControl;
  destroySystemControl = &DestroySystemControl;

  (*_log)(_root, FG_Level_NONE, "Initializing fgOpenGL...");
  if(FT_Error err = FT_Init_FreeType(&_ftlib))
    (*_log)(_root, FG_Level_ERROR, "Error %i occured while initializing FreeType", err);
  ;

  const char* image_vs =
#include "Image.vs.glsl"
    ;

  const char* image_fs =
#include "Image.fs.glsl"
    ;

  _imageshader =
    Shader(image_fs, image_vs, 0, { { FG_ShaderType_FLOAT, 4, 4, "MVP" }, { FG_ShaderType_TEXTURE, 0, 0, "texture" } });

  const char* line_vs =
#include "Line.vs.glsl"
    ;

  const char* line_fs =
#include "Line.fs.glsl"
    ;

  _lineshader =
    Shader(line_fs, line_vs, 0, { { FG_ShaderType_FLOAT, 4, 4, "MVP" }, { FG_ShaderType_FLOAT, 4, 1, "Color" } });

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

  const char* arc_fs =
#include "Arc.fs.glsl"
    ;

  std::initializer_list<FG_ShaderParameter> standardLayout = {
    { FG_ShaderType_FLOAT, 4, 4, "MVP" },     { FG_ShaderType_FLOAT, 4, 1, "DimBorderBlur" },
    { FG_ShaderType_FLOAT, 4, 1, "Corners" }, { FG_ShaderType_FLOAT, 4, 1, "Fill" },
    { FG_ShaderType_FLOAT, 4, 1, "Outline" },
  };

  _rectshader   = Shader(roundrect_fs, standard_vs, 0, standardLayout);
  _trishader    = Shader(triangle_fs, standard_vs, 0, standardLayout);
  _circleshader = Shader(circle_fs, standard_vs, 0, standardLayout);
  _arcshader    = Shader(arc_fs, standard_vs, 0, standardLayout);

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

  HRESULT hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory1),
                                   reinterpret_cast<IUnknown**>(&_writefactory));
  if(FAILED(hr))
    (*_log)(_root, FG_Level_ERROR, "DWriteCreateFactory() failed with error: %li", hr);

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

  if(_writefactory)
    _writefactory->Release();
#endif

  FT_Done_FreeType(_ftlib);
  kh_destroy_assets(_assethash);
}
void Backend::ErrorCallback(int error, const char* description)
{
  _lasterr = error;
  strncpy(_lasterrdesc, description, 1024);
  if(_singleton)
    (*_singleton->_log)(_singleton->_root, FG_Level_ERROR, description);
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
