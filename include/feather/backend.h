// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__BACKEND_H
#define FG__BACKEND_H

#include "message.h"
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

enum FG_BACKEND_FEATURES
{
  FEATURE_TEXT_ANTIALIAS     = (1 << 0),
  FEATURE_TEXT_SUBPIXEL      = (1 << 1),
  FEATURE_TEXT_BLUR          = (1 << 2),
  FEATURE_TEXT_ALPHA         = (1 << 3),
  FEATURE_RECT_CORNERS       = (1 << 4),
  FEATURE_RECT_BORDER        = (1 << 5),
  FEATURE_RECT_BLUR          = (1 << 6),
  FEATURE_RECT_ALPHA         = (1 << 7),
  FEATURE_CIRCLE_ARCS        = (1 << 8),
  FEATURE_CIRCLE_BORDER      = (1 << 9),
  FEATURE_CIRCLE_BLUR        = (1 << 10),
  FEATURE_CIRCLE_ALPHA       = (1 << 11),
  FEATURE_TRIANGLE_CORNERS   = (1 << 12),
  FEATURE_TRIANGLE_BORDER    = (1 << 13),
  FEATURE_TRIANGLE_BLUR      = (1 << 14),
  FEATURE_TRIANGLE_ALPHA     = (1 << 15),
  FEATURE_LINES_ALPHA        = (1 << 16),
  FEATURE_CURVE_STROKE       = (1 << 17), // If both stroke and fill are false, it doesn't support curves at all
  FEATURE_CURVE_FILL         = (1 << 18),
  FEATURE_LAYER_TRANSFORM    = (1 << 19),
  FEATURE_LAYER_OPACITY      = (1 << 20),
  FEATURE_SHADER_GLSL2       = (1 << 21),
  FEATURE_SHADER_GLSL4       = (1 << 22),
  FEATURE_SHADER_HLSL2       = (1 << 23),
  FEATURE_BACKGROUND_OPACITY = (1 << 24),
  FEATURE_IMMEDIATE_MODE     = (1 << 25), // Does not indicate a feature, simply signifies whether this tends to redraw every frame or tries to only update dirty regions.
};

enum FG_BACKEND_FORMATS
{
  BACKEND_GRADIENT,
  BACKEND_BMP,
  BACKEND_JPG,
  BACKEND_PNG,
  BACKEND_ICO,
  BACKEND_GIF,
  BACKEND_TIFF,
  BACKEND_TGA,
  BACKEND_WEBP,
  BACKEND_DDS,
  BACKEND_WIC,
  BACKEND_SVG,
  BACKEND_AVI,
  BACKEND_MP4,
  BACKEND_MKV,
  BACKEND_WEBM,
  BACKEND_UNKNOWN = 0xff,
};

enum FG_TEXT_ANTIALIASING
{
  FG_TEXT_NO_AA,
  FG_TEXT_AA,
  FG_TEXT_LCD,
  FG_TEXT_LCD_V,
};

enum FG_CLIPBOARD
{
  CLIPBOARD_NONE = 0,
  CLIPBOARD_TEXT,
  CLIPBOARD_WAVE,
  CLIPBOARD_BITMAP,
  CLIPBOARD_FILE,
  CLIPBOARD_ELEMENT,
  CLIPBOARD_CUSTOM,
  CLIPBOARD_ALL,
};

typedef struct
{
  void* data;
  fgVec dpi;
  float baseline;
  float lineheight;
  unsigned int pt;
} fgFont;

typedef struct
{
  void* data;
  fgVeci size;
  fgVec dpi;
  FG_BACKEND_FORMATS format;
} fgAsset;

typedef void* fgFontLayout;
struct FG__OUTLINE_NODE;

typedef struct FG__BACKEND
{
  fgError (*drawFont)(struct FG__BACKEND* self, void* data, const fgFont* font, fgFontLayout layout, const fgRect* area, fgColor color, float lineHeight, float letterSpacing, float blur, enum FG_TEXT_ANTIALIASING aa);
  fgError (*drawAsset)(struct FG__BACKEND* self, void* data, const fgAsset* asset, const fgRect* area, const fgRect* source, fgColor color, float time);
  fgError (*drawRect)(struct FG__BACKEND* self, void* data, const fgRect* area, const fgRect* corners, fgColor fillColor, float border, fgColor borderColor, float blur, const fgAsset* asset);
  fgError (*drawCircle)(struct FG__BACKEND* self, void* data, const fgRect* area, const fgRect* arcs, fgColor fillColor, float border, fgColor borderColor, float blur, const fgAsset* asset);
  fgError (*drawTriangle)(struct FG__BACKEND* self, void* data, const fgRect* area, const fgRect* corners, fgColor fillColor, float border, fgColor borderColor, float blur, const fgAsset* asset);
  fgError (*drawLines)(struct FG__BACKEND* self, void* data, const fgVec* points, size_t count, fgColor color);
  fgError (*drawCurve)(struct FG__BACKEND* self, void* data, const fgVec* anchors, size_t count, fgColor fillColor, float stroke, fgColor strokeColor);
  // fgError(*drawShader)(struct FG__BACKEND* self, void* data,fgShader);
  fgError (*pushLayer)(struct FG__BACKEND* self, void* data, fgRect area, const float* transform, float opacity);
  fgError (*popLayer)(struct FG__BACKEND* self, void* data);
  fgError(*pushClip)(struct FG__BACKEND* self, void* data, fgRect area);
  fgError(*popClip)(struct FG__BACKEND* self, void* data);
  fgError (*dirtyRect)(struct FG__BACKEND* self, void* data, fgRect area);

  fgFont* (*createFont)(struct FG__BACKEND* self, const char* family, unsigned short weight, bool italic, unsigned int pt, fgVec dpi);
  fgError (*destroyFont)(struct FG__BACKEND* self, fgFont* font);
  fgFontLayout (*fontLayout)(struct FG__BACKEND* self, fgFont* font, const char* text, fgRect* area, float lineHeight, float letterSpacing, fgFontLayout prev, fgVec dpi);
  size_t (*fontIndex)(struct FG__BACKEND* self, fgFont* font, fgFontLayout layout, const fgRect* area, float lineHeight, float letterSpacing, fgVec pos, fgVec* cursor, fgVec dpi);
  fgVec (*fontPos)(struct FG__BACKEND* self, fgFont* font, fgFontLayout layout, const fgRect* area, float lineHeight, float letterSpacing, size_t index, fgVec dpi);

  fgAsset* (*createAsset)(struct FG__BACKEND* self, const char* data, size_t count, enum FG_BACKEND_FORMATS format);
  fgError (*destroyAsset)(struct FG__BACKEND* self, fgAsset* asset);

  fgError (*putClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, const char* data, size_t count);
  size_t (*getClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, void* target, size_t count);
  bool (*checkClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind);
  fgError (*clearClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind);

  fgError (*processMessages)(struct FG__ROOT* root, struct FG__BACKEND* self);
  fgError (*setCursor)(struct FG__BACKEND* self, void* data, enum FG_CURSOR cursor);
  fgError (*requestAnimationFrame)(struct FG__BACKEND* self, struct FG__DOCUMENT_NODE* node, unsigned long long microdelay); // Requests that a frame be drawn some number of microseconds after the last frame began (NOT when this function was called). If 0, requests next screen refresh.
  void (*destroy)(struct FG__BACKEND* self);

  fgFlag features;
  fgFlag formats;
  fgVec dpi; // System-wide DPI, but not necessarily the DPI of each display
  float scale; // System-wide text scaling value
  struct FG__OUTLINE_NODE** displays; // This is a list of displays, expressed as fgOutlineNodes. These displays may themselves belong to a displaygroup, but this is optional.
  unsigned int n_displays;
  unsigned long long cursorblink; // milliseconds
  unsigned long long tooltipdelay; // milliseconds
} fgBackend;

struct FG__ROOT;
typedef fgBackend*(*fgInitBackend)(struct FG__ROOT*);

FG_COMPILER_DLLEXPORT fgInitBackend fgLoadBackend(const char* path, void** library, const char* name);
FG_COMPILER_DLLEXPORT void fgFreeBackend(void* library);
FG_COMPILER_DLLEXPORT size_t fgUTF8toUTF16(const char* FG_RESTRICT input, ptrdiff_t srclen, wchar_t* FG_RESTRICT output, size_t buflen);
FG_COMPILER_DLLEXPORT size_t fgUTF16toUTF8(const wchar_t* FG_RESTRICT input, ptrdiff_t srclen, char* FG_RESTRICT output, size_t buflen);

#ifdef  __cplusplus
}
#endif

#endif