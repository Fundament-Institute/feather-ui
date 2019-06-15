// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__BACKEND_H
#define FG__BACKEND_H

#include "message.h"

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
  FEATURE_LINES_ALPHA        = (1 << 12),
  FEATURE_CURVE_STROKE       = (1 << 13), // If both stroke and fill are false, it doesn't support curves at all
  FEATURE_CURVE_FILL         = (1 << 14),
  FEATURE_LAYER_TRANSFORM    = (1 << 15),
  FEATURE_LAYER_OPACITY      = (1 << 16),
  FEATURE_SHADER_GLSL2       = (1 << 17),
  FEATURE_SHADER_GLSL4       = (1 << 18),
  FEATURE_SHADER_HLSL2       = (1 << 19),
  FEATURE_BACKGROUND_OPACITY = (1 << 20),
  FEATURE_IMMEDIATE_MODE     = (1 << 21), // Does not indicate a feature, simply signifies whether this tends to redraw every frame or tries to only update dirty regions.
};

enum FG_BACKEND_FORMATS
{
  BACKEND_BMP = 0,
  BACKEND_JPG,
  BACKEND_PNG,
  BACKEND_GIF,
  BACKEND_TIFF,
  BACKEND_TGA,
  BACKEND_DDS,
  BACKEND_WIC,
  BACKEND_SVG,
  BACKEND_AVI,
  BACKEND_MP4,
  BACKEND_MKV,
  BACKEND_WEBM,
  BACKEND_FROM_DATA,
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
  fgRect area;
  fgVeci dpi;
  fgBehaviorFunction behavior;
  bool main;
} fgDisplay;

typedef void* fgFontPtr;
typedef void* fgLayoutPtr;

typedef struct
{
  fgFontPtr ptr;
  fgVeci dpi;
  float baseline;
  float lineheight;
  unsigned int pt;
} fgFont;

typedef void* fgAssetPtr;

typedef struct
{
  fgAssetPtr ptr;
  fgVeci size;
  fgVeci dpi;
  FG_BACKEND_FORMATS format;
} fgAsset;

typedef struct FG__BACKEND
{
  fgFlag features;
  fgFlag formats;

  fgError (*beginScene)(struct FG__BACKEND* self, const fgRect* dirty, unsigned int count);
  fgError (*drawFont)(struct FG__BACKEND* self, fgFontPtr font, fgLayoutPtr layout, fgRect area, fgColor color, float lineHeight, float letterSpacing, float blur, enum FG_TEXT_ANTIALIASING aa);
  fgError (*drawAsset)(struct FG__BACKEND* self, fgAssetPtr asset, fgRect area, fgRect source, float time);
  fgError (*drawRect)(struct FG__BACKEND* self, const fgRect* area, const fgRect* corners, fgColor fillColor, float border, fgColor borderColor, float blur);
  fgError (*drawCircle)(struct FG__BACKEND* self, const fgRect* area, const fgRect* arcs, fgColor fillColor, float border, fgColor borderColor, float blur);
  fgError (*drawLines)(struct FG__BACKEND* self, const fgVec* points, size_t count, fgColor color);
  fgError (*drawCurve)(struct FG__BACKEND* self, const fgVec* anchors, size_t count, fgColor fillColor, float stroke, fgColor strokeColor);
  // fgError(*drawShader)(struct FG__BACKEND* self, fgShader);
  fgError (*pushLayer)(struct FG__BACKEND* self, const float* transform, float opacity);
  fgError (*popLayer)(struct FG__BACKEND* self);
  fgError (*clipRect)(struct FG__BACKEND* self, fgRect area);
  fgError (*endScene)(struct FG__BACKEND* self);

  fgFont (*createFont)(struct FG__BACKEND* self, const char* family, unsigned short weight, bool italic, unsigned int pt, fgVeci dpi);
  fgError (*destroyFont)(struct FG__BACKEND* self, fgFontPtr font);
  fgLayoutPtr (*fontLayout)(struct FG__BACKEND* self, fgFontPtr font, const char* text, fgRect* area, float lineHeight, float letterSpacing, fgLayoutPtr prev);
  size_t (*fontIndex)(struct FG__BACKEND* self, fgFontPtr font, fgLayoutPtr layout, const fgRect* area, float lineHeight, float letterSpacing, fgVec pos, fgVec* cursor, fgVeci dpi);
  fgVec (*fontPos)(struct FG__BACKEND* self, fgFontPtr font, fgLayoutPtr layout, const fgRect* area, float lineHeight, float letterSpacing, size_t index, fgVeci dpi);

  fgAsset (*createAsset)(struct FG__BACKEND* self, const char* data, size_t count, enum FG_BACKEND_FORMATS format);
  fgError (*destroyAsset)(struct FG__BACKEND* self, fgAssetPtr asset);

  fgError (*putClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, const char* data, size_t count);
  const char* (*getClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind, size_t* count);
  bool (*checkClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind);
  fgError (*clearClipboard)(struct FG__BACKEND* self, enum FG_CLIPBOARD kind);

  fgError (*getDisplays)(struct FG__BACKEND* self, unsigned int index, fgDisplay* out);
  fgError (*processMessages)(struct FG__BACKEND* self);
  fgError (*setCursor)(struct FG__BACKEND* self, enum FG_CURSOR cursor);
  void (*destroy)(struct FG__BACKEND* self);
} fgBackend;

typedef fgBackend*(*fgInitBackend)();

#endif