// Copyright (c)2020 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#ifndef GL__FONT_H
#define GL__FONT_H

#include "../backend.h"
#include "compiler.h"
#include <filesystem>
#include "khash.h"

struct FT_FaceRec_;

namespace GL {
  class Backend;
  struct Context;

  // Internal Glyph object tracking an individual glyph
  struct Glyph
  {
    FG_Rect uv; // This is the same on all contexts
    float advance;
    FG_Vec bearing;
    float width;
    float height;
  };

  KHASH_DECLARE(glyphmap, int, Glyph);

  // Internal Font object
  struct Font : FG_Font
  {
    Font(Backend* backend, const char* font, int psize, FG_AntiAliasing antialias, const FG_Vec& dpi);
    ~Font();
    Glyph* LoadGlyph(char32_t codepoint);
    float GetKerning(char32_t prev, char32_t cur);
    FG_Vec CalcTextDim(const char32_t* text, const FG_Vec& maxdim, float lineheight, float letterspacing,
                       FG_BreakStyle breakstyle);
    float GetLineWidth(const char32_t*& text, float maxwidth, FG_BreakStyle breakstyle, float letterspacing);
    Glyph* RenderGlyph(Context* context, char32_t codepoint);
    std::pair<size_t, FG_Vec> GetIndex(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle, float lineheight,
                                       float letterspacing, FG_Vec pos);
    std::pair<size_t, FG_Vec> GetPos(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle, float lineheight,
                                     float letterspacing, size_t index);
    inline int GetSizePower() const { return _curpower; }
    inline float GetAscender() const { return _ascender; }

  protected:
    void _cleanup();
    void _enforceantialias(int ftaa);
    int _ftaa(FG_AntiAliasing antialias);
    bool _isspace(int c);
    Glyph* _getchar(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle, float lineheight, float letterspacing,
                    FG_Vec& cursor, FG_Rect& box, char32_t& last, float& lastadvance, bool& dobreak);

    Backend* _backend;
    std::filesystem::path _path;
    unsigned int _texture;
    struct FT_FaceRec_* _face;
    float _ascender;
    float _descender;
    bool _haskerning;
    kh_glyphmap_t* _glyphs;
    FG_Vec _cur;
    FG_Vec _last; // holds the exclusion zone of the last texture size (if any)
    int _curpower;
    float _nexty;
  };

  struct TextLayout
  {
    char32_t* text;
    const char32_t** lines;
    size_t n_lines;
    float letterspacing;
    float lineheight;
    FG_Rect area;
    FG_BreakStyle breakstyle;
  };
}

#endif