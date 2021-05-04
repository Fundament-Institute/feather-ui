// Copyright (c)2021 Fundament Software
// For conditions of distribution and use, see copyright notice in "fgOpenGL.h"

#include "BackendGL.h"
#include "Font.h"
#include "platform.h"
#include "ft2build.h"
#include FT_FREETYPE_H
#include "freetype/freetype.h"
#include <assert.h>
#include <malloc.h>
#include <math.h>

#ifdef FG_PLATFORM_WIN32
  #include <Shlobj.h>
  #include <dwrite_1.h>
#endif

namespace GL {
  __KHASH_IMPL(glyphmap, , int, Glyph, 1, kh_int_hash_func2, kh_int_hash_equal);
}

using namespace GL;

Font::Font(Backend* backend, const char* family, int weight, bool italic, int psize, FG_AntiAliasing antialias,
           const FG_Vec& _dpi) :
  _backend(backend), _path(family), _glyphs(kh_init_glyphmap()), _curpower(0), _nexty(1)
{
  _cur = { 1, 1 };
  pt   = psize;
  dpi  = _dpi;
  aa   = antialias;
  FT_Error err;
#ifdef FG_PLATFORM_WIN32
  if(exists(_path)) // Check if we were just passed an entire path instead of a font family
    err = FT_New_Face(_backend->_ftlib, _path.u8string().c_str(), 0, &_face);
  else
  {
    IDWriteTextFormat* format = 0;
    wchar_t wlocale[LOCALE_NAME_MAX_LENGTH];
    GetSystemDefaultLocaleName(wlocale, LOCALE_NAME_MAX_LENGTH);
    HRESULT hr = _backend->_writefactory->CreateTextFormat(_path.c_str(), 0, DWRITE_FONT_WEIGHT(weight),
                                                           italic ? DWRITE_FONT_STYLE_ITALIC : DWRITE_FONT_STYLE_NORMAL,
                                                           DWRITE_FONT_STRETCH_NORMAL, pt * (dpi.x / 72.0f), wlocale,
                                                           &format);

    if(FAILED(hr) || !format)
    {
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Font %s does not exist or cannot be found.",
                        _path.u8string().c_str());
      return;
    }
    TCHAR name[64];
    UINT32 findex;
    BOOL exists;
    IDWriteFontCollection* collection;
    format->GetFontFamilyName(name, 64);
    format->GetFontCollection(&collection);
    collection->FindFamilyName(name, &findex, &exists);
    if(!exists) // CreateTextFormat always succeeds even for invalid font names so we have to check to see if we actually
                // loaded a real font
    {
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Font %s does not exist or cannot be found.",
                        _path.u8string().c_str());
      format->Release();
      collection->Release();
      return;
    }

    IDWriteFontFamily* ffamily = nullptr;
    hr                         = collection->GetFontFamily(findex, &ffamily);
    IDWriteFont* font          = nullptr;
    if(SUCCEEDED(hr))
      hr = ffamily->GetFirstMatchingFont(format->GetFontWeight(), format->GetFontStretch(), format->GetFontStyle(), &font);
    IDWriteFontFace* face = nullptr;
    if(SUCCEEDED(hr))
      hr = font->CreateFontFace(&face);
    UINT32 n_files = 0;
    if(SUCCEEDED(hr))
      hr = face->GetFiles(&n_files, NULL);
    IDWriteFontFile** files = (IDWriteFontFile**)ALLOCA(sizeof(IDWriteFontFile*) * n_files);

    if(SUCCEEDED(hr))
      hr = face->GetFiles(&n_files, files);

    if(SUCCEEDED(hr) && n_files > 0)
    {
      const void* refkey;
      UINT32 refkeysize;
      hr = files[0]->GetReferenceKey(&refkey, &refkeysize);

      if(SUCCEEDED(hr))
      {
        IDWriteFontFileLoader* loader = nullptr;
        files[0]->GetLoader(&loader);
        if(loader != nullptr)
        {
          IDWriteFontFileStream* stream;
          hr = loader->CreateStreamFromKey(refkey, refkeysize, &stream);
          if(SUCCEEDED(hr))
          {
            UINT64 filesize;
            const void* start;
            void* ctx;

            hr = stream->GetFileSize(&filesize);

            if(SUCCEEDED(hr))
            {
              hr = stream->ReadFileFragment(&start, 0, filesize, &ctx);

              if(SUCCEEDED(hr))
              {
                err = FT_New_Memory_Face(_backend->_ftlib, (const FT_Byte*)start, filesize, 0, &_face);
                stream->ReleaseFileFragment(ctx);
              }
            }
          }
          loader->Release();
        }
      }

      if(FAILED(hr))
      {
        (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Could not load files for Font %s: %li",
                          _path.u8string().c_str(), hr);
        return;
      }
    }
    else
    {
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Could not load files for Font %s: %li", _path.u8string().c_str(),
                        hr);
      return;
    }

    if(ffamily)
      ffamily->Release();
    if(font)
      font->Release();
    if(face)
      face->Release();
    for(UINT32 i = 0; i < n_files; ++i)
      files[i]->Release();
    format->Release();
    collection->Release();
  }
#else
  switch(weight)
  {
  case 100: weight = FC_WEIGHT_THIN; break;
  case 200: weight = FC_WEIGHT_EXTRALIGHT; break;
  case 300: weight = FC_WEIGHT_LIGHT; break;
  case 350: weight = FC_WEIGHT_LIGHT; break;
  case 400: weight = FC_WEIGHT_NORMAL; break;
  case 500: weight = FC_WEIGHT_MEDIUM; break;
  case 600: weight = FC_WEIGHT_SEMIBOLD; break;
  case 700: weight = FC_WEIGHT_BOLD; break;
  case 800: weight = FC_WEIGHT_EXTRABOLD; break;
  case 900: weight = FC_WEIGHT_BLACK; break;
  case 950: weight = FC_WEIGHT_EXTRABLACK; break;
  default: weight /= 5; break; // This doesn't really work but we try anyway
  }

  FcConfig* config = FcInitLoadConfigAndFonts();
  FcPattern* pat   = FcNameParse((const FcChar8*)family);
  FcPatternAddInteger(pat, FC_SIZE, pt);
  FcPatternAddInteger(pat, FC_WEIGHT, weight);
  FcPatternAddInteger(pat, FC_SLANT, italic ? 100 : 0);
  FcConfigSubstitute(config, pat, FcMatchPattern);
  FcDefaultSubstitute(pat);

  char* fontFile;
  FcResult result;
  FcPattern* match = FcFontMatch(config, pat, &result);

  if(match)
  {
    FcChar8* str = NULL;

    if(FcPatternGetString(match, FC_FILE, 0, &str) == FcResultMatch)
      err = FT_New_Face(_backend->_ftlib, (const char*)str, 0, &_face);
  }

  FcPatternDestroy(match);
  FcPatternDestroy(pat);
  FcConfigDestroy(config);
#endif

  const float FT_COEF = (1.0f / 64.0f);

  if(err != 0 || !_face)
  {
    (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Font %s does not exist or cannot be found.",
                      _path.u8string().c_str());
    return;
  }

  if(!_face->charmap)
  {
    (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Font face %s does not have a unicode character map.", family);
    _cleanup();
    return;
  }

  FT_Pos ptsize = FT_F26Dot6(pt * 64);
  if(FT_Set_Char_Size(_face, ptsize, ptsize, static_cast<FT_UInt>(floor(dpi.x)), static_cast<FT_UInt>(floor(dpi.y))) != 0)
  { // certain fonts can only be rendered at specific sizes, so we iterate through them until we hit the closest one and try
    // to use that
    int bestdif = 0x7FFFFFFF;
    int cur     = 0;
    for(int i = 0; i < _face->num_fixed_sizes; ++i)
    {
      cur = abs(_face->available_sizes[i].size - ptsize);
      if(cur < bestdif)
        bestdif = cur;
    }
    if(FT_Set_Char_Size(_face, 0, cur, 0, 0) != 0)
    {
      (*_backend->_log)(_backend->_root, FG_Level_ERROR, "Font face %s can't be rendered at size %i.", family, pt);
      _cleanup();
      return;
    }
  }

  float invdpiscale = ((dpi.x == Backend::BASE_DPI && dpi.y == Backend::BASE_DPI) ?
                         1.0f :
                         (Backend::BASE_DPI / (float)dpi.y)); // y-axis DPI scaling

  if(_face->face_flags & FT_FACE_FLAG_SCALABLE) // now account for scalability
  {
    // float x_scale = d_fontFace->size->metrics.x_scale * FT_POS_COEF * (1.0/65536.0);
    float y_scale = _face->size->metrics.y_scale * (1.0f / 65536.0f);
    _ascender     = floor(_face->ascender * y_scale * FT_COEF * invdpiscale);
    _descender    = floor(_face->descender * y_scale * FT_COEF);
    lineheight    = floor(_face->height * y_scale * FT_COEF);
  }
  else
  {
    _ascender  = floor(_face->size->metrics.ascender * FT_COEF * invdpiscale);
    _descender = floor(_face->size->metrics.descender * FT_COEF * invdpiscale);
    lineheight = floor(_face->size->metrics.height * FT_COEF * invdpiscale);
  }

  // points are defined as 1/72 inches, so the scaling factor is DPI/72.0f to get the true glyph size. Example: a 12 point
  // font at 96 DPI is 12 * 96/72 = 16 pixels high
  // This finds the next highest power of two
  _curpower   = (uint32_t)ceil(log(std::max(dpi.x, dpi.y) / 72.0f * 8.0f * pt) / log(2));
  baseline    = _ascender;
  _haskerning = FT_HAS_KERNING(_face) != 0;
  data.data   = this;
  _last       = { 0, 0 };
}

Font::~Font()
{
  _cleanup();
  kh_destroy_glyphmap(_glyphs);
}

void Font::_cleanup()
{
  if(_face)
    FT_Done_Face(_face);
  _face = nullptr;
}

int Font::_ftaa(FG_AntiAliasing antialias)
{
  switch(antialias&(~FG_AntiAliasing_SDF))
  {
  default:
  case 0: return FT_LOAD_TARGET_MONO;
  case FG_AntiAliasing_AA: return FT_LOAD_TARGET_NORMAL;
  case FG_AntiAliasing_LCD: return FT_LOAD_TARGET_LCD;
  case FG_AntiAliasing_LCD_V: return FT_LOAD_TARGET_LCD_V;
  }
}

void Font::_enforceantialias(int ftaa)
{
  /*switch(ftaa)
  {
     case FT_LOAD_TARGET_LCD:
     case FT_LOAD_TARGET_LCD_V: FT_Library_SetLcdFilter(_backend->_ftlib, FT_LCD_FILTER_LIGHT); break;
     default: FT_Library_SetLcdFilter(_backend->_ftlib, FT_LCD_FILTER_NONE); break;
  }*/
}
Glyph* Font::LoadGlyph(char32_t codepoint)
{
  int r;
  auto iter = kh_put_glyphmap(_glyphs, codepoint, &r);
  if(!r)
    return &kh_val(_glyphs, iter);
  if(r < 0)
    return nullptr;

  // if this throws an error, remove it as a possible renderable codepoint
  if(FT_Load_Char(_face, codepoint, FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT | _ftaa(aa)) != 0)
  {
    (*_backend->_log)(_backend->_root, FG_Level_ERROR, "codepoint %i in %s failed to load.", codepoint,
                      _path.u8string().c_str());
    return nullptr;
  }

  FT_Bitmap& gbmp     = _face->glyph->bitmap;
  const float FT_COEF = (1.0f / 64.0f);
  uint32_t width      = (gbmp.pixel_mode == FT_PIXEL_MODE_LCD) ? (gbmp.width / 3) : gbmp.width;
  uint32_t height     = gbmp.rows;
  FG_Vec offset       = _cur;

  Glyph& g = kh_val(_glyphs, iter);
  g        = Glyph{};

  if(_cur.x + width + 1 > (1 << _curpower)) // if true we ran past the edge (+1 for one pixel buffer on edges)
  {
    _cur.y = _nexty;
    _cur.x = (_cur.y < _last.y) ? 1 + _last.x : 1;
  }
  if(_cur.y + height + 1 > (1 << _curpower)) // if true we need to resize the texture
  {
    _last = { float(1 << _curpower), float(1 << _curpower) };

    ++_curpower;
    _cur.y = 0;
    _cur.x = (_cur.y < _last.y) ? 1 + _last.x : 1;
    _nexty = 1;
  }

  FG_Vec invdpiscale = { Backend::BASE_DPI / dpi.x, Backend::BASE_DPI / dpi.y };
  g.uv               = { _cur.x, _cur.y, (_cur.x + width), (_cur.y + height) };
  g.advance          = (_face->glyph->advance.x * FT_COEF * invdpiscale.x);
  g.bearing.x        = (_face->glyph->metrics.horiBearingX * FT_COEF * invdpiscale.x);
  g.bearing.y        = (_face->glyph->metrics.horiBearingY * FT_COEF * invdpiscale.y);
  g.width            = (float)width;
  g.height           = (float)height;

  _cur.x += width + 1; // one pixel buffer
  if(_nexty < _cur.y + height)
    _nexty = _cur.y + height + 1; // one pixel buffer

  return &g;
}

Glyph* Font::RenderGlyph(Context* context, char32_t codepoint)
{
  auto iter = kh_get_glyphmap(_glyphs, codepoint);
  if(iter < kh_end(_glyphs) && kh_exist(_glyphs, iter) && context->CheckGlyph(codepoint))
    return &kh_val(_glyphs, iter);

  auto g = LoadGlyph(codepoint);
  if(!g)
    return nullptr;

  // if this throws an error, remove it as a possible renderable codepoint
  if(FT_Load_Char(_face, codepoint, FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT | _ftaa(aa)) != 0)
  {
    (*_backend->_log)(_backend->_root, FG_Level_ERROR, "codepoint %i in %s failed to load.", codepoint,
                      _path.u8string().c_str());
    return nullptr;
  }

  FT_Bitmap& gbmp = _face->glyph->bitmap;
  uint32_t width  = (gbmp.pixel_mode == FT_PIXEL_MODE_LCD) ? (gbmp.width / 3) : gbmp.width;

  // Get the texture from our context, creating it if necessary
  GLuint tex = context->GetFontTexture(this);
  _enforceantialias(_ftaa(aa));
  std::unique_ptr<uint8_t[]> buf(new uint8_t[width * 4 * gbmp.rows]);

  switch(gbmp.pixel_mode) // Now we render the glyph to our next available buffer
  {
  case FT_PIXEL_MODE_LCD:
    for(uint32_t i = 0; i < gbmp.rows; ++i)
    {
      uint8_t* src = gbmp.buffer + (i * gbmp.pitch);
      uint8_t* dst = buf.get() + (4 * width * i);
      for(uint32_t j = 0; j < g->width; ++j) // RGBA
      {
        *dst++ = src[0];
        *dst++ = src[1];
        *dst++ = src[2];
        *dst++ = 0xFF;
        src += 3;
      }
    }
    break;
  case FT_PIXEL_MODE_GRAY:
    for(uint32_t i = 0; i < gbmp.rows; ++i)
    {
      uint8_t* src = gbmp.buffer + (i * gbmp.pitch);
      uint8_t* dst = buf.get() + (4 * width * i);
      for(uint32_t j = 0; j < gbmp.width; ++j) // RGBA
      {
        uint8_t v = *src++;
        // v         = (uint8_t)(powf(v / 255.0f, (1.0f / 2.2f)) * 255.0f);
        *dst++ = v; // premultiply alpha
        *dst++ = v;
        *dst++ = v;
        *dst++ = v;
      }
    }
    break;
  case FT_PIXEL_MODE_MONO:
    for(uint32_t i = 0; i < gbmp.rows; ++i)
    {
      uint8_t* src  = gbmp.buffer + (i * gbmp.pitch);
      uint32_t* dst = reinterpret_cast<uint32_t*>(buf.get() + (4 * width * i));

      for(uint32_t j = 0; j < gbmp.width; ++j)
        dst[j] = (src[j / 8] & (0x80 >> (j & 7))) ? 0xFFFFFFFF : 0x00000000;
    }
    break;
  default: return nullptr;
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  _backend->LogError("glBindTexture");
  glTexSubImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(g->uv.left), static_cast<GLint>(g->uv.top), width, gbmp.rows,
                  GL_RGBA, GL_UNSIGNED_BYTE, buf.get());
  _backend->LogError("glTexSubImage2D");
  glBindTexture(GL_TEXTURE_2D, 0);
  _backend->LogError("glBindTexture");

  context->AddGlyph(codepoint);
  return &kh_val(_glyphs, iter);
}

float Font::GetKerning(char32_t prev, char32_t cur)
{
  if(!_haskerning)
    return 0.0f;

  FT_Vector kerning;
  FT_Get_Kerning(_face, prev, cur, FT_KERNING_DEFAULT, &kerning);
  return kerning.x * (1.0f / 64.0f); // this would return .y for vertical layouts
}

FG_Vec Font::CalcTextDim(const char32_t* text, const FG_Vec& maxdim, float curlineheight, float letterspacing,
                         FG_BreakStyle breakstyle)
{
  FG_Vec dest       = { 0, 0 };
  bool dobreak      = false;
  char32_t last     = 0;
  float lastadvance = 0;
  FG_Rect box       = { 0, 0, 0, 0 };
  FG_Vec cursor     = { 0, !lineheight ? 0 : ((curlineheight / lineheight) * _ascender) };

  float width = 0.0f;
  while(*text != 0)
  {
    _getchar(text++, maxdim.x, breakstyle, curlineheight, letterspacing, cursor, box, last, lastadvance, dobreak);
    if(box.right > dest.x)
      dest.x = box.right;
  }
  dest.y = cursor.y - _descender;
  return dest;
}

float Font::GetLineWidth(const char32_t*& text, float maxwidth, FG_BreakStyle breakstyle, float letterspacing)
{
  bool dobreak      = false;
  char32_t last     = 0;
  float lastadvance = 0;
  FG_Rect box       = { 0, 0, 0, 0 };
  FG_Vec cursor     = { 0, 0 };
  float width       = 0.0f;
  while(*text != 0 && !dobreak)
    _getchar(text++, maxwidth, breakstyle, 0.0f, letterspacing, cursor, box, last, lastadvance, dobreak);
  return box.right;
}

bool Font::_isspace(int c) // We have to make our own isspace implementation because the standard isspace() explodes if
                           // you feed it unicode characters.
{
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

Glyph* Font::_getchar(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle, float curlineheight,
                      float letterspacing, FG_Vec& cursor, FG_Rect& box, char32_t& last, float& lastadvance, bool& dobreak)
{
  cursor.x += lastadvance;
  char32_t c = *text;
  auto iter  = kh_get_glyphmap(_glyphs, c);
  Glyph* g;
  if((iter >= kh_end(_glyphs) || !kh_exist(_glyphs, iter)) && (c != '\n' && c != '\r'))
  {
    g = LoadGlyph(c);
    if(!g)
    {
      lastadvance = 0;
      last        = c;
      return 0; // Note: Bad glyphs usually just have 0 width, so we don't have to check for them.
    }
  }
  else
    g = &kh_val(_glyphs, iter);

  float advance   = 0.0f;
  box.topleft     = cursor;
  box.bottomright = cursor;

  if(c != '\n' && c != '\r')
  {
    advance = g->advance + letterspacing + GetKerning(last, c);
    box.left += g->bearing.x;
    box.top -= g->bearing.y;
    box.right  = box.left + g->width;
    box.bottom = box.top + g->height;
  }

  dobreak = c == '\n';
  if(!dobreak && (breakstyle != FG_BreakStyle_NONE) && maxwidth >= 0.0f && box.right > maxwidth)
    dobreak = true;
  if(!dobreak && (breakstyle == FG_BreakStyle_WORD) && _isspace(last) && !_isspace(c) && maxwidth >= 0.0f)
  {
    float right         = cursor.x + advance;
    const char32_t* cur = ++text; // we can increment cur by one, because if our current character had been over the end, it
                                  // would have been handled above.
    while(*cur != 0 && !isspace(*cur))
    {
      auto i = kh_get_glyphmap(_glyphs, c);
      if(i >= kh_end(_glyphs) || !kh_exist(_glyphs, i))
      {
        Glyph* gword = &kh_val(_glyphs, i);
        if(right + gword->bearing.x + gword->width > maxwidth)
        {
          dobreak = true;
          break;
        }
        // cur[-1] is safe here because we incremented cur before entering this loop.
        right += gword->advance + letterspacing + GetKerning(cur[-1], cur[0]);
      }
      ++cur;
    }
  }

  if(dobreak)
  {
    box.left -= cursor.x;
    box.right -= cursor.x;
    box.top += curlineheight;
    box.bottom += curlineheight;
    cursor.x = 0;
    cursor.y += curlineheight;
  }

  lastadvance = advance;
  last        = c;
  return g;
}
std::pair<size_t, FG_Vec> Font::GetIndex(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle,
                                         float curlineheight, float letterspacing, FG_Vec pos)
{
  std::pair<size_t, FG_Vec> cache = { 0, { 0, 0 } };
  if(!text)
    return cache;
  bool dobreak      = false;
  char32_t last     = 0;
  float lastadvance = 0;
  FG_Rect box       = { 0, 0, 0, 0 };
  Glyph* g          = 0;
  for(cache.first = 0; text[cache.first] != 0; ++cache.first)
  {
    std::pair<size_t, FG_Vec> lastcache = cache;
    lastcache.second.x += lastadvance;
    g = _getchar(text + cache.first, maxwidth, breakstyle, curlineheight, letterspacing, cache.second, box, last,
                 lastadvance, dobreak);
    if(pos.y <= cache.second.y + curlineheight && pos.x < cache.second.x + (g ? g->bearing.x + (g->width * 0.5f) : 0.0f))
      return cache;             // we immediately terminate and return, WITHOUT adding the lastadvance on.
    if(pos.y <= cache.second.y) // Too far! return our previous cache.
      return lastcache;
  }
  cache.second.x +=
    lastadvance; // We have to add the lastadvance on here because we left the loop at the end of the string.
  return cache;
}
std::pair<size_t, FG_Vec> Font::GetPos(const char32_t* text, float maxwidth, FG_BreakStyle breakstyle, float curlineheight,
                                       float letterspacing, size_t index)
{
  std::pair<size_t, FG_Vec> cache = { 0, { 0, 0 } };
  if(!text)
    return cache;
  bool dobreak      = false;
  char32_t last     = 0;
  float lastadvance = 0;
  FG_Rect box       = { 0, 0, 0, 0 };
  for(cache.first = 0; cache.first < index && text[cache.first] != 0; ++cache.first)
    _getchar(text + cache.first, maxwidth, breakstyle, curlineheight, letterspacing, cache.second, box, last, lastadvance,
             dobreak);
  cache.second.x += lastadvance;
  return cache;
}
