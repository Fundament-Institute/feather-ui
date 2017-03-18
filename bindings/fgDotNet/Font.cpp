#include "Font.h"
#include "fgRoot.h"
#include "fgText.h"

using namespace fgDotNet;
using namespace System;

Font::Font(void* p) : _p(p), _owner(false) {}
Font::Font(void* p, bool owner) : _p(p), _owner(owner) {}
Font::Font(fgFlag flags, String^ font, short weight, char italic, unsigned int size, System::Drawing::Point dpi) : _p(0), _owner(true)
{
  TOCHAR(font);
  _p = fgSingleton()->backend.fgCreateFont(flags, (const char*)pstr, weight, italic, size, &Element::From(dpi));
}
Font::~Font() { this->!Font(); }
Font::!Font() { if(_p && _owner) fgSingleton()->backend.fgDestroyFont(_p); }
Font^ Font::Clone(Desc^ desc) { return gcnew Font(fgSingleton()->backend.fgCloneFont(_p, &desc->operator fgFontDesc()), true); }

template<typename R, typename... Args>
R fontProcessString(fgFont font, String^ text, R(*F)(fgFont, const void*, size_t, Args...), Args... args)
{
  switch(fgSingleton()->backend.BackendTextFormat)
  {
  default:
  case FGTEXTFMT_UTF8:
  {
    cli::array<unsigned char>^ a = System::Text::Encoding::UTF8->GetBytes(text);
    pin_ptr<const unsigned char> pin = &a[0];
    return F(font, pin, a->Length, args...);
  }
  case FGTEXTFMT_UTF16:
  {
    cli::array<wchar_t>^ a = text->ToCharArray();
    pin_ptr<const wchar_t> pin = &a[0];
    return F(font, pin, a->Length, args...);
  }
  case FGTEXTFMT_UTF32:
  {
    cli::array<unsigned char>^ a = System::Text::Encoding::UTF32->GetBytes(text);
    pin_ptr<const unsigned char> pin = &a[0];
    return F(font, pin, a->Length / 4, args...);
  }
  }
}

void Font::Draw(String^ text, float lineheight, float letterspacing, unsigned int color, System::Drawing::RectangleF area, FABS rotation, System::Drawing::PointF center, fgFlag flags, DrawAuxData^ data, Font::TextLayout layout)
{
  fontProcessString<void, float, float, unsigned int, const AbsRect*, FABS, const AbsVec*, fgFlag, const fgDrawAuxData*, void*>(
    _p, text, fgSingleton()->backend.fgDrawFont, lineheight, letterspacing, color, &Element::From(area), rotation, &Element::From(center), flags, &data->operator fgDrawAuxData(), layout.p);
}
Font::TextLayout Font::GetLayout(String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF^ area, fgFlag flags, Font::TextLayout prevlayout)
{
  AbsRect absarea = Element::From(area);
  switch(fgSingleton()->backend.BackendTextFormat)
  {
  case FGTEXTFMT_UTF8:
  case FGTEXTFMT_UTF16:
  case FGTEXTFMT_UTF32:
  {
    void* layout = fontProcessString<void*, float, float, AbsRect*, fgFlag, void*>(
      _p, text, fgSingleton()->backend.fgFontLayout, lineheight, letterspacing, &absarea, flags, prevlayout.p);
    area->X = absarea.left;
    area->Y = absarea.top;
    area->Width = absarea.right - absarea.left;
    area->Height = absarea.bottom - absarea.top;
    return Font::TextLayout(layout);
  }
  }
  return Font::TextLayout(nullptr);
}
Font::Desc^ Font::GetDesc() {
  _FG_FONT_DESC desc;
  fgSingleton()->backend.fgFontGet(_p, &desc);
  return gcnew Font::Desc(desc);
}

size_t Font::Index(System::String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF area, fgFlag flags, System::Drawing::PointF pos, System::Drawing::PointF^ cursor, TextLayout layout)
{
  AbsVec abscur = Element::From(cursor);
  size_t r = fontProcessString<size_t, float, float, const AbsRect*, fgFlag, AbsVec, AbsVec*, void*>(
    _p, text, fgSingleton()->backend.fgFontIndex, lineheight, letterspacing, &Element::From(area), flags, Element::From(pos), &abscur, layout.p);
  cursor->X = abscur.x;
  cursor->Y = abscur.y;
  return r;
}
System::Drawing::PointF Font::Pos(System::String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF area, fgFlag flags, size_t index, TextLayout layout)
{
  AbsVec v = fontProcessString<AbsVec, float, float, const AbsRect*, fgFlag, size_t, void*>(
    _p, text, fgSingleton()->backend.fgFontPos, lineheight, letterspacing, &Element::From(area), flags, index, layout.p);
  return System::Drawing::PointF(v.x, v.y);
}