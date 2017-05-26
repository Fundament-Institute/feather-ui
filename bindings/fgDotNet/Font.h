// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgBackend.h"
#include "Element.h"
#include "fgText.h"

namespace fgDotNet {
  public ref class Font
  {
  public:
    ref struct Desc
    {
      Desc(fgFontDesc& desc) : ascender(desc.ascender), descender(desc.descender), lineheight(desc.lineheight), pt(desc.pt), dpi(desc.dpi.x, desc.dpi.y) {}
      FABS ascender;
      FABS descender;
      FABS lineheight;
      unsigned int pt;
      System::Drawing::PointF dpi;

      operator fgFontDesc() { return fgFontDesc{ ascender, descender, lineheight, pt,{ dpi.X, dpi.Y } }; }
      static operator fgFontDesc(Desc^ d) { return d->operator fgFontDesc(); }
    };

    value struct TextLayout {
      TextLayout(void* _p) : p(_p) {}
      void* p;
    };

    explicit Font(void* p);
    Font(fgFlag flags, System::String^ font, short weight, char italic, unsigned int size, System::Drawing::PointF dpi);
    ~Font();
    !Font();
    Font^ Clone(Desc^ desc);
    void Draw(System::String^ text, float lineheight, float letterspacing, unsigned int color, System::Drawing::RectangleF area, FABS rotation, System::Drawing::PointF center, fgFlag flags, DrawAuxData^ data, TextLayout layout);
    TextLayout GetLayout(System::String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF^ area, fgFlag flags, System::Drawing::PointF dpi, TextLayout prevlayout);
    Desc^ GetDesc();
    size_t Index(System::String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF area, fgFlag flags, System::Drawing::PointF pos, System::Drawing::PointF^ cursor, System::Drawing::PointF dpi, TextLayout layout);
    System::Drawing::PointF Pos(System::String^ text, float lineheight, float letterspacing, System::Drawing::RectangleF area, fgFlag flags, size_t index, System::Drawing::PointF dpi, TextLayout layout);

    inline static operator void*(Font^ r) { return r->_p; }

  private:
    Font(void* p, bool owner);
    void* _p;
    bool _owner;
  };
}