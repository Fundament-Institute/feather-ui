// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgBackend.h"
#include "Element.h"

namespace fgDotNet {
  public ref class Asset
  {
  public:
    explicit Asset(void* p);
    Asset(fgFlag flags, System::String^ file, System::Drawing::PointF dpi);
    Asset(fgFlag flags, cli::array<System::Byte>^ data, size_t length, System::Drawing::PointF dpi);
    ~Asset();
    !Asset();
    Asset^ Clone(Element^ src);
    void Draw(UnifiedRect^ uv, unsigned int color, unsigned int edge, FABS outline, System::Drawing::RectangleF^ area, FABS rotation, System::Drawing::PointF^ center, fgFlag flags, DrawAuxData^ data);
    System::Drawing::PointF^ Size(UnifiedRect^ uv, fgFlag flags, System::Drawing::PointF dpi);

    inline static operator void*(Asset^ r) { return r->_p; }

  private:
    Asset(void* p, bool owner);
    void* _p;
    bool _owner;
  };
}