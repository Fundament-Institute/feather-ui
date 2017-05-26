#include "Asset.h"
#include "fgRoot.h"
#include "fgResource.h"

using namespace fgDotNet;
using namespace System;
using namespace System::Drawing;


Asset::Asset(void* p) : _p(p), _owner(false) {}
Asset::Asset(void* p, bool owner) : _p(p), _owner(owner) {}
Asset::Asset(fgFlag flags, System::String^ file, System::Drawing::PointF dpi) : _owner(true), _p(0)
{
  TOCHAR(file);
  fgSingleton()->backend.fgCreateAssetFile(flags, (const char*)pstr, &Element::From(dpi));
}
Asset::Asset(fgFlag flags, cli::array<System::Byte>^ data, size_t length, System::Drawing::PointF dpi) : _owner(true), _p(0)
{
  pin_ptr<const unsigned char> pstr = &data[0];
  _p = fgSingleton()->backend.fgCreateAsset(flags, (char*)pstr, data->Length, &Element::From(dpi));
}
Asset::~Asset() { this->!Asset(); }
Asset::!Asset() { if(_p && _owner) fgSingleton()->backend.fgDestroyAsset(_p); }
Asset^ Asset::Clone(Element^ src) { return gcnew Asset(fgSingleton()->backend.fgCloneAsset(_p, src), true); }
void Asset::Draw(UnifiedRect^ uv, unsigned int color, unsigned int edge, FABS outline, RectangleF^ area, FABS rotation, PointF^ center, fgFlag flags, DrawAuxData^ data)
{
  fgSingleton()->backend.fgDrawAsset(_p, &uv->operator CRect(), color, edge, outline, &Element::From(area), rotation, &Element::From(center), flags, &data->operator fgDrawAuxData());
}
PointF^ Asset::Size(UnifiedRect^ uv, fgFlag flags, System::Drawing::PointF dpi)
{
  AbsVec out = { 0,0 };
  fgSingleton()->backend.fgAssetSize(_p, &uv->operator CRect(), &out, flags, &Element::From(dpi));
  return PointF(out.x, out.y);
}