// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_TRIANGLE_D2D_H__
#define __FG_TRIANGLE_D2D_H__

#include "fgEffectBase.h"

DEFINE_GUID(CLSID_fgTriangle, 0xEAB57D83, 0x9556, 0x4CD2, 0xA0, 0x66, 0xE4, 0xF1, 0x60, 0x3E, 0x90, 0xC7);
DEFINE_GUID(CLSID_fgTrianglePixelShader, 0x692EB775, 0x3EC4, 0x4B55, 0x93, 0xB6, 0x44, 0x78, 0x07, 0x6D, 0xE0, 0xD3);

class fgTriangle : public fgEffectBase
{
public:
  IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
  static HRESULT Register(_In_ ID2D1Factory1* pFactory);
  static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);

  IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

private:
  fgTriangle();
  ~fgTriangle();
};

#endif