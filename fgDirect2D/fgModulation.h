// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_IMAGE_D2D_H__
#define __FG_IMAGE_D2D_H__

#include "fgImageEffectBase.h"

DEFINE_GUID(CLSID_fgModulation, 0xFADDB0B1, 0x3E4A, 0x44F0, 0x99, 0x73, 0xFA, 0xE2, 0x00, 0xA2, 0x28, 0x53);
DEFINE_GUID(CLSID_fgModulationPixelShader, 0x41B090DD, 0x2A5D, 0x482A, 0x84, 0xAE, 0x94, 0x96, 0x95, 0xBE, 0xAA, 0xF2);

class fgModulation : public fgImageEffectBase
{
public:
  IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
  static HRESULT Register(_In_ ID2D1Factory1* pFactory);
  static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);

  IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);

private:
  fgModulation();
  ~fgModulation();
};

#endif