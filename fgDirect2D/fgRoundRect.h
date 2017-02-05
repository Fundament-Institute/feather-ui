// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "fgDirect2D.h"

#ifndef __FG_ROUNDRECT_D2D_H__
#define __FG_ROUNDRECT_D2D_H__

#include <d2d1_1.h>
#include <d2d1effectauthor.h>  
#include <d2d1effecthelpers.h>

DEFINE_GUID(CLSID_fgRoundRect, 0xC58204BE, 0x9FA1, 0x4884, 0xB1, 0x82, 0xBF, 0xE8, 0x5D, 0x65, 0x90, 0xDC);
DEFINE_GUID(CLSID_fgRoundRectPixelShader, 0x0050F29E, 0xF56A, 0xAC, 0xB2, 0x5F, 0x23, 0x90, 0x46, 0x94, 0xAA, 0x5F);

class fgRoundRect : public ID2D1EffectImpl, public ID2D1DrawTransform
{
public:
  IFACEMETHODIMP Initialize(_In_ ID2D1EffectContext* pContextInternal, _In_ ID2D1TransformGraph* pTransformGraph);
  IFACEMETHODIMP PrepareForRender(D2D1_CHANGE_TYPE changeType);
  IFACEMETHODIMP SetGraph(_In_ ID2D1TransformGraph* pGraph);
  static HRESULT Register(_In_ ID2D1Factory1* pFactory);
  static HRESULT __stdcall CreateEffect(_Outptr_ IUnknown** ppEffectImpl);
  IFACEMETHODIMP_(ULONG) AddRef();
  IFACEMETHODIMP_(ULONG) Release();
  IFACEMETHODIMP QueryInterface(_In_ REFIID riid, _Outptr_ void** ppOutput);

  HRESULT SetCorners(D2D_VECTOR_4F corners);
  D2D_VECTOR_4F GetCorners() const;

  HRESULT SetOutline(FLOAT outline);
  FLOAT GetOutline() const;

  HRESULT SetColor(UINT color);
  UINT GetColor() const;

  HRESULT SetOutlineColor(UINT color);
  UINT GetOutlineColor() const;

  IFACEMETHODIMP SetDrawInfo(_In_ ID2D1DrawInfo* pRenderInfo);
  IFACEMETHODIMP MapOutputRectToInputRects(_In_ const D2D1_RECT_L* pOutputRect,_Out_writes_(inputRectCount) D2D1_RECT_L* pInputRects,UINT32 inputRectCount) const;
  IFACEMETHODIMP MapInputRectsToOutputRect(_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputRects,_In_reads_(inputRectCount) CONST D2D1_RECT_L* pInputOpaqueSubRects,UINT32 inputRectCount,_Out_ D2D1_RECT_L* pOutputRect,_Out_ D2D1_RECT_L* pOutputOpaqueSubRect);
  IFACEMETHODIMP MapInvalidRect(UINT32 inputIndex,D2D1_RECT_L invalidInputRect,_Out_ D2D1_RECT_L* pInvalidOutputRect) const;
  IFACEMETHODIMP_(UINT32) GetInputCount() const;

private:
  fgRoundRect();
  HRESULT UpdateConstants();
  
  ID2D1DrawInfo* _drawInfo;
  ID2D1EffectContext* _effectContext;
  D2D1_RECT_L _inputRect;
  LONG _ref;

  struct
  {
    D2D_VECTOR_4F corners;
    float outline;
    UINT color;
    UINT outlinecolor;
  } _constants;
};

#endif