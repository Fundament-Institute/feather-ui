// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_LAYOUT_H__
#define __FG_LAYOUT_H__

#include "fgSkin.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct _FG_CLASS_LAYOUT;
typedef fgDeclareVector(struct _FG_CLASS_LAYOUT, ClassLayout) fgVectorClassLayout;
struct _FG_KEY_VALUE { const char* key; const char* value; };
typedef fgDeclareVector(struct _FG_KEY_VALUE, KeyValue) fgVectorKeyValue;

typedef struct _FG_CLASS_LAYOUT {
  fgSkinElement layout;
  const char* name;
  const char* id;
  fgVectorClassLayout children; // Type: fgClassLayout
  fgVectorKeyValue userdata; // Custom userdata from unknown attributes

#ifdef  __cplusplus
  FG_DLLEXPORT void AddUserString(const char* key, const char* value);
  FG_DLLEXPORT FG_UINT AddChild(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
  FG_DLLEXPORT char RemoveChild(FG_UINT child);
  FG_DLLEXPORT struct _FG_CLASS_LAYOUT* GetChild(FG_UINT child) const;
#endif
} fgClassLayout;

typedef struct _FG_LAYOUT {
  fgSkinBase base;
  fgStyle style;
  fgVectorClassLayout layout; // Type: fgClassLayout

#ifdef  __cplusplus
  FG_DLLEXPORT FG_UINT AddLayout(const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
  FG_DLLEXPORT char RemoveLayout(FG_UINT layout);
  FG_DLLEXPORT fgClassLayout* GetLayout(FG_UINT layout) const;

  FG_DLLEXPORT void LoadFileUBJSON(const char* file);
  FG_DLLEXPORT void LoadUBJSON(const char* data, FG_UINT length);
  FG_DLLEXPORT void SaveFileUBJSON(const char* file);
  FG_DLLEXPORT char LoadFileXML(const char* file);
  FG_DLLEXPORT char LoadXML(const char* data, FG_UINT length);
  FG_DLLEXPORT void SaveFileXML(const char* file);
#endif
} fgLayout;

FG_EXTERN void fgLayout_Init(fgLayout* self);
FG_EXTERN void fgLayout_Destroy(fgLayout* self);
FG_EXTERN FG_UINT fgLayout_AddLayout(fgLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN char fgLayout_RemoveLayout(fgLayout* self, FG_UINT layout);
FG_EXTERN fgClassLayout* fgLayout_GetLayout(const fgLayout* self, FG_UINT layout);

FG_EXTERN void fgLayout_LoadFileUBJSON(fgLayout* self, const char* file);
FG_EXTERN void fgLayout_LoadUBJSON(fgLayout* self, const char* data, FG_UINT length);
FG_EXTERN void fgLayout_SaveFileUBJSON(fgLayout* self, const char* file);
FG_EXTERN char fgLayout_LoadFileXML(fgLayout* self, const char* file);
FG_EXTERN char fgLayout_LoadXML(fgLayout* self, const char* data, FG_UINT length);
FG_EXTERN void fgLayout_SaveFileXML(fgLayout* self, const char* file);
FG_EXTERN void fgLayout_SaveElementXML(fgElement* root, const char* file);

FG_EXTERN void fgClassLayout_Init(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN void fgClassLayout_Destroy(fgClassLayout* self);
FG_EXTERN void fgClassLayout_AddUserString(fgClassLayout* self, const char* key, const char* value);
FG_EXTERN FG_UINT fgClassLayout_AddChild(fgClassLayout* self, const char* type, const char* name, fgFlag flags, const fgTransform* transform, short units, int order);
FG_EXTERN char fgClassLayout_RemoveChild(fgClassLayout* self, FG_UINT child);
FG_EXTERN fgClassLayout* fgClassLayout_GetChild(const fgClassLayout* self, FG_UINT child);

FG_EXTERN size_t fgDefaultLayout(fgElement* self, const FG_Msg* msg, AbsVec* dim);
FG_EXTERN size_t fgDistributeLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* dim);
FG_EXTERN size_t fgTileLayout(fgElement* self, const FG_Msg* msg, fgFlag flags, AbsVec* area);

#ifdef  __cplusplus
}
#endif

#endif