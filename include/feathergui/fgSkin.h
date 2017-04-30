// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_SKIN_H__
#define __FG_SKIN_H__

#include "fgSkinTree.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct kh_fgSkins_s;
struct kh_fgStyles_s;
struct kh_fgAssets_s;
struct kh_fgFonts_s;
struct _FG_SKIN;

struct _FG_FONT_DATA {
  fgFont font;
  fgFlag flags;
  const char* family;
  short weight;
  char italic;
  unsigned int size;

#ifdef  __cplusplus
  _FG_FONT_DATA() {}
  _FG_FONT_DATA(void* font, fgFlag flags, const char* family, short weight, char italic, unsigned int size);
  _FG_FONT_DATA(const _FG_FONT_DATA&);
  _FG_FONT_DATA(_FG_FONT_DATA&&);
  ~_FG_FONT_DATA();

  _FG_FONT_DATA& operator=(const _FG_FONT_DATA&);
  _FG_FONT_DATA& operator=(_FG_FONT_DATA&&);
#endif
};

struct _FG_ASSET_DATA {
  fgAsset asset;
  const char* file;
  void* data; // not supported yet

#ifdef  __cplusplus
  _FG_ASSET_DATA() {}
  _FG_ASSET_DATA(void* asset, const char* file);
  _FG_ASSET_DATA(const _FG_ASSET_DATA&);
  _FG_ASSET_DATA(_FG_ASSET_DATA&&);
  ~_FG_ASSET_DATA();

  _FG_ASSET_DATA& operator=(const _FG_ASSET_DATA&);
  _FG_ASSET_DATA& operator=(_FG_ASSET_DATA&&);
#endif
};

typedef struct _FG_SKIN_BASE
{
  struct kh_fgAssets_s* assets;
  struct kh_fgFonts_s* fonts;
  struct kh_fgSkins_s* skinmap;
  struct _FG_SKIN_BASE* parent;

#ifdef  __cplusplus
  FG_DLLEXPORT struct _FG_SKIN* AddSkin(const char* name);
  FG_DLLEXPORT char RemoveSkin(const char* name);
  FG_DLLEXPORT struct _FG_SKIN* GetSkin(const char* name) const;
  FG_DLLEXPORT struct _FG_ASSET_DATA* AddAssetFile(fgFlag flags, const char* file);
  FG_DLLEXPORT char RemoveAsset(fgAsset asset);
  FG_DLLEXPORT struct _FG_ASSET_DATA* GetAsset(fgAsset asset) const;
  FG_DLLEXPORT struct _FG_FONT_DATA* AddFont(fgFlag flags, const char* family, short weight, char italic, unsigned int size);
  FG_DLLEXPORT char RemoveFont(fgFont font);
  FG_DLLEXPORT struct _FG_FONT_DATA* GetFont(fgFont font) const;

  FG_DLLEXPORT struct _FG_SKIN* LoadFileUBJSON(const char* file);
  FG_DLLEXPORT struct _FG_SKIN* LoadUBJSON(const void* data, FG_UINT length);
  FG_DLLEXPORT struct _FG_SKIN* LoadFileXML(const char* file);
  FG_DLLEXPORT struct _FG_SKIN* LoadXML(const char* data, FG_UINT length);
#endif
} fgSkinBase;

typedef struct _FG_SKIN
{
  struct _FG_SKIN_BASE base;
  struct _FG_SKIN* inherit;
  fgStyle style; // style overrides
  fgSkinTree tree;
  const char* name;

#ifdef  __cplusplus
  FG_DLLEXPORT struct _FG_SKIN* GetSkin(const char* name) const;
#endif
} fgSkin;

FG_EXTERN void fgSkin_Init(fgSkin* self, const char* name);
FG_EXTERN void fgSkin_Destroy(fgSkin* self);
FG_EXTERN fgSkin* fgSkin_GetSkin(const fgSkin* self, const char* name);
FG_EXTERN char* fgSkin_ParseFontFamily(char* s, char quote, char** context);

FG_EXTERN void fgSkinBase_Destroy(fgSkinBase* self);
FG_EXTERN fgSkin* fgSkinBase_AddSkin(fgSkinBase* self, const char* name);
FG_EXTERN char fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name);
FG_EXTERN fgSkin* fgSkinBase_GetSkin(const fgSkinBase* self, const char* name);
FG_EXTERN struct _FG_ASSET_DATA* fgSkinBase_AddAssetFile(fgSkinBase* self, fgFlag flags, const char* file);
FG_EXTERN char fgSkinBase_RemoveAsset(fgSkinBase* self, fgAsset asset);
FG_EXTERN struct _FG_ASSET_DATA* fgSkinBase_GetAsset(const fgSkinBase* self, fgAsset asset);
FG_EXTERN struct _FG_FONT_DATA* fgSkinBase_AddFont(fgSkinBase* self, fgFlag flags, const char* family, short weight, char italic, unsigned int size);
FG_EXTERN char fgSkinBase_RemoveFont(fgSkinBase* self, fgFont font);
FG_EXTERN struct _FG_FONT_DATA* fgSkinBase_GetFont(const fgSkinBase* self, fgFont font);
FG_EXTERN fgFlag fgSkinBase_ParseFlagsFromString(const char* s, fgFlag* remove, int divider);

FG_EXTERN fgSkin* fgSkinBase_LoadFileUBJSON(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* fgSkinBase_LoadUBJSON(fgSkinBase* self, const void* data, FG_UINT length);
FG_EXTERN fgSkin* fgSkinBase_LoadFileXML(fgSkinBase* self, const char* file);
FG_EXTERN fgSkin* fgSkinBase_LoadXML(fgSkinBase* self, const char* data, FG_UINT length);

#ifdef  __cplusplus
}
#endif

#endif