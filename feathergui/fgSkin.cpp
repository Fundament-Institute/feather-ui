// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "bss-util/khash.h"
#include "bss-util/XML.h"
#include "fgSkin.h"
#include "feathercpp.h"

using namespace bss;

KHASH_INIT(fgSkins, const char*, fgSkin*, 1, kh_str_hash_funcins, kh_str_hash_insequal);
KHASH_INIT(fgAssets, fgAsset, _FG_ASSET_DATA*, 1, kh_ptr_hash_func, kh_int_hash_equal);
KHASH_INIT(fgFonts, fgFont, _FG_FONT_DATA*, 1, kh_ptr_hash_func, kh_int_hash_equal);

static_assert(sizeof(fgSkinLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");
static_assert(sizeof(fgClassLayoutArray) == sizeof(fgVector), "mismatch between vector sizes");

_FG_FONT_DATA::_FG_FONT_DATA(void* _font, fgFlag _flags, const char* _family, short _weight, char _italic, unsigned int _size) : 
  font(!_font ? 0 : fgroot_instance->backend.fgCloneFont(_font, 0)), flags(_flags), family(fgCopyText(_family, __FILE__, __LINE__)), weight(_weight), italic(_italic), size(_size)
{}
_FG_FONT_DATA::_FG_FONT_DATA(const _FG_FONT_DATA& copy)
{
  memcpy(this, &copy, sizeof(_FG_FONT_DATA));
  font = fgroot_instance->backend.fgCloneFont(font, 0);
  family = fgCopyText(family, __FILE__, __LINE__);
}
_FG_FONT_DATA::_FG_FONT_DATA(_FG_FONT_DATA&& mov)
{
  memcpy(this, &mov, sizeof(_FG_FONT_DATA));
  bss::bssFill(mov, 0);
}
_FG_FONT_DATA::~_FG_FONT_DATA()
{
  if(font)
    fgroot_instance->backend.fgDestroyFont(font);
  if(family)
    fgFreeText(family, __FILE__, __LINE__);
}

_FG_FONT_DATA& _FG_FONT_DATA::operator=(const _FG_FONT_DATA& copy) { this->~_FG_FONT_DATA(); new (this) _FG_FONT_DATA(copy); return *this; }
_FG_FONT_DATA& _FG_FONT_DATA::operator=(_FG_FONT_DATA&& mov) { this->~_FG_FONT_DATA(); new (this) _FG_FONT_DATA(std::move(mov)); return *this; }

_FG_ASSET_DATA::_FG_ASSET_DATA(void* _asset, const char* _file)
{
  bss::bssFill(*this, 0);
  asset = _asset;
  file = fgCopyText(_file, __FILE__, __LINE__);
}
_FG_ASSET_DATA::_FG_ASSET_DATA(const _FG_ASSET_DATA& copy)
{
  memcpy(this, &copy, sizeof(_FG_ASSET_DATA));
  asset = fgroot_instance->backend.fgCloneAsset(asset, 0);
  file = fgCopyText(file, __FILE__, __LINE__);
}
_FG_ASSET_DATA::_FG_ASSET_DATA(_FG_ASSET_DATA&& mov)
{
  memcpy(this, &mov, sizeof(_FG_ASSET_DATA));
  bss::bssFill(mov, 0);
}
_FG_ASSET_DATA::~_FG_ASSET_DATA()
{
  if(asset)
    fgroot_instance->backend.fgDestroyAsset(asset);
  if(file)
    fgFreeText(file, __FILE__, __LINE__);
}

_FG_ASSET_DATA& _FG_ASSET_DATA::operator=(const _FG_ASSET_DATA& copy) { this->~_FG_ASSET_DATA(); new (this) _FG_ASSET_DATA(copy); return *this; }
_FG_ASSET_DATA& _FG_ASSET_DATA::operator=(_FG_ASSET_DATA&& mov) { this->~_FG_ASSET_DATA(); new (this) _FG_ASSET_DATA(std::move(mov)); return *this; }

void fgSkin_Init(fgSkin* self, const char* name)
{
  bss::bssFill(*self, 0);
  self->name = fgCopyText(name, __FILE__, __LINE__);
}

template<class HASH, class T, void (*DESTROY)(T*), void(*DEL)(HASH*, khint_t)>
char DestroyHashElement(HASH* self, khiter_t iter)
{
  if(kh_exist(self, iter))
  {
    (*DESTROY)(kh_val(self, iter));
    fgfree(kh_val(self, iter), __FILE__, __LINE__);
    (*DEL)(self, iter);
    return 1;
  }
  return 0;
}

template<class HASH, class T, void(*DESTROYELEM)(T*), void(*DEL)(HASH*, khint_t), void(*DESTROYHASH)(HASH*)>
void DestroyHash(HASH* self)
{
  if(self)
  {
    khiter_t cur = kh_begin(self);
    while(cur != kh_end(self)) DestroyHashElement<HASH, T, DESTROYELEM, DEL>(self, cur++);
    (*DESTROYHASH)(self);
  }
}
void fgSkin_Destroy(fgSkin* self)
{
  fgStyle_Destroy(&self->style);
  fgSkinTree_Destroy(&self->tree);
  fgSkinBase_Destroy(&self->base);
  if(self->name)
    fgFreeText(self->name, __FILE__, __LINE__);
}

fgSkin* fgSkin_GetSkin(const fgSkin* self, const char* name)
{
  if(!self || !name)
    return 0;
  fgSkin* r = fgSkinBase_GetSkin(&self->base, name);
  return !r ? fgSkin_GetSkin(self->inherit, name) : r;
}

void _FG_ASSET_DATA_DESTROY(_FG_ASSET_DATA* p) { p->~_FG_ASSET_DATA(); }
void _FG_FONT_DATA_DESTROY(_FG_FONT_DATA* p) { p->~_FG_FONT_DATA(); }
void fgSkinBase_Destroy(fgSkinBase* self)
{
  DestroyHash<kh_fgAssets_t, _FG_ASSET_DATA, &_FG_ASSET_DATA_DESTROY, &kh_del_fgAssets, &kh_destroy_fgAssets>(self->assets);
  DestroyHash<kh_fgFonts_t, _FG_FONT_DATA, &_FG_FONT_DATA_DESTROY, &kh_del_fgFonts, &kh_destroy_fgFonts>(self->fonts);
  DestroyHash<kh_fgSkins_t, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins, &kh_destroy_fgSkins>(self->skinmap);
}

_FG_ASSET_DATA* fgSkinBase_AddAssetFile(fgSkinBase* self, fgFlag flags, const char* file)
{
  if(!self->assets)
    self->assets = kh_init_fgAssets();

  fgAsset asset = fgroot_instance->backend.fgCreateAssetFile(flags, file);
  int r;
  khiter_t iter = kh_put_fgAssets(self->assets, asset, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    kh_val(self->assets, iter) = fgmalloc<_FG_ASSET_DATA>(1, __FILE__, __LINE__);
    new(kh_val(self->assets, iter)) _FG_ASSET_DATA(asset, file);
  }
  return kh_val(self->assets, iter);
}
char fgSkinBase_RemoveAsset(fgSkinBase* self, fgAsset asset)
{
  if(!self->assets || !asset)
    return 0;

  return DestroyHashElement<struct kh_fgAssets_s, _FG_ASSET_DATA, &_FG_ASSET_DATA_DESTROY, &kh_del_fgAssets>(self->assets, kh_get_fgAssets(self->assets, asset));
}
_FG_ASSET_DATA* fgSkinBase_GetAsset(const fgSkinBase* self, fgAsset asset)
{
  assert(self != 0);
  if(!asset || !self->assets)
    return 0;
  khiter_t iter = kh_get_fgAssets(self->assets, asset);
  return (iter != kh_end(self->assets) && kh_exist(self->assets, iter)) ? kh_val(self->assets, iter) : 0;
}
_FG_FONT_DATA* fgSkinBase_AddFont(fgSkinBase* self, fgFlag flags, const char* families, short weight, char italic, unsigned int size)
{
  size_t len = strlen(families) + 1;
  DYNARRAY(char, buf, len);
  memcpy(buf, families, len);

  char* context = 0;
  char* family = fgSkin_ParseFontFamily(buf, '\'', &context);
  fgFont font = 0;
  while(family != 0)
  {
    if((font = fgroot_instance->backend.fgCreateFont(flags, family, weight, italic, size, &fgroot_instance->dpi)) != 0)
      break;
    family = fgSkin_ParseFontFamily(0, '\'', &context);
  }
  if(!font)
    return 0;

  if(!self->fonts)
    self->fonts = kh_init_fgFonts();

  int r;
  khiter_t iter = kh_put_fgFonts(self->fonts, font, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    kh_val(self->fonts, iter) = fgmalloc<_FG_FONT_DATA>(1, __FILE__, __LINE__);
    new(kh_val(self->fonts, iter)) _FG_FONT_DATA(font, flags, families, weight, italic, size);
  }
  return kh_val(self->fonts, iter);
}
char fgSkinBase_RemoveFont(fgSkinBase* self, fgFont font)
{
  if(!self->fonts || !font)
    return 0;

  return DestroyHashElement<struct kh_fgFonts_s, _FG_FONT_DATA, &_FG_FONT_DATA_DESTROY, &kh_del_fgFonts>(self->fonts, kh_get_fgFonts(self->fonts, font));
}
_FG_FONT_DATA* fgSkinBase_GetFont(const fgSkinBase* self, fgFont font)
{
  assert(self != 0);
  if(!font || !self->fonts)
    return 0;
  khiter_t iter = kh_get_fgFonts(self->fonts, font);
  return (iter != kh_end(self->fonts) && kh_exist(self->fonts, iter)) ? kh_val(self->fonts, iter) : 0;
}
fgSkin* fgSkinBase_AddSkin(fgSkinBase* self, const char* name)
{
  if(!self->skinmap)
    self->skinmap = kh_init_fgSkins();

  if(!name) return 0;
  int r = 0;
  khiter_t iter = kh_put_fgSkins(self->skinmap, name, &r);
  if(r != 0) // If r is 0 the element already exists so we don't want to re-initialize it
  {
    kh_val(self->skinmap, iter) = fgmalloc<fgSkin>(1, __FILE__, __LINE__);
    fgSkin_Init(kh_val(self->skinmap, iter), name);
    kh_val(self->skinmap, iter)->base.parent = self;
    kh_key(self->skinmap, iter) = kh_val(self->skinmap, iter)->name;
  }

  return kh_val(self->skinmap, iter);
}
char fgSkinBase_RemoveSkin(fgSkinBase* self, const char* name)
{
  if(!self->skinmap || !name)
    return 0;

  return DestroyHashElement<struct kh_fgSkins_s, fgSkin, &fgSkin_Destroy, &kh_del_fgSkins>(self->skinmap, kh_get(fgSkins, self->skinmap, name));
}
fgSkin* fgSkinBase_GetSkin(const fgSkinBase* self, const char* name)
{
  if(!self || !name || !self->skinmap)
    return 0;
  khiter_t iter = kh_get(fgSkins, self->skinmap, name);
  return (iter != kh_end(self->skinmap) && kh_exist(self->skinmap, iter)) ? kh_val(self->skinmap, iter) : 0;
}

void fgSkinBase_IterateSkins(const fgSkinBase* self, void* p, void(*f)(void*, fgSkin*, const char*))
{
  if(!self->skinmap)
    return;
  for(khiter_t i = 0; i < kh_end(self->skinmap); ++i)
  {
    if(kh_exist(self->skinmap, i))
      f(p, kh_val(self->skinmap, i), kh_key(self->skinmap, i));
  }
}

fgSkin* fgSkinBase_LoadFileUBJSON(fgSkinBase* self, const char* file)
{
  return 0;
}
fgSkin* fgSkinBase_LoadUBJSON(fgSkinBase* self, const void* data, FG_UINT length)
{
  return 0;
}

void fgSkinBase_WriteXML(XMLNode* node, fgSkinBase* base, char toplevel)
{
  // Write any skins that were stored in the root
  if(base->skinmap)
  {
    for(khiter_t i = 0; i < kh_end(base->skinmap); ++i)
      if(kh_exist(base->skinmap, i))
        fgSkin_WriteXML(node->AddNode(toplevel ? "fg:Skin" : "Skin"), kh_val(base->skinmap, i));
  }
}

size_t fgSkinTree::AddChild(const char* type, fgFlag flags, const fgTransform* transform, short units, int order) { return fgSkinTree_AddChild(this, type, flags, transform, units, order); }
char fgSkinTree::RemoveChild(FG_UINT child) { return fgSkinTree_RemoveChild(this, child); }
fgSkinLayout* fgSkinTree::GetChild(FG_UINT child) const { return fgSkinTree_GetChild(this, child); }
FG_UINT fgSkinTree::AddStyle(const char* name) { return fgSkinTree_AddStyle(this, name); }
char fgSkinTree::RemoveStyle(FG_UINT style) { return fgSkinTree_RemoveStyle(this, style); }
fgStyle* fgSkinTree::GetStyle(FG_UINT style) const { return fgSkinTree_GetStyle(this, style); }

fgSkin* fgSkinBase::AddSkin(const char* name) { return fgSkinBase_AddSkin(this, name); }
char fgSkinBase::RemoveSkin(const char* name) { return fgSkinBase_RemoveSkin(this, name); }
fgSkin* fgSkinBase::GetSkin(const char* name) const { return fgSkinBase_GetSkin(this, name); }
_FG_ASSET_DATA* fgSkinBase::AddAssetFile(fgFlag flags, const char* file) { return fgSkinBase_AddAssetFile(this, flags, file); }
char fgSkinBase::RemoveAsset(fgAsset asset) { return fgSkinBase_RemoveAsset(this, asset); }
_FG_ASSET_DATA* fgSkinBase::GetAsset(fgAsset asset) const { return fgSkinBase_GetAsset(this, asset); }
_FG_FONT_DATA* fgSkinBase::AddFont(fgFlag flags, const char* family, short weight, char italic, unsigned int size) { return fgSkinBase_AddFont(this, flags, family, weight, italic, size); }
char fgSkinBase::RemoveFont(fgFont font) { return fgSkinBase_RemoveFont(this, font); }
_FG_FONT_DATA* fgSkinBase::GetFont(fgFont font) const { return fgSkinBase_GetFont(this, font); }

fgSkin* fgSkinBase::LoadFileUBJSON(const char* file) { return fgSkinBase_LoadFileUBJSON(this, file); }
fgSkin* fgSkinBase::LoadUBJSON(const void* data, FG_UINT length) { return fgSkinBase_LoadUBJSON(this, data, length); }
fgSkin* fgSkinBase::LoadFileXML(const char* file) { return fgSkinBase_LoadFileXML(this, file); }
fgSkin* fgSkinBase::LoadXML(const char* data, FG_UINT length) { return fgSkinBase_LoadXML(this, data, length); }

fgSkin* fgSkin::GetSkin(const char* name) const { return fgSkin_GetSkin(this, name); }