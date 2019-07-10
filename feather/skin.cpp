// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/skin.h"
#include "feather/root.h"
#include "util.h"
#include <string>

extern "C" {
  __KHASH_IMPL(field, , const char*, fgCalculation, 1, kh_str_hash_funcins, kh_str_hash_equal);
  __KHASH_IMPL(outline, , const char*, struct FG__SKIN_NODE*, 1, kh_str_hash_funcins, kh_str_hash_equal);
}

fgSkinNode* fgGetSkin(fgSkin* skin, char* selector)
{
  char* type = strrchr(selector, '.');

  while(selector)
  {
    *type = '.';
    khiter_t i = kh_get_outline(skin->generators, selector);
    if(i <= kh_end(skin->generators) && kh_exist(skin->generators, i))
      return kh_val(skin->generators, i);

    *type = 0;
    i = kh_get_outline(skin->generators, selector);
    if(i <= kh_end(skin->generators) && kh_exist(skin->generators, i))
      return kh_val(skin->generators, i);

    selector = strchr(selector, '.');
    if(selector)
      ++selector;
  }

  return 0;
}


extern "C" fgOutlineNode* fgSkinGeneratorInner(struct FG__ROOT* root, fgSkin* skin, struct FG__SKIN_NODE* node, void* data, fgOutlineNode* parent, float scale, fgVec dpi, std::string& selector)
{
  size_t revert = selector.size();
  if(selector.size() > 0)
    selector += '.';
  selector += node->id;
  fgOutlineNode* outline = 0;

  if(!node->fields)
  {
    auto element = (*root->getfield)(data, 0);

    if(element.primitive == FG_DATA_ARRAY)
    {
      size_t prev = selector.size();
      int i = 0;
      fgDataField field;
      for(;;)
      {
        field = (*root->getindex)(element.obj, i);
        if(field.primitive == FG_DATA_NONE)
          break;

        selector += '.';
        selector += field.type;
        fgSkinNode* resolve = fgGetSkin(skin, selector.data());
        outline = !resolve ? 0 : fgSkinGeneratorInner(root, skin, resolve, field.obj, parent, scale, dpi, selector);
        selector.erase(prev, std::string::npos);
      }
    }
    else
    {
      selector += '.';
      selector += element.type;
      fgSkinNode* resolve = fgGetSkin(skin, selector.data());
      outline = !resolve ? 0 : fgSkinGeneratorInner(root, skin, resolve, element.obj, parent, scale, dpi, selector);
    }
  }
  else
  {
    outline = (fgOutlineNode*)calloc(1, sizeof(fgOutlineNode));
    if(outline)
    {
      outline->generator = &fgDefaultOutlineGenerator;
      outline->gendata = node;
      outline->data = data;
      outline->parent = parent;
      if(parent)
        LLAdd(outline, parent->children);

      outline->listeners = kh_init_event();
      (*outline->generator)(root, outline, scale, dpi);

      for(fgSkinNode* cur = node->children; cur; cur = cur->sibling)
        fgSkinGeneratorInner(root, skin, cur, cur->data, outline, scale, dpi, selector);
    }
  }

  selector.erase(revert, std::string::npos);
  return outline;
}

extern "C" fgOutlineNode* fgSkinGenerator(struct FG__ROOT* root, fgSkin* skin, float scale, fgVec dpi)
{
  std::string selector;
  return fgSkinGeneratorInner(root, skin, skin->root, skin->root->data, 0, scale, dpi, selector);
}