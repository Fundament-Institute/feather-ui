// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/skin.h"
#include "feather/root.h"
#include "feather/layout/default.h"
#include "util.h"
#include <string>

__KHASH_IMPL(field, extern "C", unsigned int, fgCalculation, 1, kh_int_hash_func, kh_int_hash_equal);
__KHASH_IMPL(outline, extern "C", const char*, struct FG__SKIN_NODE*, 1, kh_str_hash_funcins, kh_str_hash_equal);
__KHASH_IMPL(unresolved, extern "C", const char*, fgCalculation, 1, kh_str_hash_funcins, kh_str_hash_equal);

fgSkinNode* fgGetSkin(fgSkin* skin, char* selector)
{
  char* type = strrchr(selector, '.');

  while(selector)
  {
    *type = '.';
    khiter_t i = kh_get_outline(skin->generators, selector);
    if(kh_exist2(skin->generators, i))
      return kh_val(skin->generators, i);

    *type = 0;
    i = kh_get_outline(skin->generators, selector);
    if(kh_exist2(skin->generators, i))
      return kh_val(skin->generators, i);

    selector = strchr(selector, '.');
    if(selector)
      ++selector;
  }

  return 0;
}

extern "C" void fgSkinResolveFields(kh_field_t* fields, kh_unresolved_t* unresolved, fgResolver resolver)
{
  for(khiter_t i = kh_begin(unresolved); i < kh_end(unresolved); ++i)
  {
    if(kh_exist(unresolved, i))
    {
      unsigned int field = resolver(0, 0, 0, kh_key(unresolved, i));
      if(field != ~0)
      {
        int r;
        khiter_t iter = kh_put_field(fields, field, &r);
        if(r > 0)
          kh_val(fields, iter) = kh_val(unresolved, i);
        kh_del_unresolved(unresolved, i); // This is safe to do while iterating because khash doesn't reallocate until things are added.
      }
    }
  }
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
      outline->layout = &fgLayoutDefault;
      outline->auxresolver = &fgLayoutDefaultResolver;
      outline->behavior = &fgDefaultBehavior;
      outline->stateresolver = &fgNullResolver;

      if(parent)
        LLAdd(outline, parent->children);

      outline->listeners = kh_init_event();
      if(node->unresolved) // Resolve strings to enumerators
        fgSkinResolveFields(node->fields, node->unresolved, &fgStandardResolver);

      (*outline->generator)(root, outline, 0, scale, dpi);

      for(khiter_t i = kh_begin(node->fields); i < kh_end(node->fields); ++i)
      {
        if(kh_exist(node->fields, i))
          fgGatherEvents(root, outline, kh_key(node->fields, i), kh_val(node->fields, i).nodes, kh_val(node->fields, i).count);
      }

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
  // The root node is just a list of child skin nodes to add to a display, for now
  for(fgSkinNode* cur = skin->root->children; cur; cur = cur->sibling)
    fgSkinGeneratorInner(root, skin, cur, cur->data, root->backend->displays[0], scale, dpi, selector);
  return root->backend->displays[0];
}