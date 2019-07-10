// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/outline.h"
#include "feather/skin.h"
#include "feather/root.h"
#include "util.h"
#include <assert.h>
#include <math.h>
#include <type_traits>
#include <vector>
#include <string>

extern "C" {
  __KHASH_IMPL(event, , ptrdiff_t, fgEvent*, 1, kh_int_hash_func, kh_int_hash_equal);
}

template<typename T>
T fgResolveType(const fgCalculation& calc, void* data, fgRoot* root, fgDocumentNode* node, float scale, float dpi)
{
  auto v = fgEvaluate(root, calc.nodes, calc.count, data, node, node->outline->fontSize, node->outline->lineHeight, dpi, scale);
  if constexpr(std::is_arithmetic<T>::value)
  {
    switch(v.type)
    {
    case FG_CALC_FLOAT:
      if constexpr(!std::is_floating_point<T>::value)
        assert(false);
      return (T)v.value.d;
    case FG_CALC_INT:
      if constexpr(!std::is_integral<T>::value)
        assert(false);
      return (T)v.value.i;
    }
  }
  else
  {
    if(v.type == FG_CALC_STRING)
      return v.value.s;
  }

  assert(false);
  if constexpr(std::is_same<T, float>::value)
    return ::nanf("");
  else if constexpr(std::is_floating_point<T>::value)
    return ::nan("");
  else
    return 0;
}

template<typename T>
T fgResolveField(T zero, kh_field_t* fields, const char* id, void* data, fgRoot* root, fgDocumentNode* node, float scale, float dpi)
{
  khiter_t iter = kh_get_field(fields, id);
  return (kh_exist(fields, iter) && iter <= kh_end(fields)) ? fgResolveType<T>(kh_val(fields, iter), data, root, node, scale, dpi) : zero;
}

void fgResolveVecData(fgVec& v, kh_field_t* fields, const char* x, const char* y, void* data, fgRoot* root, fgDocumentNode* node, float scale, fgVec dpi)
{
  khiter_t xiter = kh_get_field(fields, x);
  khiter_t yiter = kh_get_field(fields, y);

  if(kh_exist(fields, xiter) && xiter <= kh_end(fields))
    v.x = fgResolveType<float>(kh_val(fields, xiter), data, root, node, scale, dpi.x);
  if(kh_exist(fields, yiter) && yiter <= kh_end(fields))
    v.y = fgResolveType<float>(kh_val(fields, yiter), data, root, node, scale, dpi.y);
}

void fgEvalCustomFields(void* data, fgResolver resolver, fgRoot* root, kh_field_t* fields, fgDocumentNode* node, float scale, float dpi)
{
  fgCalcResult result; // Evaluate all fields that might exist in our generator
  int n;
  (*resolver)(data, 0, &result, &n);
  auto names = (const char**)result.p;

  for(int i = 0; i < n; ++i)
  {
    khiter_t iter = kh_get_field(fields, names[i]);
    if(kh_exist(fields, iter) && iter <= kh_end(fields))
    {
      auto calc = kh_val(fields, iter);
      auto result = fgEvaluate(root, calc.nodes, calc.count, data, node, node->outline->fontSize, node->outline->lineHeight, dpi, scale);
      (*resolver)(data, names[i], &result.value, &result.type);
    }
  }
}

extern "C" void fgDestroyOutlineBehavior(fgRoot* root, fgDocumentNode* document, fgOutlineNode* node)
{
  if(document)
  {
    fgMessage msg = { FG_MSG_DESTROY };
    if(node->behavior)
      fgSendMessage(root, document, &msg);
    if(document->state)
      free(document->state);
  }

  if(node->statedata)
    free(node->statedata);
}

extern "C" void fgDefaultOutlineGenerator(fgRoot* root, fgOutlineNode* node, float scale, fgVec dpi)
{
  auto fields = reinterpret_cast<fgSkinNode*>(node->gendata)->fields;
  auto data = node->data;
  auto document = node->node;

  if(!document)
  {
    document = fgAllocDocumentNode();
    document->outline = node;
    //document->tstate = kh_init_transition();
  }

  {
    node->fontSize = !node->parent ? 0 : node->parent->fontSize;
    node->lineHeight = !node->parent ? 0 : node->parent->lineHeight;

    node->fontSize = fgResolveField<float>(0, fields, "fontsize", data, root, document, scale, dpi.x);
    node->lineHeight = fgResolveField<float>(0, fields, "lineheight", data, root, document, scale, dpi.y);
    node->zindex = fgResolveField<int>(0, fields, "zindex", data, root, document, scale, dpi.x);
    node->layoutFlags = fgResolveField<fgFlag>(0, fields, "flags", data, root, document, 0, 0);

    const char* behavior = fgResolveField<const char*>(nullptr, fields, "behavior", data, root, document, 0, 0);
    if(behavior)
    {
      khiter_t iter = kh_get_component(root->components, behavior);
      if(iter <= kh_end(root->components) && kh_exist(root->components, iter))
      {
        auto def = kh_val(root->components, iter);
        if(def.fn != node->behavior)
        {
          // Check existing behavior and destroy if it exists
          fgDestroyOutlineBehavior(root, document, node);

          // Create new behavior and allocate state
          node->behavior = def.fn;
          node->stateresolver = def.resolver;
          fgMessage msg = { FG_MSG_CONSTRUCT };
          auto sz = fgSendMessage(root, document, &msg).construct.size;
          if(sz > 0)
            document->state = malloc(sz);

          msg.construct.ptr = document->state;
          fgSendMessage(root, document, &msg);
          
          if(def.sz > 0)
          {
            node->statedata = malloc(def.sz);
            fgEvalCustomFields(node->statedata, node->stateresolver, root, fields, document, scale, dpi.x);
          }
        }
      }
    }

    const char* layout = fgResolveField<const char*>(nullptr, fields, "layout", data, root, document, 0, 0);
    if(layout)
    {
      khiter_t iter = kh_get_layout(root->layouts, layout);
      if(iter <= kh_end(root->layouts) && kh_exist(root->layouts, iter))
      {
        auto def = kh_val(root->layouts, iter);
        if(def.fn != node->layout)
        {
          if(node->auxdata)
            free(node->auxdata);
          
          FG__DOCUMENT_NODE* children = 0;
          fgMessage msg = { FG_MSG_REMOVE_CHILD };
          msg.child.dpi = dpi;
          msg.child.scale = scale;

          if(node->layout) // Remove all children from the layout, which should delete the corresponding R-tree
          {
            for(fgOutlineNode* n = node->children; n; n = n->next)
            {
              fgDocumentNode* next = n->node;
              while(msg.child.node = next)
              {
                next = msg.child.node->sibling;
                fgSendMessage(root, document, &msg);
                msg.child.node->sibling = children;
                children = msg.child.node;
              }
            }
          }

          // Create new behavior and allocate state
          node->layout = def.fn;
          node->auxresolver = def.resolver;

          if(def.sz > 0)
          {
            node->auxdata = malloc(def.sz);
            fgEvalCustomFields(node->auxdata, node->auxresolver, root, fields, document, scale, dpi.x);
          }

          // Re-add all children using new layout function
          msg.type = FG_MSG_ADD_CHILD;
          while(msg.child.node = children)
          {
            children = children->sibling;
            fgSendMessage(root, document, &msg);
          }
        }
      }
    }
  }

  fgResolveVecData(node->area.abs.topleft, fields, "abs_left", "abs_top", data, root, document, scale, dpi);
  fgResolveVecData(node->area.abs.bottomright, fields, "abs_right", "abs_bottom", data, root, document, scale, dpi);
  fgResolveVecData(node->area.rel.topleft, fields, "rel_left", "rel_top", data, root, document, scale, dpi);
  fgResolveVecData(node->area.rel.bottomright, fields, "rel_right", "rel_bottom", data, root, document, scale, dpi);
  fgResolveVecData(node->center.abs, fields, "c_x", "c_y", data, root, document, scale, dpi);
  fgResolveVecData(node->center.rel, fields, "c_rel_x", "c_rel_y", data, root, document, scale, dpi);
  fgResolveVecData(node->margin.topleft, fields, "margin_left", "margin_top", data, root, document, scale, dpi);
  fgResolveVecData(node->margin.bottomright, fields, "margin_right", "margin_bottom", data, root, document, scale, dpi);
  fgResolveVecData(node->padding.topleft, fields, "padding_left", "padding_top", data, root, document, scale, dpi);
  fgResolveVecData(node->padding.bottomright, fields, "padding_right", "padding_bottom", data, root, document, scale, dpi);
  fgResolveVecData(node->min, fields, "min_x", "min_y", data, root, document, scale, dpi);
  fgResolveVecData(node->max, fields, "max_x", "max_y", data, root, document, scale, dpi);

  if(!node->node)
  {
    node->node = document;
    fgMessage msg = { FG_MSG_ADD_CHILD };
    msg.child.node = node->node;
    msg.child.dpi = dpi;
    msg.child.scale = scale;
    fgSendMessage(root, node->parent->node, &msg); // This will call the layout function, which in turn will trigger a resize if necessary
  }
}

extern "C" void fgDestroyOutlineNode(struct FG__ROOT* root, fgOutlineNode* node, float scale, fgVec dpi)
{
  fgDestroyOutlineBehavior(root, node->node, node);

  if(node->node)
    fgDestroyDocumentNode(root, node->node, scale, dpi);

  if(node->parent)
    LLRemove(node, node->parent->children);

  //if(node->gendata)
  //  free(node->gendata);

  if(node->auxdata)
    free(node->auxdata);

  // Remove all dangling listeners
  if(node->listeners)
  {
    for(khiter_t i = kh_begin(node->listeners); i < kh_end(node->listeners); ++i)
    {
      if(kh_exist(node->listeners, i))
      {
        fgEvent* cur;
        fgEvent* next = kh_val(node->listeners, i);
        while(cur = next)
        {
          fgRemoveEvent(cur);
          next = cur->next;
          fgDestroyEvent(cur);
        }
      }
    }

    kh_destroy_event(node->listeners);
  }
}
