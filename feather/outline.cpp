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
#include <unordered_map>
#include <algorithm>

typedef std::pair<fgOutlineNode*, unsigned int> fgDataPair;

inline khint_t data_pair_hash(const fgDataHook* t)
{
  return kh_int64_hash_func(static_cast<uint64_t>(reinterpret_cast<fgDataPair*>(t->self)->second) | (static_cast<uint64_t>(kh_ptr_hash_func(t->data)) << 32));
}

inline khint_t data_pair_hash(const fgDataHook* l, const fgDataHook* r)
{
  return reinterpret_cast<fgDataPair*>(l->self)->second == reinterpret_cast<fgDataPair*>(r->self)->second && l->data == r->data;
}

__KHASH_IMPL(datapair, extern "C", fgDataHook*, unsigned int, 1, data_pair_hash, data_pair_hash);

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
T fgResolveField(T zero, kh_field_t* fields, unsigned int field, void* data, fgRoot* root, fgDocumentNode* node, float scale, float dpi)
{
  khiter_t iter = kh_get_field(fields, field);
  return kh_exist2(fields, iter) ? fgResolveType<T>(kh_val(fields, iter), data, root, node, scale, dpi) : zero;
}

template<class T>
void fgOutlineSetEvent(struct FG__ROOT* root, fgOutlineNode* node, T& target, T source, unsigned int field, float scale, fgVec dpi)
{
  if(target != source)
  {
    target = source;
    fgSendEvent(root, node->listeners, field, node, &target);
  }
}

void fgResolveVecData(fgVec& v, kh_field_t* fields, unsigned int x, unsigned int y, void* data, fgRoot* root, fgDocumentNode* node, float scale, fgVec dpi)
{
  khiter_t xiter = kh_get_field(fields, x);
  khiter_t yiter = kh_get_field(fields, y);

  if(kh_exist2(fields, xiter))
    fgOutlineSetEvent<float>(root, node->outline, v.x, fgResolveType<float>(kh_val(fields, xiter), data, root, node, scale, dpi.x), x, scale, dpi);
  if(kh_exist2(fields, yiter))
    fgOutlineSetEvent<float>(root, node->outline, v.y, fgResolveType<float>(kh_val(fields, yiter), data, root, node, scale, dpi.y), y, scale, dpi);
}

void fgEvalCustomFields(void* data, fgResolver resolver, fgRoot* root, kh_field_t* fields, fgDocumentNode* node, float scale, float dpi)
{
  fgCalcNode result; // Evaluate all fields that might exist in our generator
  int n = (*resolver)(0, 0, &result, 0);
  auto names = (unsigned int*)result.value.p;

  for(int i = 0; i < n; ++i)
  {
    khiter_t iter = kh_get_field(fields, names[i]);
    if(kh_exist2(fields, iter))
    {
      auto calc = kh_val(fields, iter);
      auto result = fgEvaluate(root, calc.nodes, calc.count, data, node, node->outline->fontSize, node->outline->lineHeight, dpi, scale);
      (*resolver)(data, names[i], &result, 0);
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

extern "C" void fgDefaultOutlineGenerator(fgRoot* root, fgOutlineNode* node, unsigned int field, float scale, fgVec dpi)
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
    if(!field || field == FG_EVENT_FONT_SIZE)
    {
      node->fontSize = !node->parent ? 0 : node->parent->fontSize;
      fgOutlineSetEvent(root, node, node->fontSize, fgResolveField<float>(0, fields, FG_EVENT_FONT_SIZE, data, root, document, scale, dpi.x), FG_EVENT_FONT_SIZE, scale, dpi);
    }

    if(!field || field == FG_EVENT_LINE_HEIGHT)
    {
      node->lineHeight = !node->parent ? 0 : node->parent->lineHeight;
      fgOutlineSetEvent(root, node, node->lineHeight, fgResolveField<float>(0, fields, FG_EVENT_LINE_HEIGHT, data, root, document, scale, dpi.y), FG_EVENT_LINE_HEIGHT, scale, dpi);
    }
    if(!field || field == FG_EVENT_Z_INDEX)
      fgOutlineSetEvent(root, node, node->zindex, fgResolveField<int>(0, fields, FG_EVENT_Z_INDEX, data, root, document, scale, dpi.x), FG_EVENT_Z_INDEX, scale, dpi);
    if(!field || field == FG_EVENT_LAYOUT_FLAGS)
      fgOutlineSetEvent(root, node, node->layoutFlags, fgResolveField<fgFlag>(0, fields, FG_EVENT_LAYOUT_FLAGS, data, root, document, 0, 0), FG_EVENT_LAYOUT_FLAGS, scale, dpi);

    if(!field || field == FG_EVENT_BEHAVIOR)
    {
      const char* behavior = fgResolveField<const char*>(nullptr, fields, FG_EVENT_BEHAVIOR, data, root, document, 0, 0);
      if(behavior)
      {
        khiter_t iter = kh_get_component(root->components, behavior);
        if(kh_exist2(root->components, iter))
        {
          auto def = kh_val(root->components, iter);
          if(def.fn != node->behavior)
          {
            // Check existing behavior and destroy if it exists
            fgDestroyOutlineBehavior(root, document, node);

            // Resolve custom fields to IDs
            fgSkinResolveFields(fields, reinterpret_cast<fgSkinNode*>(node->gendata)->unresolved, def.resolver);

            // Create new behavior and allocate state
            node->behavior = def.fn;
            node->stateresolver = def.resolver;
            fgMessage msg = { FG_MSG_CONSTRUCT };
            auto sz = fgSendMessage(root, document, &msg).construct.size;
            if(sz > 0)
              document->state = calloc(1, sz);

            msg.construct.ptr = document->state;
            fgSendMessage(root, document, &msg);

            if(def.sz > 0)
            {
              node->statedata = calloc(1, def.sz);
              fgEvalCustomFields(node->statedata, node->stateresolver, root, fields, document, scale, dpi.x);
            }
            fgSendEvent(root, node->listeners, FG_EVENT_BEHAVIOR, node, &node->behavior);
          }
        }
      }
    }

    if(!field || field == FG_EVENT_LAYOUT)
    {
      const char* layout = fgResolveField<const char*>(nullptr, fields, FG_EVENT_LAYOUT, data, root, document, 0, 0);
      if(layout)
      {
        khiter_t iter = kh_get_layout(root->layouts, layout);
        if(kh_exist2(root->layouts, iter))
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

            // Resolve custom fields to IDs
            fgSkinResolveFields(fields, reinterpret_cast<fgSkinNode*>(node->gendata)->unresolved, def.resolver);

            // Create new behavior and allocate state
            node->layout = def.fn;
            node->auxresolver = def.resolver;

            if(def.sz > 0)
            {
              node->auxdata = calloc(1, def.sz);
              fgEvalCustomFields(node->auxdata, node->auxresolver, root, fields, document, scale, dpi.x);
            }
            fgSendEvent(root, node->listeners, FG_EVENT_LAYOUT, node, &node->layout);

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
  }

  if(!field || field == FG_EVENT_LEFT_ABS || field == FG_EVENT_TOP_ABS)
    fgResolveVecData(node->area.abs.topleft, fields, FG_EVENT_LEFT_ABS, FG_EVENT_TOP_ABS, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_RIGHT_ABS || field == FG_EVENT_BOTTOM_ABS)
    fgResolveVecData(node->area.abs.bottomright, fields, FG_EVENT_RIGHT_ABS, FG_EVENT_BOTTOM_ABS, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_LEFT_REL || field == FG_EVENT_TOP_REL)
    fgResolveVecData(node->area.rel.topleft, fields, FG_EVENT_LEFT_REL, FG_EVENT_TOP_REL, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_RIGHT_REL || field == FG_EVENT_BOTTOM_REL)
    fgResolveVecData(node->area.rel.bottomright, fields, FG_EVENT_RIGHT_REL, FG_EVENT_BOTTOM_REL, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_CENTER_X_ABS || field == FG_EVENT_CENTER_Y_ABS)
    fgResolveVecData(node->center.abs, fields, FG_EVENT_CENTER_X_ABS, FG_EVENT_CENTER_Y_ABS, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_CENTER_X_REL || field == FG_EVENT_CENTER_Y_REL)
    fgResolveVecData(node->center.rel, fields, FG_EVENT_CENTER_X_REL, FG_EVENT_CENTER_Y_REL, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_MARGIN_LEFT || field == FG_EVENT_MARGIN_TOP)
    fgResolveVecData(node->margin.topleft, fields, FG_EVENT_MARGIN_LEFT, FG_EVENT_MARGIN_TOP, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_MARGIN_RIGHT || field == FG_EVENT_MARGIN_BOTTOM)
    fgResolveVecData(node->margin.bottomright, fields, FG_EVENT_MARGIN_RIGHT, FG_EVENT_MARGIN_BOTTOM, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_PADDING_LEFT || field == FG_EVENT_PADDING_TOP)
    fgResolveVecData(node->padding.topleft, fields, FG_EVENT_PADDING_LEFT, FG_EVENT_PADDING_TOP, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_PADDING_RIGHT || field == FG_EVENT_PADDING_BOTTOM)
    fgResolveVecData(node->padding.bottomright, fields, FG_EVENT_PADDING_RIGHT, FG_EVENT_PADDING_BOTTOM, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_MIN_WIDTH || field == FG_EVENT_MIN_HEIGHT)
    fgResolveVecData(node->min, fields, FG_EVENT_MIN_WIDTH, FG_EVENT_MIN_HEIGHT, data, root, document, scale, dpi);
  if(!field || field == FG_EVENT_MAX_WIDTH || field == FG_EVENT_MAX_HEIGHT)
    fgResolveVecData(node->max, fields, FG_EVENT_MAX_WIDTH, FG_EVENT_MAX_HEIGHT, data, root, document, scale, dpi);

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
          next = cur->next;
          fgDestroyEvent(root, cur);
        }
      }
    }

    kh_destroy_event(node->listeners);
  }

  // Remove all dangling data hooks
  if(node->datahooks)
  {
    for(khiter_t i = kh_begin(node->datahooks); i < kh_end(node->datahooks); ++i)
    {
      if(kh_exist(node->datahooks, i))
        fgRemoveDataHook(root, kh_key(node->datahooks, i));
    }

    kh_destroy_datapair(node->datahooks);
  }
}

static const std::unordered_map<std::string, unsigned int> standard_event_map = {
  { "left-abs", FG_EVENT_LEFT_ABS },
  { "top-abs", FG_EVENT_TOP_ABS },
  { "right-abs", FG_EVENT_RIGHT_ABS },
  { "bottom-abs", FG_EVENT_BOTTOM_ABS },
  { "width-abs", FG_EVENT_WIDTH_ABS },
  { "height-abs", FG_EVENT_HEIGHT_ABS },
  { "left-rel", FG_EVENT_LEFT_REL },
  { "top-rel", FG_EVENT_TOP_REL },
  { "right-rel", FG_EVENT_RIGHT_REL },
  { "bottom-rel", FG_EVENT_BOTTOM_REL },
  { "width-rel", FG_EVENT_WIDTH_REL },
  { "height-rel", FG_EVENT_HEIGHT_REL },
  { "center-x-abs", FG_EVENT_CENTER_X_ABS },
  { "center-y-abs", FG_EVENT_CENTER_Y_ABS },
  { "center-x-rel", FG_EVENT_CENTER_X_REL },
  { "center-y-rel", FG_EVENT_CENTER_Y_REL },
  { "margin-left", FG_EVENT_MARGIN_LEFT },
  { "margin-top", FG_EVENT_MARGIN_TOP },
  { "margin-right", FG_EVENT_MARGIN_RIGHT },
  { "margin-bottom", FG_EVENT_MARGIN_BOTTOM },
  { "padding-left", FG_EVENT_PADDING_LEFT },
  { "padding-top", FG_EVENT_PADDING_TOP },
  { "padding-right", FG_EVENT_PADDING_RIGHT },
  { "padding-bottom", FG_EVENT_PADDING_BOTTOM },
  { "min-width", FG_EVENT_MIN_WIDTH },
  { "min-height", FG_EVENT_MIN_HEIGHT },
  { "max-width", FG_EVENT_MAX_WIDTH },
  { "max-height", FG_EVENT_MAX_HEIGHT },
  { "lineheight", FG_EVENT_LINE_HEIGHT },
  { "fontsize", FG_EVENT_FONT_SIZE },
  { "z-index", FG_EVENT_Z_INDEX },
  { "layout-flags", FG_EVENT_LAYOUT_FLAGS },
  { "layout", FG_EVENT_LAYOUT },
  { "behavior", FG_EVENT_BEHAVIOR },
  { "area-left", FG_EVENT_AREA_LEFT },
  { "area-top", FG_EVENT_AREA_TOP },
  { "area-right", FG_EVENT_AREA_RIGHT },
  { "area-bottom", FG_EVENT_AREA_BOTTOM },
  { "area-width", FG_EVENT_HEIGHT },
  { "area-height", FG_EVENT_WIDTH },
  { "extent-left", FG_EVENT_EXTENT_LEFT },
  { "extent-top", FG_EVENT_EXTENT_TOP },
  { "extent-right", FG_EVENT_EXTENT_RIGHT },
  { "extent-bottom", FG_EVENT_EXTENT_BOTTOM },
  { "state-flags", FG_EVENT_STATE_FLAGS },
};

static unsigned int standard_event_values[] = {
  FG_EVENT_LEFT_ABS,
  FG_EVENT_TOP_ABS,
  FG_EVENT_RIGHT_ABS,
  FG_EVENT_BOTTOM_ABS,
  FG_EVENT_WIDTH_ABS,
  FG_EVENT_HEIGHT_ABS,
  FG_EVENT_LEFT_REL,
  FG_EVENT_TOP_REL,
  FG_EVENT_RIGHT_REL,
  FG_EVENT_BOTTOM_REL,
  FG_EVENT_WIDTH_REL,
  FG_EVENT_HEIGHT_REL,
  FG_EVENT_CENTER_X_ABS,
  FG_EVENT_CENTER_Y_ABS,
  FG_EVENT_CENTER_X_REL,
  FG_EVENT_CENTER_Y_REL,
  FG_EVENT_MARGIN_LEFT,
  FG_EVENT_MARGIN_TOP,
  FG_EVENT_MARGIN_RIGHT,
  FG_EVENT_MARGIN_BOTTOM,
  FG_EVENT_PADDING_LEFT,
  FG_EVENT_PADDING_TOP,
  FG_EVENT_PADDING_RIGHT,
  FG_EVENT_PADDING_BOTTOM,
  FG_EVENT_MIN_WIDTH,
  FG_EVENT_MIN_HEIGHT,
  FG_EVENT_MAX_WIDTH,
  FG_EVENT_MAX_HEIGHT,
  FG_EVENT_LINE_HEIGHT,
  FG_EVENT_FONT_SIZE,
  FG_EVENT_Z_INDEX,
  FG_EVENT_LAYOUT_FLAGS,
  FG_EVENT_LAYOUT,
  FG_EVENT_BEHAVIOR,
  FG_EVENT_AREA_LEFT,
  FG_EVENT_AREA_TOP,
  FG_EVENT_AREA_RIGHT,
  FG_EVENT_AREA_BOTTOM,
  FG_EVENT_WIDTH,
  FG_EVENT_HEIGHT,
  FG_EVENT_EXTENT_LEFT,
  FG_EVENT_EXTENT_TOP,
  FG_EVENT_EXTENT_RIGHT,
  FG_EVENT_EXTENT_BOTTOM,
  FG_EVENT_STATE_FLAGS,
};

extern "C" unsigned int fgStandardResolver(void* outline, unsigned int index, fgCalcNode* out, const char* id)
{

  if(!id && !outline)
  {
    out->value.p = standard_event_values;
    return sizeof(standard_event_values) / sizeof(unsigned int);
  }
  if(id && !outline)
  {
    std::string raw(id);
    std::transform(raw.begin(), raw.end(), raw.begin(), ::tolower);

    if(auto i = standard_event_map.find(raw.c_str()); i != standard_event_map.end())
      return i->second;
    return ~0;
  }

  if(out->type == FG_CALC_NONE)
    return ~0; // You cannot set any of these values, only read them

  auto data = reinterpret_cast<struct FG__OUTLINE_NODE*>(outline);
  switch(index)
  {
  case FG_EVENT_LEFT_ABS:
    out->value.d = data->area.abs.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_TOP_ABS:
    out->value.d = data->area.abs.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_RIGHT_ABS:
    out->value.d = data->area.abs.right;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_BOTTOM_ABS:
    out->value.d = data->area.abs.bottom;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_WIDTH_ABS:
    out->value.d = (double)data->area.abs.right - data->area.abs.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_HEIGHT_ABS:
    out->value.d = (double)data->area.abs.bottom - data->area.abs.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_LEFT_REL:
    out->value.d = data->area.rel.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_TOP_REL:
    out->value.d = data->area.rel.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_RIGHT_REL:
    out->value.d = data->area.rel.right;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_BOTTOM_REL:
    out->value.d = data->area.rel.bottom;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_WIDTH_REL:
    out->value.d = (double)data->area.rel.right - data->area.rel.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_HEIGHT_REL:
    out->value.d = (double)data->area.rel.bottom - data->area.rel.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_CENTER_X_ABS:
    out->value.d = data->center.abs.x;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_CENTER_Y_ABS:
    out->value.d = data->center.abs.y;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_CENTER_X_REL:
    out->value.d = data->center.rel.x;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_CENTER_Y_REL:
    out->value.d = data->center.rel.y;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MARGIN_LEFT:
    out->value.d = data->margin.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MARGIN_TOP:
    out->value.d = data->margin.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MARGIN_RIGHT:
    out->value.d = data->margin.right;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MARGIN_BOTTOM:
    out->value.d = data->margin.bottom;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_PADDING_LEFT:
    out->value.d = data->padding.left;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_PADDING_TOP:
    out->value.d = data->padding.top;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_PADDING_RIGHT:
    out->value.d = data->padding.right;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_PADDING_BOTTOM:
    out->value.d = data->padding.bottom;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MIN_WIDTH:
    out->value.d = data->min.x;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MIN_HEIGHT:
    out->value.d = data->min.y;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MAX_WIDTH:
    out->value.d = data->max.x;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_MAX_HEIGHT:
    out->value.d = data->max.y;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_LINE_HEIGHT:
    out->value.d = data->fontSize;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_FONT_SIZE:
    out->value.d = data->lineHeight;
    return out->type = FG_CALC_FLOAT;
  case FG_EVENT_Z_INDEX:
    out->value.i = data->zindex;
    return out->type = FG_CALC_INT;
  case FG_EVENT_LAYOUT_FLAGS:
    out->value.i = data->layoutFlags;
    return out->type = FG_CALC_INT;
  case FG_EVENT_LAYOUT:
    out->value.fn = (fgDelegate)data->layout;
    return out->type = FG_CALC_FUNC;
  case FG_EVENT_BEHAVIOR:
    out->value.fn = (fgDelegate)data->behavior;
    return out->type = FG_CALC_FUNC;
  }

  return ~0;
}

template<class T>
inline fgEvent* fgAllocEventAux(fgListener fn, unsigned int srcfield, kh_event_t* source, unsigned int destfield, struct FG__EVENT** dest, const T& aux)
{
  fgEvent* p = fgAllocEvent(0, fn, srcfield, source, destfield, dest, sizeof(T));
  *reinterpret_cast<T*>(p->self) = aux;
  return p;
}

void fgStandardDataRemove(fgDataHook* hook)
{
  auto& data = *reinterpret_cast<fgDataPair*>(hook->self);
  khiter_t iter = kh_get_datapair(data.first->datahooks, hook);
  if(iter < kh_end(data.first->datahooks) && kh_exist(data.first->datahooks, iter))
    kh_del_datapair(data.first->datahooks, iter);
}

template<class T>
inline fgDataHook* fgAddDataHookAux(struct FG__ROOT* root, const void* data, fgDataListener f, void (*remove)(struct FG__DATA_HOOK*), const T& aux)
{
  fgDataHook* p = fgAddDataHook(root, data, 0, f, remove, sizeof(T));
  *reinterpret_cast<T*>(p->self) = aux;
  return p;
}

extern "C" void fgStandardListener(struct FG__ROOT* root, void* self, void* target, unsigned int field, void* pfield)
{
  auto outline = reinterpret_cast<fgOutlineNode*>(self);
  fgMessage msg = { FG_MSG_TEXT_SCALE };
  auto scale = fgSendMessage(root, outline->node, &msg).scale.scale;
  msg = { FG_MSG_DPI };
  auto dpi = fgSendMessage(root, outline->node, &msg).dpi.dpi;

  (*outline->generator)(root, outline, field, scale, *dpi);
}

extern "C" void fgStandardDataListener(struct FG__ROOT* root, void* self, const void* data, unsigned int start, unsigned int count)
{
  auto pair = *reinterpret_cast<std::pair<fgOutlineNode*, unsigned int>*>(self);
  auto outline = pair.first;
  fgMessage msg = { FG_MSG_TEXT_SCALE };
  auto scale = fgSendMessage(root, outline->node, &msg).scale.scale;
  msg = { FG_MSG_DPI };
  auto dpi = fgSendMessage(root, outline->node, &msg).dpi.dpi;

  (*outline->generator)(root, outline, pair.second, scale, *dpi);
}

extern "C" void fgGatherEvents(struct FG__ROOT* root, fgOutlineNode* node, unsigned int srcfield, fgCalcNode* nodes, unsigned int count)
{
  for(unsigned int i = 0; i < count; ++i)
  {
    if(nodes[i].type & FG_CALC_STATE)
    {
      fgOutlineNode* target = node;
      const char* id = nodes[i].value.s;
      while(id[0] == '^') // something that refers to ^^left-abs is referring to the left-abs value of the parent of the parent of this node.
      {
        target = target->parent;
        ++id;
      }
      assert(target);
      if(!target)
        continue;

      fgMessage msg = { FG_MSG_STATE_GET, 0 };
      msg.getState = { id, 0 };
      int field = fgSendMessage(root, target->node, &msg).error; // Note that we are inserting events in reverse here, so target and source are swapped.
      if(!fgCheckEventTuple(root, target->listeners, field, &node->node->events, srcfield))
        fgAllocEvent(node, &fgStandardListener, field, target->listeners, srcfield, &node->node->events, 0);
    }
    else if(nodes[i].type & FG_CALC_DATA)
    {
      auto d = fgGetData(root, node->data, nodes[i].value.s);

      fgDataPair pair = { node, srcfield };
      fgDataHook hook = { node->data, &pair };
      khiter_t iter = kh_get_datapair(node->datahooks, &hook);
      if(iter < kh_end(node->datahooks) && kh_exist(node->datahooks, iter))
        ++kh_val(node->datahooks, iter);
      else
      {
        fgDataHook* ptr = fgAddDataHookAux(root, d.obj, &fgStandardDataListener, &fgStandardDataRemove, fgDataPair{ node, srcfield });
        int r;
        iter = kh_put_datapair(node->datahooks, ptr, &r);
        if(r >= 0)
          kh_val(node->datahooks, iter) = 1;
      }
    }
  }
}

extern "C" bool fgCheckEventTuple(struct FG__ROOT* root, kh_event_t* source, unsigned int srcfield, struct FG__EVENT** dest, unsigned int destfield)
{
  int r;
  khiter_t iter = kh_put_eventnode(root->eventnodes, fgEventTuple{ source, dest, srcfield, destfield }, &r);
  assert(r >= 0);
  if(!r)
    kh_val(root->eventnodes, iter) = 1;
  else if(r > 0)
    ++kh_val(root->eventnodes, iter);

  return r > 0;
}

extern "C" unsigned int fgNullResolver(void*, unsigned int, fgCalcNode*, const char* n)
{
  return ~0;
}
