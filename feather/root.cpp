// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/root.h"
#include "feather/component/box.h"
#include "feather/component/text.h"
#include "feather/component/window.h"
#include "feather/layout/default.h"
#include "util.h"
#include <stdarg.h>
#include <string>
#include <assert.h>

__KHASH_IMPL(component, extern "C", const char*, fgBehaviorDefinition, 1, kh_str_hash_funcins, kh_str_hash_equal);
__KHASH_IMPL(layout, extern "C", const char*, fgLayoutDefinition, 1, kh_str_hash_funcins, kh_str_hash_equal);
__KHASH_IMPL(data, extern "C", const void*, fgDataHook*, 1, kh_ptr_hash_func, kh_int_hash_equal);

extern "C" void fgLog(fgRoot* root, FG_LOGLEVEL level, const char* format, ...)
{
  static const char* LEVELS[] = { "FATAL: ", "ERROR: ", "WARNING: ", "NOTICE: ", "INFO: ", "DEBUG: " };
  if(level != -1 && level < (sizeof(LEVELS) / sizeof(const char*)))
    fprintf(root->log, LEVELS[level]);

  va_list vl;
  va_start(vl, format);
  if(level <= root->level)
    vfprintf(root->log, format, vl);
  va_end(vl);

  fwrite("\n", 1, 1, root->log);
  fflush(root->log);
}

extern "C" fgMessageResult fgInject(fgRoot* root, struct FG__DOCUMENT_NODE* node, fgMessage* msg)
{
  switch(msg->type)
  {
  case FG_MSG_MOUSE_DOWN:
  case FG_MSG_MOUSE_DBLCLICK:
  case FG_MSG_MOUSE_UP:
  case FG_MSG_MOUSE_ON:
  case FG_MSG_MOUSE_OFF:
  case FG_MSG_MOUSE_MOVE:
  case FG_MSG_MOUSE_SCROLL:
  case FG_MSG_TOUCH_BEGIN:
  case FG_MSG_TOUCH_MOVE:
  case FG_MSG_TOUCH_END:
    // If this is a pointer event, iterate through the r-tree to find the most precise hit, then propagate upwards each time it's rejected.


  case FG_MSG_KEY_UP:
  case FG_MSG_KEY_DOWN:
  case FG_MSG_KEY_CHAR:
  case FG_MSG_JOY_BUTTON_DOWN:
  case FG_MSG_JOY_BUTTON_UP:
  case FG_MSG_JOY_AXIS:
    // If this is a key event, send it directly to the node that currently holds key focus, and propagate through the node parents as it's rejected.
    node = root->keyfocus; // fall through because processing is the same
  case FG_MSG_CUSTOM:
  case FG_MSG_GOT_FOCUS:
  case FG_MSG_LOST_FOCUS:
  case FG_MSG_DRAG_DROP:
  {
    fgError err = -1;
    // For these events, send to the given node and propagate up through the parents as it's rejected.
    while(node != nullptr && (err = fgSendMessage(root, node, msg).error) != 0)
      node = node->outline->parent->node;
    return fgMessageResult{ err };
  }
  default:
    break;
  }

  assert(node != nullptr);
  // For all other events, only send it to the given node, do not propagate.
  return fgSendMessage(root, node, msg);
}

extern "C" bool fgGetKey(fgRoot* root, unsigned char key)
{
  return root->keys[key / (sizeof(int) * 8)] & (1 << (key % (sizeof(int) * 8)));
}

extern "C" fgDataField fgGetData(struct FG__ROOT* root, void* data, const char* accessor)
{
  if(!accessor || !accessor[0])
    return fgDataField{ 0 };

  if(accessor[0] != '.')
    data = root->data;
  else
    ++accessor;

  // TODO: properly drill down properties
  return (*root->getfield)(data, accessor);
}

extern "C" void fgUpdateData(struct FG__ROOT* root, const void* obj)
{
  fgUpdateDataRange(root, obj, 0, 0);
}

extern "C" void fgUpdateDataRange(struct FG__ROOT* root, const void* obj, unsigned int offset, unsigned int count)
{
  khiter_t iter = kh_get_data(root->listeners, obj);
  if(iter < kh_end(root->listeners) && kh_exist(root->listeners, iter))
  {
    fgDataHook* cur = kh_val(root->listeners, iter);
    while(cur)
    {
      (*cur->f)(root, cur->self, obj, offset, count);
      cur = cur->next;
    }
  }
}

extern "C" void fgAddComponent(fgRoot* root, const char* name, fgBehaviorFunction fn, fgResolver resolver, unsigned int sz)
{
  int r;
  khiter_t i = kh_put_component(root->components, name, &r);
  if(r > 0)
    kh_val(root->components, i) = { sz, fn, resolver };
}

extern "C" void fgAddLayout(fgRoot* root, const char* name, fgLayout fn, fgResolver resolver, unsigned int sz)
{
  int r;
  khiter_t i = kh_put_layout(root->layouts, name, &r);
  if(r > 0)
    kh_val(root->layouts, i) = { sz, fn, resolver };
}

extern "C" void fgInitialize(fgRoot* root)
{
  *root = {};
  root->log = stdout;
  root->level = FGLOG_WARNING;
  root->operators = kh_init_function();
  root->layouts = kh_init_layout();
  root->components = kh_init_component();
  root->listeners = kh_init_data();

  fgAddComponent(root, "box", &fgBoxBehavior, &fgBoxResolver, sizeof(fgBoxData));
  fgAddComponent(root, "text", &fgTextBehavior, &fgTextResolver, sizeof(fgTextData));
  fgAddComponent(root, "window", &fgWindowBehavior, &fgWindowResolver, sizeof(fgWindowData));
  fgAddLayout(root, "default", &fgLayoutDefault, &fgLayoutDefaultResolver, 0);
}

FG_COMPILER_DLLEXPORT void fgLayoutDefault(fgDocumentNode*, const fgRect* area, const fgOutlineNode* parent, float scale, fgVec dpi);
FG_COMPILER_DLLEXPORT unsigned int fgLayoutDefaultResolver(void* outline, unsigned int index, fgCalcNode* out, const char* id);

extern "C" void fgTerminate(fgRoot* root)
{
  (*root->backend->destroy)(root->backend);

  if(root->operators)
    kh_destroy_function(root->operators);
  if(root->layouts)
    kh_destroy_layout(root->layouts);
  if(root->components)
    kh_destroy_component(root->components);
  if(root->listeners)
    kh_destroy_data(root->listeners);

  if(root->pointers)
    free(root->pointers);
  *root = {};
}

extern "C" fgError fgProcessMessages(fgRoot* root)
{
  return (*root->backend->processMessages)(root, root->backend);
}

extern "C" fgDataHook* fgAddDataHook(struct FG__ROOT* root, const void* data, void* obj, fgDataListener f, void (*remove)(struct FG__DATA_HOOK*), unsigned int aux)
{
  int r;
  auto ptr = (fgDataHook*)calloc(1, sizeof(fgDataHook) + aux);
  if(!ptr)
    return 0;
  *ptr = { data, (!obj && aux > 0) ? (ptr + 1) : obj, f, remove };

  khiter_t iter = kh_put_data(root->listeners, data, &r);
  if(r > 0)
    kh_val(root->listeners, iter) = 0;

  LLAdd(ptr, kh_val(root->listeners, iter));
  return ptr;
}

extern "C" void fgRemoveDataHook(struct FG__ROOT* root, fgDataHook* hook)
{
  khiter_t iter = kh_get_data(root->listeners, hook->data);
  if(iter < kh_end(root->listeners) && kh_exist(root->listeners, iter))
  {
    LLRemove(hook, kh_value(root->listeners, iter));
    if(!kh_value(root->listeners, iter))
      kh_del_data(root->listeners, iter);
  }
  if(hook->remove)
    (*hook->remove)(hook);
  free(hook);
}

extern "C" void fgRemoveData(struct FG__ROOT* root, void* data)
{
  khiter_t iter = kh_get_data(root->listeners, data);
  if(iter < kh_end(root->listeners) && kh_exist(root->listeners, iter))
  {
    fgDataHook* cur;
    fgDataHook* next = kh_value(root->listeners, iter);
    while(cur = next)
    {
      next = cur->next;
      if(cur->remove)
        (*cur->remove)(cur);
      free(cur);
    }
  }
}
