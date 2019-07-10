// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#include "feather/root.h"
#include <stdarg.h>
#include <string>
#include <assert.h>

extern "C" {
  __KHASH_IMPL(component, , const char*, fgBehaviorDefinition, 1, kh_str_hash_funcins, kh_str_hash_equal);
  __KHASH_IMPL(layout, , const char*, fgLayoutDefinition, 1, kh_str_hash_funcins, kh_str_hash_equal);
  __KHASH_IMPL(data, , const void*, fgDataListener, 1, kh_ptr_hash_func, kh_int_hash_equal);
}

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
    (*kh_val(root->listeners, iter).f)(root, kh_val(root->listeners, iter).obj, obj, offset, count);
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
}

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
