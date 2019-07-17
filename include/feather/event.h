// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__EVENT_H
#define FG__EVENT_H

#include "message.h"
#include "khash.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct FG__ROOT;
struct FG__EVENT;
typedef void(*fgListener)(struct FG__ROOT*, void*, void*, unsigned int, void*);

KHASH_DECLARE(event, unsigned int, struct FG__EVENT*);

typedef struct {
  kh_event_t* source;
  struct FG__EVENT** dest;
  unsigned int srcfield; // field of source
  unsigned int destfield; // field of dest
} fgEventTuple;

KHASH_DECLARE(eventnode, fgEventTuple, unsigned int);

typedef struct FG__EVENT
{
  void* self; // Function object pointer
  fgListener event;
  kh_event_t* source; // Hash that this event belongs to
  struct FG__EVENT** dest; // root node of the source object this belongs to
  unsigned int srcfield;
  unsigned int destfield;
  struct FG__EVENT* next;
  struct FG__EVENT* prev;
  struct FG__EVENT* objnext;
  struct FG__EVENT* objprev;
} fgEvent;

enum FG_EVENTS
{
  FG_EVENT_NONE = 0,
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
  FG_WINDOW_TITLE,
  FG_WINDOW_ICON,
  FG_WINDOW_LEFT,
  FG_WINDOW_RIGHT,
  FG_WINDOW_TOP,
  FG_WINDOW_BOTTOM,
  FG_WINDOW_STATE,
  FG_WINDOW_ONMINIMIZE,
  FG_WINDOW_ONMAXIMIZE,
  FG_WINDOW_ONOPEN,
  FG_WINDOW_ONCLOSE,
  FG_LAYER_MOUSE,
  FG_CIRCLE_ARC_INNER_MIN,
  FG_CIRCLE_ARC_INNER_MAX,
  FG_CIRCLE_ARC_OUTER_MIN,
  FG_CIRCLE_ARC_OUTER_MAX,
  FG_BOX_CORNER_NW,
  FG_BOX_CORNER_NE,
  FG_BOX_CORNER_SE,
  FG_BOX_CORNER_SW,
  FG_TRIANGLE_CORNER_TOP,
  FG_TRIANGLE_CORNER_LEFT,
  FG_TRIANGLE_CORNER_RIGHT,
  FG_TRIANGLE_POINT,
  FG_EVENT_FILL_COLOR,
  FG_EVENT_BORDER,
  FG_EVENT_BORDER_COLOR,
  FG_EVENT_BLUR,
  FG_EVENT_ASSET,
  FG_EVENT_QUIET = 0x8000,
};

FG_COMPILER_DLLEXPORT fgEvent* fgAllocEvent(void* self, fgListener fn, unsigned int srcfield, kh_event_t* source, unsigned int destfield, struct FG__EVENT** dest, unsigned int aux);
FG_COMPILER_DLLEXPORT void fgDestroyEvent(struct FG__ROOT* root, fgEvent* e);
FG_COMPILER_DLLEXPORT void fgRemoveEvent(struct FG__ROOT* root, fgEvent* e);
FG_COMPILER_DLLEXPORT void fgSendEvent(struct FG__ROOT* root, kh_event_t* hash, unsigned int key, void* target, void* field);

#ifdef  __cplusplus
}
#endif

#endif