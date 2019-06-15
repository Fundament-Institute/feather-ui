// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__MESSAGE_H
#define FG__MESSAGE_H

#include "feather.h"

typedef unsigned short fgMsgType;

enum FG_MESSAGE_TYPE
{
  FG_MSG_CUSTOM,
  FG_MSG_CONSTRUCT,
  FG_MSG_DESTORY,
  FG_MSG_DRAW,
  FG_MSG_MOUSE_DOWN,
  FG_MSG_MOUSE_DBLCLICK,
  FG_MSG_MOUSE_UP,
  FG_MSG_MOUSE_OFF,
  FG_MSG_MOUSE_MOVE,
  FG_MSG_MOUSE_SCROLL,
  FG_MSG_TOUCH_BEGIN,
  FG_MSG_TOUCH_MOVE,
  FG_MSG_TOUCH_END,
  FG_MSG_KEY_UP,
  FG_MSG_KEY_DOWN,
  FG_MSG_KEY_CHAR,
  FG_MSG_JOY_BUTTON_DOWN,
  FG_MSG_JOY_BUTTON_UP,
  FG_MSG_JOY_AXIS,
  FG_MSG_GOT_FOCUS,
  FG_MSG_LOST_FOCUS,
  FG_MSG_DRAG_DROP,
};

typedef struct
{
  fgMsgType type;
  fgMsgType subtype;
  union
  {
    struct Custom {};
    struct Construct {};
    struct Destroy {};
    struct Draw {};
    struct MouseDown {};
    struct MouseDblClick {};
    struct MouseUp {};
    struct MouseOff {};
    struct MouseMove {};
    struct MouseScroll {};
    struct TouchBegin {};
    struct TouchEnd {};
    struct TouchMove {};
    struct KeyUp {};
    struct KeyDown {};
    struct KeyChar {};
    struct JoyButtonDown {};
    struct JoyButtonUp {};
    struct JoyAxis {};
    struct GotFocus {};
    struct LostFocus {};
    struct DragDrop {};
  };
} fgMessage;

typedef union
{
  fgError error;
  void* ptr;
  struct Custom {};
  struct Construct {};
  struct Destroy {};
  struct Draw {};
  struct MouseDown {};
  struct MouseDblClick {};
  struct MouseUp {};
  struct MouseOff {};
  struct MouseMove {};
  struct MouseScroll {};
  struct TouchBegin {};
  struct TouchEnd {};
  struct TouchMove {};
  struct KeyUp {};
  struct KeyDown {};
  struct KeyChar {};
  struct JoyButtonDown {};
  struct JoyButtonUp {};
  struct JoyAxis {};
  struct GotFocus {};
  struct LostFocus {};
  struct DragDrop {};
} fgMessageResult;

typedef struct {} fgBehaviorState;

typedef fgMessageResult(*fgBehaviorFunction)(fgMessage* msg, fgBehaviorState* state);

#endif
