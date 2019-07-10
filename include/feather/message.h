// Copyright (c)2019 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feather.h"

#ifndef FG__MESSAGE_H
#define FG__MESSAGE_H

#include "feather.h"
#include "calc.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef unsigned short fgMsgType;
struct FG__OUTLINE_NODE;
struct FG__DOCUMENT_NODE;

#ifdef  __cplusplus
enum FG_MESSAGE_TYPE : fgMsgType
#else
enum FG_MESSAGE_TYPE
#endif
{
  FG_MSG_CUSTOM,
  FG_MSG_CONSTRUCT,
  FG_MSG_DESTROY,
  FG_MSG_DRAW,
  FG_MSG_MOUSE_DOWN,
  FG_MSG_MOUSE_DBLCLICK,
  FG_MSG_MOUSE_UP,
  FG_MSG_MOUSE_ON,
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
  FG_MSG_STATE_GET,
  FG_MSG_STATE_SET,
  FG_MSG_ADD_CHILD,
  FG_MSG_REMOVE_CHILD,
  FG_MSG_ACTION,
};

typedef struct
{
  fgMsgType type;
  fgMsgType subtype;
  union
  {
    struct Custom { } custom;
    struct Construct { void* ptr; } construct;
    struct Destroy {} destroy;
    struct Draw { void* data; fgRect area; } draw;
    struct Mouse { float x; float y; unsigned char button; unsigned char all; } mouse;
    struct MouseScroll { float x; float y; short delta; short hdelta; } mouseScroll;
    struct Touch { float x; float y; short index; } touch;
    struct Key { unsigned char code; unsigned char sigkeys; } key;
    struct KeyChar { int unicode; unsigned char sigkeys; } keyChar;
    struct KeyRaw { short rawcode; unsigned char sigkeys; } keyRaw;
    struct JoyButton { short button; } joyButton;
    struct JoyAxis { short axis; float value; } joyAxis;
    struct GotFocus {} gotFocus;
    struct LostFocus {} lostFocus;
    struct DragDrop {} dragDrop;
    struct GetState { const char* id; fgCalcResult* value; } getState;
    struct SetState { const char* id; fgCalcResult value; } setState;
    struct Child { struct FG__DOCUMENT_NODE* node; float scale; fgVec dpi; } child;
  };
} fgMessage;

typedef union
{
  fgError error;
  void* ptr;
  struct Custom { } custom;
  struct Construct { size_t size; } construct;
  struct Destroy {} destroy;
  struct Draw { } draw;
  struct Mouse {} mouse;
  struct Touch {} touch;
  struct Key {} key;
  struct JoyButton {} joyButton;
  struct JoyAxis {} joyAxis;
  struct GotFocus {} gotFocus;
  struct LostFocus {} lostFocus;
  struct DragDrop {} dragDrop;
} fgMessageResult;

typedef fgMessageResult(*fgBehaviorFunction)(const struct FG__ROOT*, struct FG__DOCUMENT_NODE*, const fgMessage*);

FG_COMPILER_DLLEXPORT fgMessageResult fgDefaultBehavior(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);
FG_COMPILER_DLLEXPORT fgMessageResult fgSendMessage(const struct FG__ROOT* root, struct FG__DOCUMENT_NODE* node, const fgMessage* msg);

#ifdef  __cplusplus
}
#endif

#endif
