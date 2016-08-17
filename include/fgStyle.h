// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef _FG_STYLE_H__
#define _FG_STYLE_H__

#include "feathergui.h"

#ifdef  __cplusplus
extern "C" {
#endif

  typedef struct _FG_STYLE_MSG
  {
    FG_Msg msg;
    struct _FG_STYLE_MSG* next;
  } fgStyleMsg;

  typedef struct _FG_STYLE
  {
    fgStyleMsg* styles;
  } fgStyle;

  FG_EXTERN void FG_FASTCALL fgStyle_Init(fgStyle* self);
  FG_EXTERN void FG_FASTCALL fgStyle_Destroy(fgStyle* self);
  FG_EXTERN FG_UINT FG_FASTCALL fgStyle_GetName(const char* name, bool flag);

  FG_EXTERN fgStyleMsg* FG_FASTCALL fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, const void* arg1, size_t arglen1, const void* arg2, size_t arglen2);
  FG_EXTERN void FG_FASTCALL fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg);
#ifdef  __cplusplus
}

template<FG_MSGTYPE type, typename... Args>
inline fgStyleMsg* AddStyleMsg(fgStyle* style, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, 0, 0, 0, 0);
}
template<FG_MSGTYPE type, typename Arg, typename... Args>
inline fgStyleMsg* AddStyleMsgArg(fgStyle* style, const Arg* arg, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<2, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, arg, sizeof(Arg), 0, 0);
}
#endif

#endif