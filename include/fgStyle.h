// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_STYLE_H__
#define __FG_STYLE_H__

#include "feathergui.h"

#ifdef  __cplusplus
extern "C" {
#endif

  typedef struct _FG_STYLE_MSG
  {
    FG_Msg msg;
    struct _FG_STYLE_MSG* next;
    unsigned int sz;
  } fgStyleMsg;

  typedef struct _FG_STYLE
  {
    fgStyleMsg* styles;

#ifdef  __cplusplus
    FG_DLLEXPORT fgStyleMsg* AddStyleMsg(const FG_Msg* msg);
    FG_DLLEXPORT void RemoveStyleMsg(fgStyleMsg* msg);
#endif
  } fgStyle;

  FG_EXTERN void fgStyle_Init(fgStyle* self);
  FG_EXTERN void fgStyle_Destroy(fgStyle* self);
  FG_EXTERN FG_UINT fgStyle_GetName(const char* name, char flag);

  FG_EXTERN fgStyleMsg* fgStyle_AddStyleMsg(fgStyle* self, const FG_Msg* msg, unsigned int arg1size, unsigned int arg2size);
  FG_EXTERN fgStyleMsg* fgStyle_CloneStyleMsg(const fgStyleMsg* self);
  FG_EXTERN void fgStyle_RemoveStyleMsg(fgStyle* self, fgStyleMsg* msg);

#ifdef  __cplusplus
}

template<FG_MSGTYPE type, typename... Args>
inline fgStyleMsg* AddStyleMsg(fgStyle* style, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, 0, 0);
}
template<FG_MSGTYPE type, typename... Args>
inline fgStyleMsg* AddStyleSubMsg(fgStyle* style, unsigned short sub, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  fgSendMsgCall<1, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, 0, 0);
}
template<FG_MSGTYPE type, typename Arg, typename... Args>
inline fgStyleMsg* AddStyleMsgArg(fgStyle* style, const Arg* arg, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.p = (void*)arg;
  fgSendMsgCall<2, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, sizeof(Arg), 0);
}
template<FG_MSGTYPE type, typename Arg, typename... Args>
inline fgStyleMsg* AddStyleSubMsgArg(fgStyle* style, unsigned short sub, const Arg* arg, Args... args)
{
  FG_Msg msg = { 0 };
  msg.type = type;
  msg.subtype = sub;
  msg.p = (void*)arg;
  fgSendMsgCall<2, Args...>::F(msg, args...);
  return fgStyle_AddStyleMsg(style, &msg, sizeof(Arg), 0);
}
#endif

#endif