// Copyright ©2012 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgWinAPI.h"
#include "win32_includes.h"

extern WinAPIfgRoot* _fgroot;

void FG_FASTCALL WinAPIfgText_Destroy(WinAPIfgText* self)
{
  if(self->text) free(self->text);
  fgStatic_Destroy((fgStatic*)self);
}
fgStatic* FG_FASTCALL WinAPIfgText_Clone(WinAPIfgText* self)
{
  WinAPIfgText* ret = bssmalloc<WinAPIfgText>(1);
  size_t len=strlen(self->text)+1;
  fgStatic_Clone(&ret->st.render,self); // this copies over the destroy function
  ret->st.handle=0; //do handle duplication
  ret->text=(char*)malloc(len);
  strncpy(ret->text,self->text,len*sizeof(char));
  return ret;
}

void FG_FASTCALL WinAPIfgText_Message(fgStatic* self, unsigned char type, void* arg, int other)
{
  size_t len;
  char* str;
  WinAPIfgText* s=(WinAPIfgText*)self;
  assert(self!=0);

  switch(type)
  {
  case FG_RSETTEXT:
    if(s->text) free(s->text);
    len=strlen((const char*)arg)+1;
    str=malloc(len);
    strncpy(str,(const char*)arg,len);
    s->text=str;
    break;
  }

  fgStatic_Message(self,type,arg,other);
}

fgStatic* FG_FASTCALL fgLoadText(const char* text, fgFlag flags, const char* font, unsigned int fontsize)
{
  size_t len=strlen(text)+1;
  char* str=(char*)malloc(len);
  WinAPIfgText* r = bssmalloc<WinAPIfgText>(1);
  fgStatic_Init((fgStatic*)r);
  r->st.render.clone=&WinAPIfgText_Clone;
  r->st.render.element.flags=flags;
  r->st.render.element.destroy=WinAPIfgText_Destroy;
  strncpy(str,text,len);
  r->text=str;
  return (fgStatic*)r;
}