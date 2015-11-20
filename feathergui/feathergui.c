// Copyright ©2015 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathergui.h"
#include <intrin.h>
#include <limits.h>

const fgElement fgElement_DEFAULT = { {0, 0, 0, 0, 0, 1, 0, 1 }, 0, {0, 0, 0, 0} };

void FG_FASTCALL fgVector_Init(fgVector* self)
{
  memset(self,0,sizeof(fgVector));
}
void FG_FASTCALL fgVector_Destroy(fgVector* self)
{
  if(self->p) free(self->p);
}
// TODO: Speed this up using a custom simplistic heap allocator
void FG_FASTCALL fgVector_SetSize(fgVector* self, FG_UINT length, FG_UINT size)
{
  self->s=length*size;
  self->p=realloc(self->p,self->s);
  if(self->l>length) self->l = length;
}
void FG_FASTCALL fgVector_CheckSize(fgVector* self, FG_UINT size)
{
  if((self->l*size)>=self->s)
    fgVector_SetSize(self,fbnext(self->l),size);
  assert((self->l*size)<self->s);
}
void FG_FASTCALL fgVector_Remove(fgVector* self, FG_UINT index, FG_UINT size)
{
  assert(index<self->l);
  memmove(((char*)self->p)+(index*size),((char*)self->p)+((index+1)*size),((--self->l)-index)*size);
}

AbsVec FG_FASTCALL ResolveVec(const CVec* v, const AbsRect* last)
{
  AbsVec r = { v->x.abs, v->y.abs };
  assert(last!=0);
  r.x += lerp(last->left,last->right,v->x.rel);
  r.y += lerp(last->top,last->bottom,v->y.rel);
  return r; 
}

// This uses a standard inclusive-exclusive rectangle interpretation.
char FG_FASTCALL HitAbsRect(const AbsRect* r, FABS x, FABS y)
{
  assert(r!=0);
  return (x < r->right) && (x >= r->left) && (y < r->bottom) && (y >= r->top);
}

char FG_FASTCALL MsgHitAbsRect(const FG_Msg* msg, const AbsRect* r)
{
  assert(msg!=0 && r!=0);
  return HitAbsRect(r,(FABS)msg->x,(FABS)msg->y);
}

char FG_FASTCALL CompareAbsRects(const AbsRect* l, const AbsRect* r)
{
  assert(l != 0 && r != 0);
  return ((((l->left - r->left) != (l->right - r->right))) << 1)
    | ((((l->top - r->top) != (l->bottom - r->bottom))) << 2)
    | ((((l->left != r->left) || (l->right != r->right))) << 3)
    | ((((l->top != r->top) || (l->bottom != r->bottom))) << 4);
}
char FG_FASTCALL CompareCRects(const CRect* l, const CRect* r)
{
  assert(l!=0 && r!=0);
  return ((((l->left.abs - r->left.abs) != (l->right.abs - r->right.abs))) << 1)
    | ((((l->top.abs - r->top.abs) != (l->bottom.abs - r->bottom.abs))) << 2)
    | ((((l->left.abs != r->left.abs) || (l->right.abs != r->right.abs))) << 3)
    | ((((l->top.abs != r->top.abs) || (l->bottom.abs != r->bottom.abs))) << 4)
    | ((((l->left.rel - r->left.rel) != (l->right.rel - r->right.rel))) << 1)
    | ((((l->top.rel - r->top.rel) != (l->bottom.rel - r->bottom.rel))) << 2)
    | ((((l->left.rel != r->left.rel) || (l->right.rel != r->right.rel))) << 3)
    | ((((l->top.rel != r->top.rel) || (l->bottom.rel != r->bottom.rel))) << 4);
}
char FG_FASTCALL CompareElements(const fgElement* l, const fgElement* r)
{
  assert(l != 0 && r != 0);
  return CompareCRects(&l->area, &r->area)
    | (((l->center.x.abs != r->center.x.abs) || (l->center.x.rel != r->center.x.rel)) << 5)
    | (((l->center.y.abs != r->center.y.abs) || (l->center.y.rel != r->center.y.rel)) << 6)
    | ((l->rotation != r->rotation) << 7);
}

void FG_FASTCALL MoveCRect(AbsVec v, CRect* r)
{
  FABS dx,dy;
  dx=r->right.abs-r->left.abs;
  dy=r->bottom.abs-r->top.abs;
  r->left.abs=v.x;
  r->top.abs=v.y;
  r->right.abs=v.x+dx;
  r->right.abs=v.y+dy;
}

/*char FG_FASTCALL CompChildOrder(const fgChild* l, const fgChild* r)
{
  const fgChild* cur;
  unsigned int ldepth=0,rdepth=0,diff;
  assert(l!=0 && r!=0);
  if(l==r) return 0;
  cur=l;
  while((cur=cur->parent)!=0) ++ldepth;
  cur=r;
  while((cur=cur->parent)!=0) ++rdepth;

  diff=(ldepth<rdepth) - (ldepth>rdepth);
  while(ldepth<rdepth) { r=r->parent; --rdepth; }
  while(ldepth>rdepth) { l=l->parent; --ldepth; }

  if(l==r) return diff;

  while(l->parent!=r->parent)
  {
    r=r->parent;
    l=l->parent;
  }

  return (l->order>r->order) - (l->order<r->order);
}*/

//void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[static 4]) // Gotta love VC++ not supporting C99 which is 14 FUCKING YEARS OLD
void FG_FASTCALL ToIntAbsRect(const AbsRect* r, int target[4])
{
  _mm_storeu_si128((__m128i*)target,_mm_cvttps_epi32(_mm_loadu_ps(&r->left)));
}

void FG_FASTCALL ToLongAbsRect(const AbsRect* r, long target[4])
{
#if ULONG_MAX==UINT_MAX
  _mm_storeu_si128((__m128i*)target,_mm_cvttps_epi32(_mm_loadu_ps(&r->left)));
#else
  int hold[4];
  _mm_storeu_si128((__m128i*)hold,_mm_cvtps_epi32(_mm_loadu_ps(&r->left)));
  target[0]=hold[0];
  target[1]=hold[1];
  target[2]=hold[2];
  target[3]=hold[3];
#endif

}
