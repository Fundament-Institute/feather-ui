// Copyright ©2016 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTest.h"
#include "../fgNull/fgNull.h"
#include "fgButton.h"
#include "fgMenu.h"
#include "fgWindow.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/fgNull64_d.lib")
#elif defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/fgNull64.lib")
#elif defined(BSS_DEBUG)
#pragma comment(lib, "../bin/fgNull_d.lib")
#else
#pragma comment(lib, "../bin/fgNull.lib")
#endif

typedef struct {
  size_t first;
  size_t second;
} RETPAIR;

typedef struct
{
  const char* NAME;
  RETPAIR (*FUNC)();
} TESTDEF;

static FILE* failedtests=0;

#define BEGINTEST RETPAIR __testret = {0,0}
#define ENDTEST return __testret
#define FAILEDTEST(t) fprintf(failedtests, "Test #%u Failed  < %s >\n",  __testret.first,MAKESTRING(t))
#define TEST(t) { ++__testret.first; if(t) ++__testret.second; else FAILEDTEST(t); }
//#define TESTARRAY(t,f) _ITERFUNC(__testret,t,[&](uint i) -> bool { f });
//#define TESTALL(t,f) _ITERALL(__testret,t,[&](uint i) -> bool { f });
//#define TESTCOUNT(c,t) { for(uint i = 0; i < c; ++i) TEST(t) }
//#define TESTCOUNTALL(c,t) { bool __val=true; for(uint i = 0; i < c; ++i) __val=__val&&(t); TEST(__val); }

static fgRoot* gui=0;

// Highly optimized traditional tolerance based approach to comparing floating point numbers, found here: http://www.randydillon.org/Papers/2007/everfast.htm
char BSS_FASTCALL fcompare(float af, float bf, __int32 maxDiff) // maxDiff default is 1
{ 
  __int32 ai,bi,test,diff,v1,v2;
  assert(af!=0.0f && bf!=0.0f); // Use fsmall for this
  ai = *((__int32*)&af);
  bi = *((__int32*)&bf);
  test = (-(__int32)(((unsigned __int32)(ai^bi))>>31));
  assert((0 == test) || (0xFFFFFFFF == test));
  diff = ((ai + test) ^ (test & 0x7fffffff)) - bi;
  v1 = maxDiff + diff;
  v2 = maxDiff - diff;
  return (v1|v2) >= 0;
}
char BSS_FASTCALL dcompare(double af, double bf, __int64 maxDiff)
{ 
  __int64 ai,bi,test,diff,v1,v2;
  assert(af!=0.0 && bf!=0.0); // Use fsmall for this
  ai = *((__int64*)&af);
  bi = *((__int64*)&bf);
  test = (-(__int64)(((unsigned __int64)(ai^bi))>>63));
  assert((0 == test) || (0xFFFFFFFFFFFFFFFF == test));
  diff = ((ai + test) ^ (test & 0x7fffffffffffffff)) - bi;
  v1 = maxDiff + diff;
  v2 = maxDiff - diff;
  return (v1|v2) >= 0;
}

RETPAIR test_feathergui()
{
  BEGINTEST;
  fgElement ch;
  fgElement ch2;
  fgElement ch3;
  fgElement top;
  fgTransform zeroelement;
  fgTransform elem;
  AbsRect out;
  AbsRect last;
  CRect crect;
  CRect cother;
  CVec vec;
  AbsVec res;
  FG_Msg msg;
  fgVector v; //fgVector functionality tests

  TEST(fbnext(0)==1); // Basic function tests
  TEST(fbnext(1)==2);
  TEST(fbnext(2)==4);
  TEST(lerp(0.0f,1.0f,0.5f)==0.5f);
  TEST(lerp(-1.0f,1.0f,0.5f)==0.0f);
  TEST(lerp(0.0f,1.0f,1.0f)==1.0f);
  TEST(lerp(-1.0f,1.0f,0.0f)==-1.0f);
  
  fgVector_Init(&v);
  TEST(!v.p);
  fgVector_Add(v,2,int);
  fgVector_Remove(&v,0,sizeof(int));
  TEST(v.l==0);
  fgVector_Insert(v,1,0,int);
  TEST(fgVector_Get(v,0,int)==1);
  TEST(v.l==1);
  fgVector_Insert(v,4,0,int);
  TEST(fgVector_Get(v,0,int)==1);
  TEST(fgVector_Get(v,1,int)==4);
  TEST(v.l==2);
  fgVector_Insert(v,8,2,int);
  TEST(fgVector_Get(v,0,int)==1);
  TEST(fgVector_Get(v,1,int)==4);
  TEST(fgVector_Get(v,2,int)==8);
  TEST(v.l==3);
  fgVector_Add(v,16,int);
  TEST(fgVector_Get(v,3,int)==16);
  TEST(v.l==4);
  fgVector_SetSize(&v,2,sizeof(int));
  TEST(v.l==2);
  TEST(fgVector_Get(v,0,int)==1);
  TEST(fgVector_Get(v,1,int)==4);
  fgVector_Remove(&v,1,sizeof(int));
  fgVector_Remove(&v,0,sizeof(int));
  TEST(v.l==0);
  fgVector_Destroy(&v);

  fgElement_Init(&ch, 0, 0, 0, 0); // Basic initialization test
  memset(&zeroelement,0,sizeof(fgTransform));
  TEST(!memcmp(&ch.element,&zeroelement,sizeof(fgTransform)));
  //TEST(ch.destroy==&fgElement_Destroy); // Once upon a time, these tests worked, but VS2012 likes having different function pointers to the same function, which is technically legal, so we can't check this. Luckily it would be really weird if it broke anyway.
  //TEST(ch.free==&free);
  TEST(ch.next==0);
  TEST(ch.order==0);
  TEST(ch.parent==0);
  TEST(ch.prev==0);
  TEST(ch.root==0);
  TEST(ch.last==0);

  fgElement_Init(&top, 0, 0, 0, 0); // Basic inheritance test
  fgElement_SetParent(&ch,&top);
  TEST(ch.next==0);
  TEST(ch.prev==0);
  TEST(ch.parent==&top);
  TEST(top.parent==0);
  TEST(top.root==&ch);
  TEST(top.last==&ch);
  fgElement_Destroy(&ch);
  TEST(top.root==0);
  TEST(top.last==0);

  fgElement_Init(&ch, 0, 0, 0, 0); // Inheritance ordering test.
  fgElement_Init(&ch2, 0, 0, 0, 0);
  fgElement_Init(&ch3, 0, 0, 0, 0);
  ch.order=1;
  ch2.order=2;
  ch3.order=3;

  fgElement_SetParent(&ch,&top);
  TEST(top.root==&ch);
  TEST(top.last==&ch);
  fgElement_SetParent(&ch2,&top);
  TEST(top.root==&ch2);
  TEST(top.root->next==&ch);
  TEST(top.last==&ch);
  fgElement_SetParent(&ch3,&top);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch2);
  TEST(top.root->next->next==&ch);
  TEST(top.root->next->next->next==0);
  TEST(top.last==&ch);
  TEST(ch.parent==&top);
  TEST(ch2.parent==&top);
  TEST(ch3.parent==&top);

  fgElement_Destroy(&ch);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch2);
  TEST(top.root->next->next==0);
  TEST(top.last==&ch2);

  fgElement_SetParent(&ch2,&ch3, 0);
  TEST(top.root==&ch3);
  TEST(top.root->next==0);
  TEST(top.last==&ch3);
  TEST(ch3.root==&ch2);
  TEST(ch2.parent==&ch3);
  TEST(ch3.parent==&top);

  fgElement_SetParent(&ch2,0, 0);
  TEST(ch3.root==0);
  TEST(ch3.last==0);

  fgElement_Init(&ch, 0, 0, 0);
  ch.order=1;

  fgElement_SetParent(&ch2,&top, 0);
  fgElement_SetParent(&ch,&top, 0);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch2);
  TEST(top.root->next->next==&ch);
  TEST(top.root->next->next->next==0);
  TEST(top.last==&ch);

  fgElement_Destroy(&ch2);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch);
  TEST(top.root->next->next==0);
  TEST(top.last==&ch);
  
  fgElement_Init(&ch2, 0, 0, 0);
  ch2.order=2;

  fgElement_SetParent(&ch2,&top, 0);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch2);
  TEST(top.root->next->next==&ch);
  TEST(top.root->next->next->next==0);
  TEST(top.last==&ch);

  fgElement_SetParent(&ch2,0, 0);
  ch2.order=1;
  fgElement_SetParent(&ch2,&top, 0);
  TEST(top.root==&ch3);
  TEST(top.root->next==&ch2); // Equal orders should be appended before the first one encountered.
  TEST(top.root->next->next==&ch);
  TEST(top.root->next->next->next==0);
  TEST(top.last==&ch);

  memset(&elem,0,sizeof(fgTransform));
  memset(&last,0,sizeof(AbsRect));
  last.right=last.bottom=1.0f;
  elem.area.top.abs=0.2f;
  elem.area.left.abs=0.3f;
  elem.area.right.abs=0.4f;
  elem.area.bottom.abs=0.5f;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==0.2f && // Surprisingly, this works with floats but may break in single-precision mode. Replace with fcompare if that happens.
        out.left==0.3f &&
        out.right==0.4f &&
        out.bottom==0.5f);

  elem.area.top.rel=1;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==1.2f &&
        out.left==0.3f &&
        out.right==0.4f &&
        out.bottom==0.5f);

  elem.area.left.rel=1;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==1.2f &&
        out.left==1.3f &&
        out.right==0.4f &&
        out.bottom==0.5f);
  
  elem.area.right.rel=2;
  elem.area.bottom.rel=0.5;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==1.2f &&
        out.left==1.3f &&
        out.right==2.4f &&
        out.bottom==1.0f);

  elem.area.top.abs=0;
  elem.area.left.abs=0;
  elem.area.right.abs=0;
  elem.area.bottom.abs=0;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==1.0f &&
        out.left==1.0f &&
        out.right==2.0f &&
        out.bottom==0.5f);

  last.top=last.left=1.0f;
  ResolveRectCache(&elem, &out, &last);
  TEST(out.top==1.0f &&
        out.left==1.0f &&
        out.right==1.0f &&
        out.bottom==1.0f);

  vec.x.abs=5;
  vec.y.abs=6;
  vec.x.rel=0;
  vec.y.rel=0;
  res=ResolveVec(&vec,&last);
  TEST(res.x==6 && res.y==7);

  last.top=0;
  last.left=0;
  res=ResolveVec(&vec,&last);
  TEST(res.x==5 && res.y==6);

  vec.x.rel=0.5;
  vec.y.rel=0.5;
  res=ResolveVec(&vec,&last);
  TEST(res.x==5.5 && res.y==6.5);
  
  last.right=0.5;
  last.bottom=0.5;
  res=ResolveVec(&vec,&last);
  TEST(res.x==5.25 && res.y==6.25);

  vec.x.abs=0;
  vec.y.abs=0;
  res=ResolveVec(&vec,&last);
  TEST(res.x==0.25 && res.y==0.25);
  
  last.top=2;
  last.left=1;
  last.right=2;
  last.bottom=4;
  res=ResolveVec(&vec,&last);
  TEST(res.x==1.5 && res.y==3);

  memset(&msg,0,sizeof(FG_Msg));
  TEST(MsgHitAbsRect(&msg,&last)==0); 

  msg.x=1;
  msg.y=2;
  TEST(MsgHitAbsRect(&msg,&last)!=0); // Like everything else, feather uses inclusive-exclusive coordinates.
  
  msg.x=1;
  msg.y=3;
  TEST(MsgHitAbsRect(&msg,&last)!=0); 

  msg.x=2;
  msg.y=4;
  TEST(MsgHitAbsRect(&msg,&last)==0); 

  fgElement_SetParent(&ch3,&ch2);
  fgElement_SetParent(&ch2,&ch);
  top.element.area.left.abs=0;
  top.element.area.top.abs=1;
  top.element.area.right.abs=1;
  top.element.area.bottom.abs=2;
  ch.element.area.left.rel=0.5;
  ch.element.area.top.abs=1;
  ch.element.area.right.rel=1;
  ch.element.area.bottom.rel=0;
  ch.element.area.bottom.abs=3;
  ch2.element.area.left.abs=-5;
  ch2.element.area.left.rel=2;
  ch2.element.area.top.rel=0.5;
  ch2.element.area.right.rel=0.5;
  ch2.element.area.bottom.abs=2;
  ch3.element.area.top.abs=0.5;
  ch3.element.area.top.rel=-1;
  ch3.element.area.right.abs=2;
  ch3.element.area.bottom.rel=0;

  ResolveRect(&top, &out);
  TEST(out.left==0 &&
        out.top==1.0f &&
        out.right==1.0f &&
        out.bottom==2.0f);

  ResolveRect(&ch, &out);
  TEST(out.left==0.5f &&
        out.top==2.0f &&
        out.right==1.0f &&
        out.bottom==4.0f);

  ResolveRect(&ch2, &out);
  TEST(out.left==-3.5f &&
        out.top==3.0f &&
        out.right==0.75f &&
        out.bottom==4.0f);

  ResolveRect(&ch3, &out);
  TEST(out.left==-3.5f &&
        out.top==2.5f &&
        out.right==-1.5f &&
        out.bottom==3.0f);

  MsgHitCRect(&msg,&top);
  MsgHitCRect(&msg,&ch);
  MsgHitCRect(&msg,&ch2);
  MsgHitCRect(&msg,&ch3);

  memset(&crect,0,sizeof(CRect));
  memset(&cother,0,sizeof(CRect));
  TEST(CompareCRects(&crect, &cother)==0);
  crect.left.abs=1;
  TEST(CompareCRects(&crect, &cother)!=0);
  crect.left.abs=0;
  crect.right.rel=1;
  TEST(CompareCRects(&crect, &cother)!=0);
  cother.left.abs=0;
  cother.right.rel=1;
  TEST(CompareCRects(&crect, &cother)==0);

  {
    int conv[4];
    long lconv[4];
    CRect ctest = { 2,0.3,4,2.0,8,0.5,16,1.0 };
    AbsVec cmov = { -1,-2 };
    CRect ctestans = { 1,0.3,2,2.0,7,0.5,14,1.0 };
    MoveCRect(cmov,&ctest);
    TEST(!memcmp(&ctest,&ctestans,sizeof(CRect)));
    ToIntAbsRect(&last,conv);
    ToLongAbsRect(&last,lconv);
  }


//FG_EXTERN void FG_FASTCALL LList_Remove(fgElement* self, fgElement** root, fgElement** last);
//FG_EXTERN void FG_FASTCALL LList_Add(fgElement* self, fgElement** root, fgElement** last, fgFlag flag);
//FG_EXTERN void FG_FASTCALL LList_Insert(fgElement* self, fgElement* target, fgElement** last);
//FG_EXTERN void FG_FASTCALL LList_ChangeOrder(fgElement* self, fgElement** root, fgElement** last, fgFlag flag);


  ENDTEST;
}

RETPAIR test_Window()
{
  BEGINTEST;
  ENDTEST;
}

char test_root_STAGE=0;
char FG_FASTCALL donothing(void* a) { test_root_STAGE=2; return 1; } // Returning one auto deallocates the node
char FG_FASTCALL dontfree(void* a) { test_root_STAGE=1; return 0; } // Returning zero means the node won't be automatically deallocated

RETPAIR test_Root()
{
  BEGINTEST;
  fgWindow* top;
  fgDeferAction* action = fgRoot_AllocAction(&donothing,0,5);
  fgDeferAction* action2 = fgRoot_AllocAction(&dontfree,0,2);
  gui = fgInitialize();
  top = (fgWindow*)fgWindow_Create("test",0,1,0);

  fgRoot_AddAction(gui,action);
  fgRoot_AddAction(gui,action2);
  (*gui->update)(gui,1);
  TEST(test_root_STAGE==0);
  (*gui->update)(gui,1);
  TEST(test_root_STAGE==1);
  fgRoot_DeallocAction(gui,action2);
  (*gui->update)(gui,1);
  TEST(test_root_STAGE==1);
  (*gui->update)(gui,10);
  TEST(test_root_STAGE==2);
  fgVoidMessage((fgControl*)top,FG_ADDCHILD,fgButton_Create(fgResource_Create("fake"),0,0,2,0));
  fgVoidMessage((fgControl*)top,FG_ADDCHILD,top->region.root->root);
  fgVoidMessage((fgControl*)top,FG_ADDCHILD,fgButton_Create(fgLoadText("fake",0,"arial.ttf",14,0),0,0,4,0));
  fgVoidMessage((fgControl*)top,FG_ADDCHILD,fgLoadImage("fake"));
  //fgStatic_Message(top->region.rlist,FG_RADDCHILD,fgLoadText("fake",0,"arial.ttf",14,0),0);
  //fgVoidMessage((fgControl*)top,FG_ADDCHILD,top->region.rlist->transform.root);
  fgRoot_Render(gui);
  ENDTEST;
}

RETPAIR test_Window()
{
  BEGINTEST;
  ENDTEST;
}

RETPAIR test_Button()
{
  BEGINTEST;
  ENDTEST;
}

RETPAIR test_List()
{
  BEGINTEST;
  ENDTEST;
}

RETPAIR test_Menu()
{
  BEGINTEST;
  ENDTEST;
}

RETPAIR test_TERMINATE()
{
  BEGINTEST;
  fgTerminate(gui);
  ENDTEST;
}
//
//void mystery(int* a, int* b)
//{
//	if((*a%2)==0)
//		*b = *a/2;
//	else
//		*b = 3*(*a)+1;
//}
//
//int main(int argc, char** argv)
//{
//	int n = 6;
//	int m = 0;
//	const char* str = ".!sBen..i.u.....n";
//	
//	while(n>1)
//	{
//		mystery(&n,&m);
//		printf("%c",str[m]);
//    //printf("%d\n",m);
//		n=m;
//	}
//
//  getc(stdin);
//	return 0;	
//}
//
//void mystery(int* a, int* b, int* c, int* d)
//{
//	*c = *a;
//	*c = *c * *a + *c;
//	*d = *b * 2 + 1;
//}
//
//int main()
//{
//	int x = 5;
//	int y = 2;
//	int z = 0;
//	int w = 0;
//
//	mystery(&x,&y,&z,&w);
//	mystery(&z,&w,&x,&y);
//	mystery(&w,&z,&y,&x);
//	printf("%i,%i",z,w);
//  getc(stdin);
//	return 0;
//}

//int files_match(char * fname1, char * fname2)
//{
//  char buf1[200];
//  char buf2[200];
//  FILE* f1 = fopen(fname1,"r");
//  FILE* f2 = fopen(fname2,"r");
//  
//  if(!f1 || !f2)
//    return 0;
//
//  while(!feof(f1) && !feof(f2))
//  {
//    fgets(buf1,200,f1);
//    fgets(buf2,200,f2);
//    if(strcmp(buf1,buf2)!=0)
//      return 0;
//  }
//
//  return 1;
//}
//
//int main() {
//  printf("%d",files_match("D:/NOTES.txt","D:/NOTES.txt"));
//  printf("%d",files_match("D:/NOTES.txt","D:/NOTES2.txt"));
//  printf("%d",files_match("D:/NOTES3.txt","D:/NOTES.txt"));
//  getc(stdin);
//  return 0;
//}

int main(int argc, char** argv)
{
  static const int COLUMNS[3] = { 24, 11, 8 };
  static TESTDEF tests[] = {
    { "feathergui.h", &test_feathergui },
    { "fgControl.h", &test_Window },
    { "fgRoot.h", &test_Root },
    { "fgWindow.h", &test_Window },
    { "fgButton.h", &test_Button },
    { "fgList.h", &test_List },
    { "fgMenu.h", &test_Menu },
    { "TERMINATE", &test_TERMINATE },
  };

  static const size_t NUMTESTS=sizeof(tests)/sizeof(TESTDEF);
  RETPAIR numpassed;
  unsigned int i;
  char buf[44]; // 21 + 21 + 1 + null terminator
  unsigned int failures[sizeof(tests)/sizeof(TESTDEF)];
  unsigned int nfail=0;

  srand(time(NULL));
  failedtests=fopen("failedtests.txt","wb");

  printf("Feather GUI Abstraction Layer v%i.%i.%i: Unit Tests\nCopyright (c)2012 Black Sphere Studios\n\n",FGUI_VERSION_MAJOR,FGUI_VERSION_MINOR,FGUI_VERSION_REVISION);
  printf("%-*s %-*s %-*s\n",COLUMNS[0],"Test Name", COLUMNS[1],"Subtests", COLUMNS[2],"Pass/Fail");

  for(i = 0; i < NUMTESTS; ++i)
  {
    numpassed=tests[i].FUNC(); //First is total, second is succeeded
    if(numpassed.first!=numpassed.second) failures[nfail++]=i;
    sprintf(buf,"%u/%u",numpassed.second,numpassed.first);
    printf("%-*s %*s %-*s\n",COLUMNS[0],tests[i].NAME, COLUMNS[1],buf, COLUMNS[2],(numpassed.first==numpassed.second)?"PASS":"FAIL");
  }

  if(!nfail)
    printf("\nAll tests passed successfully!\n");
  else
  {
    printf("\nThe following tests failed: \n");
    for (i = 0; i < nfail; i++)
      printf("  %s\n", tests[failures[i]].NAME);
    printf("\nThese failures indicate either a misconfiguration on your system, or a potential bug. Please report all bugs to http://code.google.com/p/feathergui/issues/list\n\nA detailed list of failed tests was written to failedtests.txt\n");
  }

  printf("\nPress Enter to exit the program.");
  getc(stdin);
  return 0;
}

struct HINSTANCE__;

// WinMain function, simply a catcher that calls the main function
int __stdcall WinMain(struct HINSTANCE__* hInstance, struct HINSTANCE__* hPrevInstance, char* lpCmdLine, int nShowCmd)
{
  main(0,(char**)hInstance);
}