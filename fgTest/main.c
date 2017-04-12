// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "fgTest.h"
#include "fgWindow.h"
#include "fgRoot.h"
#include "fgLayout.h"
#include "fgDebug.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

static FILE* failedtests=0;
static fgRoot* gui=0;

// Highly optimized traditional tolerance based approach to comparing floating point numbers, found here: http://www.randydillon.org/Papers/2007/everfast.htm
char fcompare(float af, float bf, __int32 maxDiff) // maxDiff default is 1
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
char dcompare(double af, double bf, __int64 maxDiff)
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
  AbsRect last = { 1.1,2.9,3,-4 };
  CRect crect;
  CRect cother;
  CVec vec;
  AbsVec res;
  FG_Msg msg;

  TEST(fglerp(0.0f,1.0f,0.5f)==0.5f);
  TEST(fglerp(-1.0f,1.0f,0.5f)==0.0f);
  TEST(fglerp(0.0f,1.0f,1.0f)==1.0f);
  TEST(fglerp(-1.0f,1.0f,0.0f)==-1.0f);

  int rect[4];
  ToIntAbsRect(&last, rect);
  TEST(rect[0] == 1);
  TEST(rect[1] == 2);
  TEST(rect[2] == 3);
  TEST(rect[3] == -4);

  long lrect[4];
  ToLongAbsRect(&last, lrect);
  TEST(lrect[0] == 1);
  TEST(lrect[1] == 2);
  TEST(lrect[2] == 3);
  TEST(lrect[3] == -4);

  int size = sizeof(fgElement);
  ENDTEST;
}

RETPAIR test_Window()
{
  BEGINTEST;

  //fgRegisterFunction("statelistener", [](fgElement* self, const FG_Msg*) { fgElement* progbar = fgRoot_GetID(fgSingleton(), "#progbar"); progbar->SetValueF(self->GetValueF(0) / self->GetValueF(1), 0); progbar->SetText(cStrF("%i", self->GetValue(0))); });
  //fgRegisterFunction("makepressed", [](fgElement* self, const FG_Msg*) { self->SetText("Pressed!"); });

  fgLayout layout;
  fgLayout_Init(&layout);
  fgLayout_LoadFileXML(&layout, "../media/feathertest.xml");
  fgVoidMessage(&fgSingleton()->gui.element, FG_LAYOUTLOAD, &layout, 0);

  fgElement* tabfocus = fgGetID("#tabfocus");
  if(tabfocus)
  {
    fgElement* tabbutton = (fgElement*)fgVoidMessage(tabfocus, FG_GETSELECTEDITEM, 0, 0);
    FG_Msg m = { FG_ACTION, 0 };
    fgSendMessageAsync(tabbutton, &m, 0, 0);
  }
  while(fgSingleton()->backend.fgProcessMessages());

  fgLayout_Destroy(&layout);
  ENDTEST;
}

//char test_root_STAGE=0;
//char donothing(void* a) { test_root_STAGE=2; return 1; } // Returning one auto deallocates the node
//char dontfree(void* a) { test_root_STAGE=1; return 0; } // Returning zero means the node won't be automatically deallocated

RETPAIR test_Root()
{
  BEGINTEST;
  /*fgWindow* top;
  fgDeferAction* action = fgRoot_AllocAction(&donothing,0,5);
  fgDeferAction* action2 = fgRoot_AllocAction(&dontfree,0,2);

  gui = (fgRoot*)malloc(sizeof(fgRoot));
  AbsRect area = { 0 };
  fgRoot_Init(gui, &area, 0, 0);
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
  //fgVoidMessage((fgControl*)top,FG_ADDCHILD,top->region.rlist->transform.root);*/
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

// Visual studio cannot debug a dynamically loaded DLL, so for debugging purposes we statically link to the backend we are testing
#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/fgDirect2D_d.lib")
#elif defined(BSS_CPU_x86_64)
#pragma comment(lib, "../bin/fgDirect2D.lib")
#elif defined(BSS_DEBUG)
#pragma comment(lib, "../bin32/fgDirect2D32_d.lib")
#else
#pragma comment(lib, "../bin32/fgDirect2D32.lib")
#endif


size_t test_inject(fgRoot* self, const FG_Msg* msg)
{
  if(msg->type == FG_KEYDOWN)
  {
    if(msg->keycode == FG_KEY_F11)
    {
      if(fgDebug_Get() != 0 && !(fgDebug_Get()->tabs.control.element.flags&FGELEMENT_HIDDEN))
        fgDebug_Hide();
      else
        fgDebug_Show(0, 1);
    }
  }
  return fgRoot_DefaultInject(self, msg);
}

int main(int argc, char** argv)
{
  static const int COLUMNS[3] = { 24, 11, 8 };
  static TESTDEF tests[] = {
    { "feathergui.h", &test_feathergui },
    { "fgRoot.h", &test_Root },
    { "fgWindow.h", &test_Window },
    { "fgButton.h", &test_Button },
    { "fgList.h", &test_List },
    { "fgMenu.h", &test_Menu },
  };

  static const size_t NUMTESTS=sizeof(tests)/sizeof(TESTDEF);
  RETPAIR numpassed;
  unsigned int i;
  char buf[44]; // 21 + 21 + 1 + null terminator
  unsigned int failures[sizeof(tests)/sizeof(TESTDEF)];
  unsigned int nfail=0;

  fgInitialize();
  fgSetInjectFunc(&test_inject);
  /*
#if defined(BSS_DEBUG) && defined(BSS_CPU_x86_64)
  fgLoadBackend("fgDirect2D_d.dll");
#elif defined(BSS_CPU_x86_64)
  fgLoadBackend("fgDirect2D.dll");
#elif defined(BSS_DEBUG)
  fgLoadBackend("fgDirect2D32_d.dll");
#else
  fgLoadBackend("fgDirect2D32.dll");
#endif
  */

  srand(time(NULL));
  failedtests=fopen("failedtests.txt","wb");
  
  printf("Feather GUI Abstraction Layer v%i.%i.%i: Unit Tests\nCopyright (c)2017 Black Sphere Studios\n\n",FGUI_VERSION_MAJOR,FGUI_VERSION_MINOR,FGUI_VERSION_REVISION);
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
    printf("\nThese failures indicate either a misconfiguration on your system, or a potential bug.\n\nA detailed list of failed tests was written to failedtests.txt\n");
  }

  //fgUnloadBackend();
  fgSingleton()->backend.fgTerminate();

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