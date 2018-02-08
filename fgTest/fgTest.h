// Copyright ©2018 Black Sphere Studios
// FgNull - A null-implementation of Feather GUI
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#ifndef __FG_TEST_H__
#define __FG_TEST_H__

#include <stddef.h>

#define FGTEST_VERSION_MAJOR 0
#define FGTEST_VERSION_MINOR 1
#define FGTEST_VERSION_REVISION 0

#define MAKESTRING2(x) #x
#define MAKESTRING(x) MAKESTRING2(x)
#define BEGINTEST RETPAIR __testret = {0,0}
#define ENDTEST return __testret
#define FAILEDTEST(t,num) fprintf(failedtests, "Test #%zu Failed  < %s >\n", num,MAKESTRING(t))
#define TEST(t) { ++__testret.first; if(t) ++__testret.second; else FAILEDTEST(t, __testret.first); }
#define TESTP(t) { ++__testret->first; if(t) ++__testret->second; else FAILEDTEST(t, __testret->first); }
//#define TESTARRAY(t,f) _ITERFUNC(__testret,t,[&](uint i) -> bool { f });
//#define TESTALL(t,f) _ITERALL(__testret,t,[&](uint i) -> bool { f });
//#define TESTCOUNT(c,t) { for(uint i = 0; i < c; ++i) TEST(t) }
//#define TESTCOUNTALL(c,t) { bool __val=true; for(uint i = 0; i < c; ++i) __val=__val&&(t); TEST(__val); }

typedef struct {
  size_t first;
  size_t second;
} RETPAIR;

typedef struct
{
  const char* NAME;
  RETPAIR(*FUNC)();
} TESTDEF;

#endif
