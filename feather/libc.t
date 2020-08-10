return terralib.includecstring [[
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <malloc.h>

#if !(defined(WIN32) || defined(_WIN32) || defined(_WIN64) || defined(__TOS_WFG__) || defined(__WINDOWS__))
#include <alloca.h>
#endif

// Windows defines alloca to _alloca but terra can't process defines.
void *fg_alloca(size_t size) { return alloca(size); }

]]
