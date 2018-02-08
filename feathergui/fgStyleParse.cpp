// Copyright ©2018 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#include "feathercpp.h"
#include "fgStyle.h"
#include "bss-util/Array.h"
#include <ctype.h>
#include <stdlib.h>
#include <malloc.h>

const char* fgStyle_ParseFont(const char* font, char quote, int* size, short* weight, char* italic)
{
  *size = 14; // Set defaults
  *weight = 400;
  *italic = 0;
  if(!font) return 0;
  const char* s = strchr(font, ' ');
  if(!s) return font;
  while(isspace(*++s));
  if(!s[0]) return font;
  *size = (int)strtol(font, 0, 10);

  if(s[0] == quote) return s;
  const char* f = strchr(s, ' ');
  size_t slen = f - s;
  if(!f || f[-1] == ',') return s;
  while(isspace(*++f));
  if(!f[0]) return s;

  if(!STRNICMP(s, "bold", slen))
    *weight = 700;
  else if(!STRNICMP(s, "bolder", slen))
    *weight = 900;
  else if(!STRNICMP(s, "normal", slen))
    *weight = 400;
  else if(!STRNICMP(s, "light", slen))
    *weight = 300;
  else if(!STRNICMP(s, "lighter", slen))
    *weight = 100;
  else if(!STRNICMP(s, "italic", slen))
    *italic = true;
  else
    *weight = (short)strtol(s, 0, 10);

  if(f[0] == quote) return f;
  const char* g = strchr(f, ' ');
  size_t flen = g - f;
  if(!g || g[-1] == ',') return f;
  while(isspace(*++g));
  if(!g[0]) return f;

  if(!STRNICMP(f, "italic", flen))
    *italic = true;

  return g;
}

int fgStyle_ParseUnit(const char* str, size_t len)
{
  int flags = FGUNIT_DP;
  if(!len) len = strlen(str);
  if(len > 2)
  {
    if(str[len - 2] == 'e' && str[len - 1] == 'm')
      flags = FGUNIT_EM;
    else if(str[len - 2] == 's' && str[len - 1] == 'p')
      flags = FGUNIT_SP;
    else if(str[len - 2] == 'p' && str[len - 1] == 'x')
      flags = FGUNIT_PX;
  }

  return flags;
}

int fgStyle_ParseCoord(const char* attribute, size_t len, Coord* coord)
{
  *coord = Coord{ 0, 0 };
  if(!attribute) return 0;
  if(!len) len = strlen(attribute);
  const char* rel = strchr(attribute, ':');
  size_t first = len;
  if(rel)
  {
    coord->rel = (FREL)atof(rel + 1);
    if(attribute[len - 1] == '%') // Check if this is a percentage and scale it accordingly.
      coord->rel *= 0.01f;
    first = attribute - rel;
  }
  else if(attribute[len - 1] == '%')
  {
    coord->rel = (FREL)(atof(attribute) * 0.01);
    return 0;
  }

  coord->abs = (FABS)atof(attribute);
  return fgStyle_ParseUnit(attribute, first);
}
int fgStyle_ParseAbsRect(const char* attribute, AbsRect* r)
{
  size_t len = strlen(attribute) + 1;
  VARARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  const char* s2 = STRTOK(0, ", ", &context);
  const char* s3 = STRTOK(0, ", ", &context);

  r->left = (FABS)atof(s0);
  r->top = (FABS)atof(s1);
  r->right = (FABS)atof(s2);
  r->bottom = (FABS)atof(s3);
  return (fgStyle_ParseUnit(s0, 0) << FGUNIT_LEFT) |
    (fgStyle_ParseUnit(s1, 0) << FGUNIT_TOP) |
    (fgStyle_ParseUnit(s2, 0) << FGUNIT_RIGHT) |
    (fgStyle_ParseUnit(s3, 0) << FGUNIT_BOTTOM);
}

int fgStyle_ParseAbsVec(const char* attribute, AbsVec* r)
{
  size_t len = strlen(attribute) + 1;
  VARARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);

  r->x = (FABS)atof(s0);
  r->y = (FABS)atof(s1);
  return (fgStyle_ParseUnit(s0, 0) << FGUNIT_X) |
    (fgStyle_ParseUnit(s1, 0) << FGUNIT_Y);
}

int fgStyle_ParseCRect(const char* attribute, CRect* r)
{
  size_t len = strlen(attribute) + 1;
  VARARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  const char* s2 = STRTOK(0, ", ", &context);
  const char* s3 = STRTOK(0, ", ", &context);

  return (fgStyle_ParseCoord(s0, 0, &r->left) << FGUNIT_LEFT) |
    (fgStyle_ParseCoord(s1, 0, &r->top) << FGUNIT_TOP) |
    (fgStyle_ParseCoord(s2, 0, &r->right) << FGUNIT_RIGHT) |
    (fgStyle_ParseCoord(s3, 0, &r->bottom) << FGUNIT_BOTTOM);
}

int fgStyle_ParseCVec(const char* attribute, CVec* v)
{
  size_t len = strlen(attribute) + 1;
  VARARRAY(char, buf, len);
  MEMCPY(buf, len, attribute, len);
  char* context;
  const char* s0 = STRTOK(buf, ", ", &context);
  const char* s1 = STRTOK(0, ", ", &context);
  return (fgStyle_ParseCoord(s0, 0, &v->x) << FGUNIT_X) | (fgStyle_ParseCoord(s1, 0, &v->y) << FGUNIT_Y);
}