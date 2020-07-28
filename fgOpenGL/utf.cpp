#include "utf.h"

/*
 * Copyright 2001-2004 Unicode, Inc.
 *
 * Disclaimer
 *
 * This source code is provided as is by Unicode, Inc. No claims are
 * made as to fitness for any particular purpose. No warranties of any
 * kind are expressed or implied. The recipient agrees to determine
 * applicability of information provided. If this file has been
 * purchased on magnetic or optical media from Unicode, Inc., the
 * sole remedy for any claim will be exchange of defective media
 * within 90 days of receipt.
 *
 * Limitations on Rights to Redistribute This Code
 *
 * Unicode, Inc. hereby grants the right to freely use the information
 * supplied in this file in the creation of products supporting the
 * Unicode Standard, and to make copies of this file in any form
 * for internal or external distribution as long as this notice
 * remains attached.
 */

/* ---------------------------------------------------------------------
Conversions between UTF32, UTF-16, and UTF-8. Source code file.
Author: Mark E. Davis, 1994.
Rev History: Rick McGowan, fixes & updates May 2001.
Sept 2001: fixed const & error conditions per
mods suggested by S. Parent & A. Lillich.
June 2002: Tim Dodd added detection and handling of incomplete
source sequences, enhanced error detection, added casts
to eliminate compiler warnings.
July 2003: slight mods to back out aggressive FFFE detection.
Jan 2004: updated switches in from-UTF8 conversions.
Oct 2004: updated to use UNI_MAX_LEGAL_UTF32 in UTF-32 conversions.
See the header file "ConvertUTF.h" for complete documentation.
------------------------------------------------------------------------ */

typedef unsigned int UTF32;
typedef unsigned short UTF16;
typedef unsigned char UTF8;

static const UTF32 halfBase = 0x0010000UL;
static const UTF32 halfMask = 0x3FFUL;

#define UNI_REPLACEMENT_CHAR (UTF32)0x0000FFFD
#define UNI_MAX_BMP          (UTF32)0x0000FFFF
#define UNI_MAX_UTF16        (UTF32)0x0010FFFF
#define UNI_MAX_UTF32        (UTF32)0x7FFFFFFF
#define UNI_MAX_LEGAL_UTF32  (UTF32)0x0010FFFF
#define UNI_SUR_HIGH_START   (UTF32)0xD800
#define UNI_SUR_HIGH_END     (UTF32)0xDBFF
#define UNI_SUR_LOW_START    (UTF32)0xDC00
#define UNI_SUR_LOW_END      (UTF32)0xDFFF
#define false 0
#define true 1

/*
 * Index into the table below with the first byte of a UTF-8 sequence to
 * get the number of trailing bytes that are supposed to follow it.
 * Note that *legal* UTF-8 values can't have 4 or 5-bytes. The table is
 * left as-is for anyone who may want to do such conversion, which was
 * allowed in earlier algorithms.
 */
static const char trailingBytesForUTF8[256] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5
};

/*
 * Magic values subtracted from a buffer value during UTF8 conversion.
 * This table contains as many values as there might be trailing bytes
 * in a UTF-8 sequence.
 */
static const UTF32 offsetsFromUTF8[6] = {
  0x00000000UL, 0x00003080UL, 0x000E2080UL, 0x03C82080UL, 0xFA082080UL, 0x82082080UL
};

/*
 * Utility routine to tell whether a sequence of bytes is legal UTF-8.
 * This must be called with the length pre-determined by the first byte.
 * If not calling this from ConvertUTF8to*, then the length can be set by:
 *  length = trailingBytesForUTF8[*source]+1;
 * and the sequence is illegal right away if there aren't that many bytes
 * available.
 * If presented with a length > 4, this returns false.  The Unicode
 * definition of UTF-8 goes up to 4-byte sequences.
 */

char isLegalUTF8(const UTF8* source, int length)
{
  UTF8 a;
  const UTF8* srcptr = source + length;
  switch(length)
  {
  default:
    return false;
    /* Everything else falls through when "true"... */
  case 4:
    if((a = (*--srcptr)) < 0x80 || a > 0xBF)
      return false;
  case 3:
    if((a = (*--srcptr)) < 0x80 || a > 0xBF)
      return false;
  case 2:
    if((a = (*--srcptr)) < 0x80 || a > 0xBF)
      return false;

    switch(*source)
    {
      /* no fall-through in this inner switch */
    case 0xE0:
      if(a < 0xA0)
        return false;
      break;
    case 0xED:
      if(a > 0x9F)
        return false;
      break;
    case 0xF0:
      if(a < 0x90)
        return false;
      break;
    case 0xF4:
      if(a > 0x8F)
        return false;
      break;
    default:
      if(a < 0x80)
        return false;
    }

  case 1:
    if(*source >= 0x80 && *source < 0xC2)
      return false;
  }
  if(*source > 0xF4)
    return false;
  return true;
}

size_t UTF8toUTF32(const char* FG_RESTRICT input, ptrdiff_t srclen, char32_t* FG_RESTRICT output,
                                 size_t buflen)
{
  if(!srclen)
    return 0;
  const UTF8* source = (UTF8*)input;
  UTF32* target      = (UTF32*)output;
  UTF32* targetEnd   = target + buflen;
  if(srclen < 0)
    srclen = PTRDIFF_MAX;
  buflen = 1;
  while(*input && (input - (const char*)source) < srclen)
  {
    buflen += (((*input) & 0b11000000) != 0b10000000);
    ++input;
  }
  if(!output)
    return buflen;
  const UTF8* sourceEnd = ((input - (const char*)source) < srclen) ? (UTF8*)(++input) : (source + srclen);

  while(source < sourceEnd)
  {
    UTF32 ch                        = 0;
    unsigned short extraBytesToRead = trailingBytesForUTF8[*source];
    if(extraBytesToRead >= sourceEnd - source)
    {
      break;
    }
    /* Do this check whether lenient or strict */
    if(!isLegalUTF8(source, extraBytesToRead + 1))
    {
      break;
    }
    /*
     * The cases all fall through. See "Note A" below.
     */
    switch(extraBytesToRead)
    {
    case 5: ch += *source++; ch <<= 6;
    case 4: ch += *source++; ch <<= 6;
    case 3: ch += *source++; ch <<= 6;
    case 2: ch += *source++; ch <<= 6;
    case 1: ch += *source++; ch <<= 6;
    case 0: ch += *source++;
    }
    ch -= offsetsFromUTF8[extraBytesToRead];

    if(target >= targetEnd)
    {
      source -= (extraBytesToRead + 1); /* Back up the source pointer! */
      break;
    }
    if(ch <= UNI_MAX_LEGAL_UTF32)
    {
      /*
       * UTF-16 surrogate values are illegal in UTF-32, and anything
       * over Plane 17 (> 0x10FFFF) is illegal.
       */
      if(ch >= UNI_SUR_HIGH_START && ch <= UNI_SUR_LOW_END)
      {
        // if(flags == strictConversion)
        //{
        //  source -= (extraBytesToRead + 1); /* return to the illegal value itself */
        //  result = -1;
        //  break;
        //}
        // else
        //{
        *target++ = UNI_REPLACEMENT_CHAR;
        //}
      }
      else
      {
        *target++ = ch;
      }
    }
    else
    { /* i.e., ch > UNI_MAX_LEGAL_UTF32 */
      *target++ = UNI_REPLACEMENT_CHAR;
    }
  }
  return (size_t)(((char32_t*)target) - output);
}