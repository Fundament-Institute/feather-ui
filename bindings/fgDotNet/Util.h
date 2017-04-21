// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include <vcclr.h>
#include <string>

using namespace System::Runtime::InteropServices;
using namespace System::Text;

namespace fgDotNet {
  inline std::string StringToUTF8(System::String^ s)
  {
    array<System::Byte>^ bytes = Encoding::UTF8->GetBytes(s);
    std::string str;
    str.resize(bytes->Length);
    Marshal::Copy(bytes, 0, System::IntPtr(const_cast<char*>(str.data())), bytes->Length);
    return str;
  }
}
