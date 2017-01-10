// Copyright ©2017 Black Sphere Studios
// For conditions of distribution and use, see copyright notice in "feathergui.h"

#pragma once
#include "fgMonitor.h"
#include "Element.h"

namespace fgDotNet {
  public ref class Monitor : Element
  {
  public:
    inline Monitor(fgMonitor* p) : Element(&p->element) {}
    property System::Drawing::RectangleF Coverage {
      System::Drawing::RectangleF get() {
        AbsRect& r = ((fgMonitor*)_p)->coverage;
        return System::Drawing::RectangleF(r.left, r.top, r.right - r.left, r.bottom - r.top);
      }
    }
    property System::Drawing::Point Dpi {
      System::Drawing::Point get() {
        fgIntVec& r = ((fgMonitor*)_p)->dpi;
        return System::Drawing::Point(r.x, r.y);
      }
    }
    property Monitor^ Next {
      Monitor^ get() { return GenNewManagedPtr<Monitor, fgMonitor>(((fgMonitor*)_p)->mnext); }
    }
    property Monitor^ Prev {
      Monitor^ get() { return GenNewManagedPtr<Monitor, fgMonitor>(((fgMonitor*)_p)->mprev); }
    }
  };
}