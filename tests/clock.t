local f = require 'feather'
local scaffolding = require 'feather.scaffolding'
local Iterator = require 'std.iterator'
import 'feather/bind'

local clockring = f.template {
  radius = f.required,
  width = f.required,
  progress = f.required,
  pos = f.required
}
{
  f.arc {
    outline = bind f.Color{0},
    color = f.getcontext.clockcolor,
    innerRadius = bind radius - width,
    angles = bind f.Vec {
        array(
          0f,
          progress * [float]([math.pi * 2.0])
        )
                    },
    ext = bind f.Vec3{arrayof(float, radius, radius, 0)},
    pos = bind pos
  }
}

local clock = f.template {
  times = f.required,
  radius = f.required,
  pos = f.required
}
{
  f.each "times" (bind times) {
    f.let {
      radius = bind (radius * 1.0f) * times._1
    }
    {
      clockring {
        radius = bind radius,
        pos = bind pos,
        width = bind radius * 0.1f,
        progress = bind times._0
      }
    }
  }
}

local C = terralib.includecstring [[
#include <time.h>

time_t feathertime(time_t* const _Time) { return time(_Time); }
#if defined(WIN32) || defined(_WIN32) || defined(_WIN64)
struct tm *feathergmtime(const time_t *timep, struct tm *result) { gmtime_s(result, timep); return result; }
#endif
]]

local struct clockapp {
  time: C.tm
}

terra clockapp:update()
  var rawtime = C.feathertime(nil)
  [jit.os == "Windows" and C.feathergmtime or C.gmtime_r](&rawtime, &self.time)
end
terra clockapp:init(argc: int, argv: &rawstring)
  self:update()
end
terra clockapp:exit() end

local TimePair = tuple(float, float)
local Triplet = tuple(TimePair, TimePair, TimePair)
Triplet.metamethods.__for = Iterator.Tuple
Triplet.elem = TimePair

local ui = f.ui {
  application = &clockapp,
  queries = {},
  f.letcontext {
    clockcolor = bind `f.Color{0xccccccff}
  }
  {
    f.window {
      clock {
        times = bind Triplet{
          TimePair{app.time.tm_hour/24.0f, 0.7f},
          TimePair{app.time.tm_min/60.0f, 0.85f},
          TimePair{app.time.tm_sec/60.0f, 1.0f}
        },
        radius = bind 200,
        pos = bind f.Vec3{arrayof(float, 250, 250, 0)}
      }
    }
  }
}


return scaffolding.simple_application("clock", clockapp, ui, {})
