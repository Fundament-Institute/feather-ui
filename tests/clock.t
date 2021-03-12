local f = require 'feather'
local scaffolding = require 'feather.scaffolding'
import 'feather/bind'

local clockring = f.template {
  radius = f.required,
  width = f.required,
  progress = f.required,
  color = `f.Color{0xccccccff},
  pos = f.required
}
{
  f.arc {
    outline = bind f.Color{0},
    color = bind color,
    innerRadius = bind radius - width,
    angles = bind f.Vec {
        array(
          0f,
          progress * [float]([math.pi])
        )
                    },
    ext = bind f.Vec3D{arrayof(float, radius, radius, 0)},
    pos = bind pos
  }
}

local clock = f.template {
  hour = f.required,
  minute = f.required,
  second = f.required,
  radius = f.required,
  pos = f.required
}
{
  clockring {
    radius = bind radius,
    pos = bind pos,
    width = bind radius * 0.1f,
    progress = bind second /60.0f
  },
  clockring {
    radius = bind radius * 0.85f,
    pos = bind pos,
    width = bind radius * 0.1f,
    progress = bind minute / 60.0f
  },
  clockring {
    radius = bind radius * 0.7f,
    pos = bind pos,
    width = bind radius * 0.1f,
    progress = bind hour / 24.0f
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

local ui = f.ui {
  application = &clockapp,
  queries = {},
  f.window {
    clock {
      hour = bind app.time.tm_hour,
      minute = bind app.time.tm_min,
      second = bind app.time.tm_sec,
      radius = bind 200,
      pos = bind f.Vec3D{arrayof(float, 250, 250, 0)}
    }
  }
}


return scaffolding.simple_application("clock", clockapp, ui, {})
