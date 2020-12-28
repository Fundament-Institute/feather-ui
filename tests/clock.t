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
  f.circle {
    outline = bind color,
    color = bind f.Color{0},
    innerRadius = bind radius - width,
    innerBorder = bind width,
    angles = bind f.Vec {
        array(
          0f,
          progress * [float]([math.pi])
        )
                    },
    ext = bind f.Vec3D{arrayof(float, radius * 2, radius * 2, 0)},
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
]]

local struct clockapp {
  time: C.tm
                      }

terra clockapp:update()
  var rawtime = C.time(nil)
  C.gmtime_r(&rawtime, &self.time)
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
      pos = bind f.Vec3D{arrayof(float, 500, 500, 0)}
    }
  }
}


return scaffolding.simple_application("clock", clockapp, ui, {})
