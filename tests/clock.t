local f = require 'feather'
local scaffolding = require 'feather.scaffolding'

local function pass(name)
  return function(env)
    return `[env[name] ]
  end
end
local function K(value)
  return function(env)
    return value
  end
end

local clockring = f.template {
  radius = f.required,
  width = f.required,
  progress = f.required,
  color = `f.Color{0xccccccff},
  pos = f.required
}
{
  f.circle {
    outline = pass "color",
    color = K(`f.Color{0}),
    innerRadius = function(env) return `env.radius - env.width end,
    -- border = pass "width",
    innerBorder = pass "width",
    angles = function(env)
      -- for k, v in pairs(env) do print("circle env", k, v) end
      return `f.Vec {
        array(
          0f,--env.progress * [float]([math.pi])/2f,
          env.progress * [float]([math.pi])
        )
                    }
    end,
    ext = function(env) return `f.Vec3D{arrayof(float, env.radius * 2, env.radius * 2, 0)} end,
    pos = pass "pos"
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
    radius = pass "radius",
    pos = pass "pos",
    width = function(env) return `env.radius * 0.1f end,
    progress = function(env) return `env.second /60.0f end
  },
  clockring {
    radius = function(env) return `env.radius * 0.85f end,
    pos = pass "pos",
    width = function(env) return `env.radius * 0.1f end,
    progress = function(env) return `env.minute / 60.0f end
  },
  clockring {
    radius = function(env) return `env.radius * 0.7f end,
    pos = pass "pos",
    width = function(env) return `env.radius * 0.1f end,
    progress = function(env) return `env.hour / 24.0f end
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
      hour = function(env) return `env.app.time.tm_hour end,
      minute = function(env) return `env.app.time.tm_min end,
      second = function(env) return `env.app.time.tm_sec end,
      radius = K(`200),
      pos = K(`f.Vec3D{arrayof(float, 500, 500, 0)})
    }
  }
}


return scaffolding.simple_application("clock", clockapp, ui, {})
