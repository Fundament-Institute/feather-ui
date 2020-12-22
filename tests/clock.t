local f = require 'feather'
local shared = require 'feather.shared'

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
#include <stdio.h>
#include <time.h>
]]

local struct clockapp {
  time: C.tm
                      }

terra clockapp:update()
  var rawtime = C.time(nil)
  C.gmtime_r(&rawtime, &self.time)
end

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


local va_start = terralib.intrinsic("llvm.va_start", {&int8} -> {})
local va_end = terralib.intrinsic("llvm.va_end", {&int8} -> {})
terra FakeLog(root : &opaque, level : L.Level, f : shared.conststring, ...) : {}
  if level.val >= 0 then
    C.printf(L.Levels[level.val])
  end
  var vl : C.va_list
  va_start([&int8](&vl))
  C.vprintf(f, vl)
  va_end([&int8](&vl))
  C.printf("\n");
end

local terra run_app(dllpath: rawstring)
  var u: ui
  var b: &f.Backend
  var a = clockapp {
    C.tm{}
  }

  b = f.Backend.new(&u, [f.Behavior](ui.behavior), FakeLog, dllpath, nil)._0
  if b == nil then
    return 1
  end
  u:init(&a, [ui.query_store]{}, b)

  u:enter()
  while b:ProcessMessages() ~= 0 do
    a:update()
    u:update()
    b:DirtyRect(u.data._0.window, nil)
  end
  u:exit()

  u:destruct()
  return 0
end

local terra main(argc: int, argv: &rawstring)
  run_app(argv[1])
end

print(ui.methods.update)

local targetname = "clock_test"
local clangargs = { }

for i,v in ipairs(arg) do
  if v == "-debug" or v == "-g" then
    targetname = targetname .. "_d"
    table.insert(clangargs, "-g")
  end
end

if jit.os == "Windows" then
  targetname = "bin-"..jit.arch.."/".. targetname .. ".exe"
end

if jit.os == "Linux" then
  table.insert(clangargs, "-ldl")
end

table.insert(clangargs, "-lm")

if terralib.saveobj(targetname, "executable", {main = main}, clangargs) ~= nil then
  return -1
end
