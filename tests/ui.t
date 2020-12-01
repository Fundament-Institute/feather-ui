local f = require 'feather'
local shared = require 'feather.shared'
local C = require 'feather.libc'
terralib.fulltrace = true

local struct app {
  pos: f.Vec3D
                 }

local ui = f.ui {
  queries = {},
  application = &app,
  f.window {
    f.rect {
      pos = function(env) return `[env.app].pos end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
    },
    f.triangle {
      pos = function(env) return `[env.app].pos + [f.Vec3D]{array(50f, 50f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
    },
    f.circle {
      pos = function(env) return `[env.app].pos + [f.Vec3D]{array(50f, 0f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
    },
    f.mousearea {
      pos = function(env) return `[env.app].pos + [f.Vec3D]{array(100f, 100f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end
    },
  },
  f.window {
    f.rect {
      pos = function(env) return `[env.app].pos end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
    },
    f.triangle {
      pos = function(env) return `[env.app].pos + [f.Vec3D]{array(50f, 50f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
    },
    f.circle {
      pos = function(env) return `[env.app].pos + [f.Vec3D]{array(50f, 0f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      color = function(env) return `[f.Color]{0xffffffff} end
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
  var a = app {
    f.Vec3D{array(60f, 60f, 0f)}
  }

  b = f.Backend.new(&u, [f.Behavior](ui.behavior), FakeLog, dllpath, nil)._0
  if b == nil then
    return 1
  end
  u:init(&a, [ui.query_store]{}, b)

  u:enter()
  while b:ProcessMessages() ~= 0 do
    u:update()
    b:DirtyRect(u.data._0.window, nil)
    b:DirtyRect(u.data._1.window, nil)
  end
  u:exit()

  u:destruct()
  return 0
end

local terra main(argc: int, argv: &rawstring)
  run_app(argv[1])
end

local targetname = "ui_test"
local clangargs = { }

for i,v in ipairs(arg) do
  if v == "-debug" or v == "-g" then
    targetname = targetname .. "_d"
    table.insert(clangargs, "-g")
    if jit.os == "Windows" then
      table.insert(clangargs, "/DEBUG")
    end
  end
end

if jit.os == "Windows" then
  targetname = "bin-"..jit.arch.."/".. targetname .. ".exe"
end

if jit.os == "Linux" then
  table.insert(clangargs, "-ldl")
end

if terralib.saveobj(targetname, "executable", {main = main}, clangargs) ~= nil then
  return -1
end


