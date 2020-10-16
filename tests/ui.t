local f = require 'feather'
local shared = require 'feather.shared'
local C = require 'feather.libc'

local ui = f.ui {
  queries = {},
  application = terralib.types.unit,
  f.window {
    f.rect {
      pos = function(env) return `[f.Vec3D]{array(20f, 20f, 0f)} end,
      ext = function(env) return `[f.Vec3D]{array(40f, 40f, 0f)} end,
      color = function(env) return `[f.Color]{0x888888ff} end
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

  b = f.Backend.new(&u, [f.Behavior](ui.behavior), FakeLog, dllpath, nil)._0
  if b == nil then
    return 1
  end
  u:init({}, [ui.query_store]{}, b)

  u:enter()
  while b:ProcessMessages() ~= 0 do
    u:update()
    -- u:render()
  end
  u:exit()

  u:destruct()
  return 0
end

local terra main(argc: int, argv: &rawstring)
  run_app(argv[1])
end

print(ui.methods.render)

terralib.saveobj("ui_test", {main = main}, {"-ldl", "-g"})
