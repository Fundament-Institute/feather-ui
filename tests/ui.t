local f = require 'feather'

local ui = f.ui {
  queries = {},
  application = terralib.types.unit,
  f.window {
    f.rect {
      pos = `[f.Vec3D]{array(20f, 20f, 0f)},
      ext = `[f.Vec3D]{array(40f, 40f, 0f)},
      color = `[f.Color]{0x888888ff}
    }
  }
}

terra FakeLog(root : &opaque, level : L.Level, f : F.conststring, ...) : {}
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

  b = f.Backend.new(&u, [&f.Behavior](ui.behavior), FakeLog, dllpath, "test backend")
  if b == nil then
    return 1
  end
  u:init({}, {}, b)

  u:enter()
  while b:ProcessMessages() ~= 0 do
    u:update()
    u:render()
  end
  u:exit()

  u:destruct()
  return 0
end

local terra main(argc: int, argv: &rawstring)
  run_app(argv[1])
end

terralib.saveobj("ui_test", {main = main})
