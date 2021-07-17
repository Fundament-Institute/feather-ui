local C = require 'feather.libc'

local DynArray = require 'feather.dynarray'
local F = require 'feather.shared'
local B = require 'feather.backend'
local Enum = require 'feather.enum'
local Flags = require 'feather.flags'
local Msg = require 'feather.message'
local Messages = require 'feather.messages'
local Hash = require 'feather.hash'
local cond = require 'std.cond'
local OS = require 'feather.os'
local Constraint = require 'std.constraint'
local List = require 'terralist'
local Object = require 'std.object'
local Alloc = require 'std.alloc'
local RTree = require 'feather.rtree'
local Math = require 'std.math'
local Virtual = require 'feather.virtual'
local Log = require 'feather.log'
local Partitioner = require 'feather.partitioner'
local Transform = require 'feather.transform'

--local String = require 'feather.string'

local backends = {}
local isdebug = false
for i,v in ipairs(arg) do
  if v == "-debug" or v == "-g" then
    isdebug = true
  else
    table.insert(backends, v)
  end
end

local struct TestHarness {
  passed : int;
  total : int;
}

TestHarness.methods.Test = Constraint(function(a) return Constraint.MetaMethod(TestHarness, {a, Constraint.Cast(a), Constraint.Optional(Constraint.Rawstring)}, nil,
function(self, result, expected, op)
  op = op or "__eq"
  return quote
    self.total = self.total + 1
    var r = result
    var e = expected
    if operator(op, r, e) then
      self.passed = self.passed + 1
    else
      --s = String:append("Test Failed, expected ", e, " but got ", r, ": ", [ result:prettystring() ])
      --buf = s:torawstring()
      --C.printf(bf)
      
      C.printf("Test Failed: %s", [ result:prettystring() ])
    end
  end
end) end, Constraint.TerraType)

terra TestHarness:dynarray()
  var a : DynArray(int)
  a:init()
  a:add(1)
  self:Test(a(0), 1)
  self:Test(a.size, 1)
  a:reserve(5)
  self:Test(a.capacity, 5)
  self:Test(a.size, 1)
  a:add(2)
  self:Test(a.size, 2)
  a:add(3)
  self:Test(a.size, 3)
  a:add(4)
  self:Test(a.size, 4)
  a:insert(5, 4)
  self:Test(a.size, 5)
  self:Test(a(1), 2)
  self:Test(a(2), 3)
  self:Test(a(3), 4)
  self:Test(a(4), 5)
  a:insert(0, 0)
  self:Test(a.size, 6)
  self:Test(a.capacity, 6, "__ge")
  self:Test(a(0), 0)
  self:Test(a(1), 1)
  self:Test(a(2), 2)
  self:Test(a(3), 3)
  self:Test(a(4), 4)
  self:Test(a(5), 5)

  var t : int = 0
  for v in a do
    t = t + v
  end
  self:Test(t, 15)
  
  a:clear()
  self:Test(a.size, 0)
  a:add(1)
  self:Test(a.size, 1)
  self:Test(a(0), 1)
  a:destruct()
end

local va_start = terralib.intrinsic("llvm.va_start", {&int8} -> {})
local va_end = terralib.intrinsic("llvm.va_end", {&int8} -> {})

local UIMock = struct{}
terra FakeLog(root : &opaque, level : Log.Level, f : F.conststring, ...) : {}
  if level.val >= 0 then
    C.printf(Log.Levels[level.val])
  end
  var vl : C.va_list
  va_start([&int8](&vl))
  C.vprintf(f, vl)
  va_end([&int8](&vl))
  C.printf("\n");
end

struct MockMsgReceiver {
  image : &B.Asset
  font : &B.Font
  layout : &opaque
  layer : &B.Asset
  flags : uint64
  close : bool
}

Virtual.extends(Msg.Receiver)(MockMsgReceiver)

SetShape = macro(function(c, kind, area, fill, border, outline, blur)
  local r = symbol(F.Rect)
  return quote
    var r = [area]
    c.category          = [kind]
    c.shape.area        = &r
    c.shape.border      = [border]
    c.shape.blur        = [blur]
    c.shape.fillColor   = [fill]
    c.shape.borderColor = [outline]
    c.shape.z           = 0.0f
    c.shape.asset       = nil
  end
end)

terra MockMsgReceiver:Behavior(w : &Msg.Window, ui : &opaque, m : &Msg.Message) : Msg.Result
  if m.kind.val == [Msg.Receiver.virtualinfo.info["Draw"].index] then
    var b = @[&&B.Backend](ui)
    b:Clear(w, 0x00000000U)
    var list : B.Command[6]

    var c = F.rect(0f, 4f, 8f, 12f)
    SetShape(list[0], B.Category.RECT, F.rect(50f, 100f, 200f, 300f), 0xFF0000FF, 5f, 0xFF00FFFF, 0f)
    list[0].shape.rect.corners = &c
    list[0].shape.rect.rotate = 0.0f
    
    SetShape(list[1], B.Category.ARC, F.rect(350f, 100f, 500f, 300f), 0xFF0000FF, 5f, 0xFF00FFFF, 0f)
    list[1].shape.arc.angles = F.vec(0f, 3f)
    list[1].shape.arc.innerRadius = 10.0f

    SetShape(list[2], B.Category.CIRCLE, F.rect(250f, 150f, 300f, 200f), 0xFFFFFFFF, 0f, 0xFFFFFFFF, 0f)
    list[2].shape.circle.innerRadius = 10.0f
    list[2].shape.circle.innerBorder = 2.0f
    
    var c3 = F.rect(0f, 4f, 8f, 0.5f)
    SetShape(list[3], B.Category.TRIANGLE, F.rect(150f, 300f, 300f, 500f), 0xFF0000FF, 5f, 0xFF00FFFF, 0f)
    list[3].shape.triangle.corners = &c3
    list[3].shape.triangle.rotate = 0.45f

    list[4].category    = B.Category.ASSET
    var r4 = F.rect(300f, 300f, 620f, 580f)
    list[4].asset.area  = &r4
    list[4].asset.asset = self.image
    list[4].asset.color = 0xFFFFFFFF
    list[4].asset.source = nil
    list[4].asset.rotate = 0.0f
    list[4].asset.z = 0.0f

    list[5].category    = B.Category.TEXT
    var r5 = F.rect(200f, 10f, 550f, 40f)
    list[5].text.area  = &r5
    list[5].text.font = self.font
    list[5].text.layout = self.layout
    list[5].text.color = 0xFFFFFFFF
    list[5].text.rotate = 0.0f
    list[5].text.z = 0.0f
    
    b:Draw(w, list, 6, nil)

    if self.layer ~= nil then -- On the initial window creation the layer won't exist
      var transform = array(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 550.0f, 50.0f, 0.0f, 1.0f);
      b:PushLayer(w, self.layer, transform, 0.5f, nil);
      b:Clear(w, 0x00000000U)
      var lshape : B.Command 
      SetShape(lshape, B.Category.RECT, F.rect(0.0f, 0.0f, 100.0f, 80.0f), 0xFFFF0000U, 5.0f, 0xFFFFFF00U, 0.0f)
      lshape.shape.rect.corners = &c
      lshape.shape.rect.rotate = 0.0f
      b:Draw(w, &lshape, 1, nil)
      b:PopLayer(w);
    end

    return Msg.Result{0}
  end
  if m.kind.val == [Msg.Receiver.virtualinfo.info["GetWindowFlags"].index] then
    if self.close then
      return Msg.Result{Messages.WindowFlag.CLOSED}
    else
      return Msg.Result{0}
    end
  end
  if m.kind.val == [Msg.Receiver.virtualinfo.info["SetWindowFlags"].index] then
    self.close = self.close or ((m.setWindowFlags.flags and [Messages.WindowFlag.enum_values.CLOSED]) ~= 0)
  end
  if (m.kind.val == [Msg.Receiver.virtualinfo.info["KeyDown"].index] and m.keyDown.key ~= Messages.Keys.LMENU.val and m.keyDown.scancode ~= 84) or m.kind.val == [Msg.Receiver.virtualinfo.info["MouseDown"].index] then
    self.close = true
  end
  
  return Msg.DefaultBehavior(&self.super, w, ui, m)
end

function dumpfile(path)
  local f = io.open(path, "rb")
  local dump = f:read("*all")
  f:close()
  return dump
end

local TEST_TEXT = constant("testtext")
local example_png = dumpfile("tests/example.png")

terra TestHarness:TestBackend(dllpath : rawstring, aa : B.AntiAliasing)
  var fakeUI : &B.Backend = nil

  C.printf("Loading Backend: %s\n", dllpath) 
  var bl = B.Backend.new(&fakeUI, [Msg.Behavior](MockMsgReceiver.Behavior), FakeLog, dllpath, nil)
  self:Test(bl._0 == nil, false)
  if bl._0 == nil then
    return
  end

  fakeUI = bl._0
  var b = bl._0
  var textrect = F.rect(0f,0f,1000f,700f)
  var e = MockMsgReceiver{Msg.Receiver{Msg.Receiver.virtual_initializer}}
  e.flags = Messages.WindowFlag.RESIZABLE
  -- e.image = b:CreateAsset("../tests/example.png", 0, B.Format.PNG)
  e.image = b:CreateAsset(nil, [constant(`example_png)], [#example_png], B.Format.PNG, 0)
  e.font = b:CreateFont("Arial", 700, false, 16, F.vec(96f, 96f), aa)
  e.layout = b:FontLayout(e.font, "Example Text!", &textrect, 16f, 0f, B.BreakStyle.NONE, nil);
  e.layer = nil
  e.close = false

  var pos = F.vec(200f,100f)
  var dim = F.vec(800f,600f)
  var w = b:CreateWindow(&e.super, nil, &pos, &dim, "Feather Test", e.flags)
  self:Test(w ~= nil, true)

  var layerdim = F.vec(200.0f,80.0f)
  e.layer = b:CreateLayer(w, &layerdim, 0)
  self:Test(e.layer ~= nil, true)

  self:Test(b:SetCursor(w, B.Cursor.CROSS), 0)
  self:Test(b:DirtyRect(w, nil), 0)

  self:Test(b:SetWindow(w, nil, nil, nil, nil, nil, Messages.WindowFlag.RESIZABLE), 0)
  self:Test(b:SetWindow(w, &e.super, nil, nil, nil, "Feather Test Changed", Messages.WindowFlag.RESIZABLE), 0)

  self:Test(b:ProcessMessages() ~= 0, true)

  self:Test(w ~= nil, true)
  self:Test(b:ProcessMessages() ~= 0, true)
  self:Test(b:ClearClipboard(w, B.Clipboard.ALL), 0)
  self:Test(b:CheckClipboard(w, B.Clipboard.TEXT), false)
  self:Test(b:CheckClipboard(w, B.Clipboard.WAVE), false)
  self:Test(b:CheckClipboard(w, B.Clipboard.ALL), false)
  self:Test(b:PutClipboard(w, B.Clipboard.TEXT, TEST_TEXT, 9), 0)
  self:Test(b:CheckClipboard(w, B.Clipboard.TEXT), true)
  self:Test(b:CheckClipboard(w, B.Clipboard.WAVE), false)
  self:Test(b:CheckClipboard(w, B.Clipboard.ALL), true)

  var hold : int8[10]

  self:Test(b:GetClipboard(w, B.Clipboard.TEXT, [&int8](hold), 10), 9)
  for i = 0,8 do 
    self:Test(TEST_TEXT[i], hold[i])
  end

  self:Test(b:GetClipboard(w, B.Clipboard.WAVE, [&int8](hold), 10), 0)

  while b:ProcessMessages() ~= 0 and e.close == false do
    b:DirtyRect(w, nil)
  end

  self:Test(b:DestroyAsset(e.layer), 0) -- must be destroyed before the window is destroyed
  self:Test(b:DestroyWindow(w), 0)
  self:Test(b:DestroyAsset(e.image), 0)
  self:Test(b:DestroyLayout(e.layout), 0)
  self:Test(b:DestroyFont(e.font), 0)
  b:free(bl._1)
  self:Test(true, true) -- ensure tests didn't crash
end

terra TestHarness:TestBackends()
  escape 
    for i,v in ipairs(backends) do
      emit(quote self:TestBackend([v], B.AntiAliasing.AA) end)
    end
  end
end

local printraw = macro(function(a) terralib.printraw(a:gettype()) return `a == a end)
local checktype = macro(function(a, b) return a:gettype() == b:astype() end)

terra TestHarness:alloc()
  var test = Alloc.alloc(int8, 5)
  self:Test(checktype(test, [&int8]), true)
  Alloc.clear(test, 5, 64)
  for i = 0,5 do self:Test(test[i], 64) end
  test = Alloc.realloc(test, 5, 10)
  for i = 0,5 do self:Test(test[i], 64) end
  Alloc.copy(test + 5, test, 5)
  for i = 0,10 do self:Test(test[i], 64) end
  Alloc.clear(test, 10, 0)
  for i = 0,10 do self:Test(test[i], 0) end
  Alloc.free(test)
  var test2 = Alloc.calloc(float, 4)
  for i = 0,4 do self:Test(test2[i], 0.0) end
  Alloc.free(test2)
end

terra TestHarness:color()
  var c : F.Color = F.Color{ 0 }
  self:Test(c.r, 0)
  self:Test(c.g, 0)
  self:Test(c.b, 0)
  self:Test(c.a, 0)
  self:Test(c.v, 0)

  c.r = 0xFF
  self:Test(c.r, 0xFF)
  self:Test(c.g, 0)
  self:Test(c.b, 0)
  self:Test(c.a, 0)
  self:Test(c.v, 0x00FF0000)

  c.g = 0xFF
  c.b = 0xFF
  c.a = 0xFF
  self:Test(c.r, 0xFF)
  self:Test(c.g, 0xFF)
  self:Test(c.b, 0xFF)
  self:Test(c.a, 0xFF)
  self:Test(c.v, 0xFFFFFFFF)

  c.r = 0x34
  c.g = 0x56
  c.b = 0x78
  c.a = 0x12
  self:Test(c.r, 0x34)
  self:Test(c.g, 0x56)
  self:Test(c.b, 0x78)
  self:Test(c.a, 0x12)
  self:Test(c.v, 0x12345678)
  
  var c2 = F.Color { 0x12345678 }
  self:Test(c2.r, 0x34)
  self:Test(c2.g, 0x56)
  self:Test(c2.b, 0x78)
  self:Test(c2.a, 0x12)
  self:Test(c2.v, 0x12345678)
end

local TestEnum = Enum{
  "First",
  "second",
  "THIRD",
}

terra TestHarness:enum()
  self:Test(TestEnum.First.val, 0)
  self:Test(TestEnum.second.val, 1)
  self:Test(TestEnum.THIRD.val, 2)

  var e : TestEnum = TestEnum.First
  self:Test(e.val, 0)
  e.val = 0
  e = TestEnum.second
  e.val = 1
  self:Test(e.val, 1)
  e = TestEnum.THIRD
  self:Test(e.val, 2)
  e.val = 2
  self:Test(e, TestEnum.THIRD)
end

local TestFlags = Flags{
  "First",
  "second",
  "THIRD",
}

terra TestHarness:flags()
  self:Test(TestFlags.First.val, 1)
  self:Test(TestFlags.second.val, 2)
  self:Test(TestFlags.THIRD.val, 4)
  var f = TestFlags.First + TestFlags.THIRD
  self:Test(f.First and f.second, false)
  self:Test(f.First, true)
  self:Test(not f.second, true)
  self:Test(f.THIRD, true)
  f.THIRD = false
  f.second = true
  self:Test(f.First and f.second, true)
  self:Test(f.First and f.second and not f.THIRD, true)
end

terra TestHarness:hash()

end

terra TestHarness:message()

end

terra TestHarness:conststring()
  var s : F.conststring
  s.s = "other"
  self:Test(s.s, "other")
  var str : rawstring = "test"
  s = str
  self:Test(s.s, "test")
  s = "foo"
  self:Test(s.s, "foo")
  self:Test(str, "test")
  str = s
  self:Test(str, "foo")

end

terra TestHarness:rect()
  var a = F.rect(1f,2f,3f,4f)
  self:Test(a.l, 1)
  self:Test(a.t, 2)
  self:Test(a.r, 3)
  self:Test(a.b, 4)
  self:Test(a.left, 1)
  self:Test(a.top, 2)
  self:Test(a.right, 3)
  self:Test(a.bottom, 4)
  self:Test(a.ltrb[0], 1)
  self:Test(a.ltrb[1], 2)
  self:Test(a.ltrb[2], 3)
  self:Test(a.ltrb[3], 4)

  var b = F.rect(5f,6f,7f,8f)
  var c = a + b
  self:Test(c.l, 6f)
  self:Test(c.t, 8f)
  self:Test(c.r, 10f)
  self:Test(c.b, 12f)
  c = b - a
  self:Test(c.l, 4f)
  self:Test(c.t, 4f)
  self:Test(c.r, 4f)
  self:Test(c.b, 4f)
  c = a - b
  self:Test(c.l, -4f)
  self:Test(c.t, -4f)
  self:Test(c.r, -4f)
  self:Test(c.b, -4f)
  c = b * a
  self:Test(c.l, 5f)
  self:Test(c.t, 12f)
  self:Test(c.r, 21f)
  self:Test(c.b, 32f)
  c = b / a
  self:Test(c.l, 5f)
  self:Test(c.t, 3f)
  self:Test(c.r, (7f/3f))
  self:Test(c.b, 2f)

  c = a + 1f
  self:Test(c.l, 2f)
  self:Test(c.t, 3f)
  self:Test(c.r, 4f)
  self:Test(c.b, 5f)
  c = 1f + a
  self:Test(c.l, 2f)
  self:Test(c.t, 3f)
  self:Test(c.r, 4f)
  self:Test(c.b, 5f)
  c = 2f * a
  self:Test(c.l, 2f)
  self:Test(c.t, 4f)
  self:Test(c.r, 6f)
  self:Test(c.b, 8f)

  var d = a.topleft + b.bottomright
  self:Test(d.x, 8)
  self:Test(d.y, 10)

  self:Test(a, b, "__ne")
  self:Test(a, a)
  self:Test(a == b, false)
end

terra TestHarness:vec()
  var a = F.veci(1,2)
  self:Test(a.x, 1)
  self:Test(a.y, 2)
  self:Test(a.v[0], 1)
  self:Test(a.v[1], 2)
  a.x = 5
  a.y = 6
  self:Test(a.x, 5)
  self:Test(a.y, 6)

  var b = F.veci(-3,-4)
  var c = a + b
  self:Test(c.x, 2)
  self:Test(c.y, 2)
  c = a - b
  self:Test(c.x, 8)
  self:Test(c.y, 10)

  c = a * -3
  self:Test(c.x, -15)
  self:Test(c.y, -18)
  a.x = 1
  a.y = 2
  c = b / -1
  self:Test(c.x, 3)
  self:Test(c.y, 4)

  c = a + F.veci(3,3)
  self:Test(c.x, 4)
  self:Test(c.y, 5)

  c = F.veci(4,4) - b
  self:Test(c.x, 7)
  self:Test(c.y, 8)

  self:Test(a, b, "__ne")
  b.x = a.x
  b.y = a.y
  self:Test(a, b)
end

do
  local primitives = List{int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, bool, function() end,
    float, double, tuple(), tuple(int, float), int32[8], vector(float, 4), "string", 3.14, {}, {"fake"}, `6, `3.2,
    &float, &&float }

  local constraints = { Constraint.Number, Constraint.String, Constraint.Table, Constraint.LuaValue, Constraint.TerraType,
    Constraint.Field, Constraint.Method, Constraint.Integral, Constraint.Float, Constraint.Pointer, Constraint.Type,
    Constraint.Value, Constraint.Empty, Constraint.BasicConstraint, function(e) return Constraint.BasicConstraint(e, true) end }

  local function testconstraint(target, pred)
    local f = function(v) 
      local ok, err = pcall(target, v)
      if ok ~= pred(v) then
        error ("Expected "..tostring(target).." to return "..tostring(pred(v)).." for "..tostring(v).." but got "..tostring(ok))
      end
    end
    primitives:app(f)
    f(nil)
  end
  
  testconstraint(Constraint.TerraType, function(v) return terralib.types.istype(v) end)
  testconstraint(Constraint.Number, function(v) return type(v) == "number" end)
  testconstraint(Constraint.String, function(v) return type(v) == "string" end)
  testconstraint(Constraint.Table, function(v) return type(v) == "table" end)
  testconstraint(Constraint.LuaValue, function(v) return v ~= nil end)
  testconstraint(Constraint.Integral, function(v) return terralib.types.istype(v) and v:isintegral() end)
  testconstraint(Constraint.Float, function(v) return terralib.types.istype(v) and v:isfloat() end)
  testconstraint(Constraint.Pointer(float, 0), function(v) return v == float end)
  testconstraint(Constraint.Pointer(float), function(v) return v == &float end)
  testconstraint(Constraint.Pointer(float, 2), function(v) return v == &&float end)
  testconstraint(Constraint.Type(Constraint.Integral), function(v) return terralib.types.istype(v) and v:isintegral() end)
  testconstraint(Constraint.Value(Constraint.Integral), function(v) return type(v) == "table" and v.gettype ~= nil and v:gettype():isintegral() end)
  testconstraint(Constraint.Empty, function(v) return v == nil end)
  testconstraint(Constraint.BasicConstraint(uint8), function(v) return v == uint8 end)
  testconstraint(Constraint.BasicConstraint(int8, true), function(v) return v == bool or (terralib.types.istype(v) and v:isarithmetic()) end)

  local function testfunction(params, ret, vararg, expected, fn) 
    local ok, err = pcall(Constraint.Function(params, ret, vararg), fn)
    if ok ~= expected then
      error ("Expected "..tostring(expected).." but got "..tostring(ok).." for "..tostring(params).." : "..tostring(ret))
    end
  end

  testfunction({}, nil, nil, true, terra() end)
  testfunction({}, tuple(), nil, true, terra() end)
  testfunction({}, nil, Constraint.Integral, true, terra() end)
  testfunction({}, nil, nil, true, Constraint.Meta({}, nil, function() end))
  testfunction({}, tuple(), nil, true, Constraint.Meta({}, nil, function() end))
  testfunction({}, nil, Constraint.Integral, true, Constraint.Meta({}, nil, function() end))
  testfunction({}, nil, nil, true, Constraint.Meta({}, tuple(), function() end))
  
  local primtypes = primitives:filter(function(e) return terralib.types.istype(e) end)
  primtypes:app(function(e) return testfunction({e}, nil, nil, true, Constraint.Meta({e}, nil, function() end)) end)
  primtypes:app(function(e) return testfunction({bool}, nil, nil, e == bool, Constraint.Meta({e}, nil, function() end)) end)
  primtypes:app(function(e) return testfunction({bool}, nil, nil, e == bool, terra(asdf: e) end) end)
  
  primtypes:app(function(e) return testfunction({}, e, nil, true, Constraint.Meta({}, e, function() end)) end)
  primtypes:app(function(e) return testfunction({}, bool, nil, e == bool, Constraint.Meta({}, e, function() end)) end)
  primtypes:app(function(e) return testfunction({e}, e, nil, true, Constraint.Meta({e}, e, function() end)) end)
  primtypes:app(function(e) return testfunction({e}, e, nil, true, terra(asdf: e) : e end) end)
  primtypes:app(function(e) return testfunction({bool}, bool, nil, e == bool, Constraint.Meta({e}, e, function() end)) end)
  primtypes:app(function(e) return testfunction({bool}, bool, nil, e == bool, terra(asdf: e) : e end) end)

  testfunction({int, int}, int, nil, false, Constraint.Meta({int}, int, function() end))
  testfunction({int, int}, int, nil, false, Constraint.Meta({int, int}, nil, function() end))
  testfunction({int, int}, int, nil, false, Constraint.Meta({}, int, function() end))
  testfunction({}, int, nil, false, Constraint.Meta({int}, int, function() end))
  testfunction({int}, nil, nil, false, Constraint.Meta({int}, int, function() end))
  testfunction({int, int}, int, nil, false, terra(asdf: int) : int end)
  testfunction({int, int}, int, nil, false, terra(asdf: int, a: int) : {} end)
  testfunction({int, int}, int, nil, false, terra() : int end)
  testfunction({}, int, nil, false, terra(asdf: int) : int end)
  testfunction({int}, nil, nil, false, terra(asdf: int) : int end)
  testfunction({int}, {int, int}, nil, false, terra(asdf: int) : int end)

  primtypes:app(function(e) return testfunction({e, e}, e, nil, true, terra(a: e, b: e) : e end) end)
  primtypes:app(function(e) return testfunction({e, e}, e, nil, true, Constraint.Meta({e, e}, e, function() end)) end)
  primtypes:app(function(e) return testfunction({e, e}, {e, e}, nil, true, terra(a: e, b: e) : {e, e} end) end)

  local function testmethod(params, ret, static, vararg, expected, fn, name)
    name = name or "foobar"
    local s = fn
    if not s.isstruct or not s:isstruct() then
      s = struct{}
      s.methods[name] = fn
    end
    local ok, err = pcall(Constraint.MethodConstraint(name, params, ret, static, vararg), s)
    if ok ~= expected then
      error ("Expected "..tostring(expected).." but got "..tostring(ok).." for "..tostring(params).." : "..tostring(ret))
    end
  end

  function expect(constraint, expect, ...)
    local ok, err = pcall(constraint, ...)
    if ok ~= expect then
      error ("Expected "..tostring(constraint).." to return "..tostring(expect).." for "..tostring(unpack({...})).." but got "..tostring(ok))
    end
  end

  testmethod({}, nil, true, nil, true, terra() end)
  testmethod({}, tuple(), true, nil, true, terra() end)
  testmethod({}, nil, true, Constraint.Integral, true, terra() end)
  testmethod({}, nil, true, nil, true, Constraint.Meta({}, nil, function() end))
  testmethod({}, tuple(), true, nil, true, Constraint.Meta({}, nil, function() end))
  testmethod({}, nil, true, Constraint.Integral, true, Constraint.Meta({}, nil, function() end))
  testmethod({}, nil, true, nil, true, Constraint.Meta({}, tuple(), function() end))

  testmethod({}, nil, true, nil, false, terra(self : int) end)
  testmethod({}, nil, true, nil, false, Constraint.Meta({int}, tuple(), function() end))
  testmethod({}, nil, false, nil, false, terra() end)
  testmethod({}, nil, false, nil, false, Constraint.Meta({}, tuple(), function() end))

  local fiddle = struct{ foobar : float, meta : bool }
  fiddle.methods["terra"] = terra(self : &fiddle) : {} end
  fiddle.methods["ret"] = terra(self : &fiddle) : int end
  fiddle.methods["param"] = terra(self : &fiddle, a : int) : int end
  fiddle.methods["meta"] = Constraint.MetaMethod(fiddle, {}, tuple(), function(self) end)
  fiddle.methods["metaret"] = Constraint.MetaMethod(fiddle, {}, int, function(self) end)
  fiddle.methods["metaparam"] = Constraint.MetaMethod(fiddle, {int}, int, function(self) end)
  
  testmethod({}, nil, false, nil, true, fiddle, "terra")
  testmethod({}, int, false, nil, true, fiddle, "ret")
  testmethod({int}, int, false, nil, true, fiddle, "param")
  testmethod({}, nil, false, nil, true, fiddle, "meta")
  testmethod({}, int, false, nil, true, fiddle, "metaret")
  testmethod({int}, int, false, nil, true, fiddle, "metaparam")
  testmethod({}, int, false, nil, false, fiddle, "metaparam")

  expect(Constraint.Negative("terra", false), true, fiddle)
  List{"foobar", "asdf", ""}:app(function(s) expect(Constraint.Negative(s, true), true, fiddle) end)

  primtypes:app(function(e) return expect(Constraint.Field("meta", e), e == bool, fiddle) end)
  expect(Constraint.Field("foobar", Constraint.Float), true, fiddle)

  local nested = struct{ f : fiddle }

  expect(Constraint.Field("f", Constraint.MakeConstraint(fiddle)), true, nested)
  expect(Constraint.Field("f", Constraint.Field("foobar", float) * Constraint.Field("meta", bool)), true, nested)
  expect(Constraint.Field("f", Constraint.MakeConstraint(float)), false, nested)
  expect(Constraint.Field("f", Constraint.Field("foobar", int) * Constraint.Field("meta", bool)), false, nested)

  testfunction({Constraint.Type(int)}, nil, nil, true, Constraint.Meta({Constraint.Type(int)}, nil, function(a) end))
  testfunction({Constraint.Type(int)}, nil, nil, false, Constraint.Meta({int}, nil, function(a) end))
  testfunction({}, Constraint.Type(int64), nil, true, Constraint.Meta({}, Constraint.Type(int64), function() return int64 end))
  testfunction({}, Constraint.Type(int64), nil, false, Constraint.Meta({}, int64, function() return int64 end))
  
  local topointer = Constraint(function(a) return Constraint.Meta({Constraint.Pointer(a, 0)}, Constraint.Pointer(a, 1), function(e) return `&[e] end) end, Constraint.TerraType)
  expect(topointer, true, quote var a : int = 3 in a end)
  expect(topointer, true, quote var a : bool = true in a end)
  expect(topointer:Types(int), true, quote var a : int = 3 in a end)
  expect(topointer:Types(float), false, quote var a : int = 3 in a end)

  local inner = terra(a : int) : int return a end
  local innerbad = terra(a : float) : int return a end
  local wrap = Constraint.Meta({Constraint.Number, Constraint.Function({int}, int)}, Constraint.Number, function(a, b) return b(a) end)
  expect(wrap, true, 3, inner)
  expect(wrap, false, `3, inner)
  expect(wrap, false, 3, innerbad)

  local nest = Constraint.Meta({Constraint.Number, Constraint.Function({int}, int), Constraint.Function({Constraint.Number, Constraint.Function({int}, int)}, Constraint.Number)}, Constraint.Number, function(a, b, c) return c(a, b) end)
  expect(nest, true, 3, inner, wrap)
  expect(nest, false, 3, wrap, wrap)
  expect(nest, false, 3, inner, inner)
  expect(nest, false, `3, inner, wrap)
  expect(nest, false, 3, 3, wrap)
  expect(nest, false, 3, inner, 3)
  expect(nest, false, 3, innerbad, wrap)

  expect(nest, true, 3, macro(function(a) return a end, function(a) return a end), wrap)

  expect(Constraint.Negative("f") + Constraint.Field("f", Constraint.Field("foobar", float) * Constraint.Field("meta", bool)), true, nested)
  expect(Constraint.Negative("f") + Constraint.Field("f2", Constraint.Field("foobar", float) * Constraint.Field("meta", bool)), false, nested)
  expect(Constraint.Negative("f") + Constraint.Field("f", Constraint.Field("foobar", int) * Constraint.Field("meta", bool)), false, nested)
  expect(Constraint.Negative("f") + Constraint.Field("f2", Constraint.Field("foobar", float) * Constraint.Field("meta", bool)), false, nested)
  --local nest = Constraint.Meta(Constraint.Number, Constraint.Function({Constraint.Number, Constraint.Function({int}, int)}, Constraint.Number), function(a, b) return b(a, inner) end)
  
  local opt = Constraint.Meta({Constraint.Empty + Constraint.Number}, Constraint.Number, function(a) return 3 end)
  expect(opt, true, 2)
  expect(opt, true)
  expect(opt, false, 2, 2)

  local opt2 = Constraint.Meta({Constraint.Empty + Constraint.Number}, Constraint.Empty + Constraint.Number, function(a) return a end)
  expect(opt2, true, 2)
  expect(opt2, true)
  expect(opt2, false, 2, 2)

  expect(Constraint.Meta({Constraint.Empty + Constraint.String}, Constraint.Empty + Constraint.String, function(a) return a end), true, "asdf")

  fiddle.methods["over"] = terralib.overloadedfunction("overload", { terra(self: &fiddle, s : int) : int return s end, terra(self: &fiddle, s : rawstring) : rawstring return s end })
  testmethod({int}, int, false, nil, true, fiddle, "over")
  testmethod({rawstring}, rawstring, false, nil, true, fiddle, "over")
  testmethod({rawstring}, int, false, nil, false, fiddle, "over")
  testmethod({Constraint.Value(rawstring) + int}, int, false, nil, true, fiddle, "over")
  testmethod({Constraint.Empty + int}, int, false, nil, true, fiddle, "over")
  testmethod({Constraint.Empty + int}, rawstring, false, nil, false, fiddle, "over")

  expect(Constraint.IsConstraint, true, Constraint.IsConstraint)
  expect(Constraint.IsConstraint, false, {})
end

local struct ObjectTest(Object.Object) { i : bool, d : bool }
terra ObjectTest:init()
  self:_init{ true, false }
end
terra ObjectTest:destruct()
  self.d = true
end

terra TestHarness:object()
  var obj : ObjectTest = ObjectTest{ false, false }
  self:Test(obj.i, false)
  self:Test(obj.d, false)

  Object.init(obj)
  self:Test(obj.i, true)
  self:Test(obj.d, false)

  Object.destruct(obj)
  self:Test(obj.i, true)
  self:Test(obj.d, true)
end

terra TestHarness:switchstat()
  var a : int = 999
  switch a do 
    case 3 then 
      a = 3
    case 1 then
      a = 1
    case 999 then
      a = 9
    else
      a = 10
  end
  
  self:Test(a, 9)
end

local RTreeDefault = RTree(Alloc.libc_allocator)
Partitioner(RTreeDefault)

local verify_num = global(int[10], `array(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1))
local verify_index = global(int, `0)

terra verifywalk(n : &RTreeDefault.Node, p : &F.Vec3, r : &F.Vec3, i : int) : bool 
  verify_num[verify_index] = [intptr](n.data)
  verify_index = verify_index + 1
  return false 
end

terra testquery(n : &RTreeDefault.Node, p : &F.Vec3, r : &F.Vec3, i : int) : bool 
  verify_num[verify_index] = [intptr](n.data)
  verify_index = verify_index + 1
  return n.data == [&Msg.Receiver](3)
end

terra TestRayBoxIntersect(pos : F.Vec3, v : F.Vec3, extent : F.Vec3) : float
  return RTreeDefault.RayBoxIntersect(&pos, &v, &extent)
end

terra TestHarness:rtree()
  var simple : RTreeDefault
  Object.new(simple)
  var vec1000 = Transform{F.vec3(1000.0f,1000.0f,0.0f), F.zero, F.zero}
  var vec100 = Transform{F.vec3(100.0f,100.0f,0.0f), F.zero, F.zero}
  var vec0 = Transform{F.zero, F.zero, F.zero}
  var vec1i = F.veci(1,1)
  var root = simple:create(nil, &vec0, vec1000.pos, F.zero)
  self:Test(root ~= nil, true)
  var child1 = simple:create(root, &vec0, vec100.pos, F.zero)
  self:Test(child1 ~= nil, true)
  var child2 = simple:create(root, &vec100, vec100.pos, F.zero)
  self:Test(child2 ~= nil, true)
  var child3 = simple:create(child2, &vec0, vec100.pos, vec1i)
  var child4 = simple:create(child2, &vec0, vec100.pos, F.zero)
  simple:sort(child2, nil)
  root.data = [&Msg.Receiver](0)
  child1.data = [&Msg.Receiver](1)
  child2.data = [&Msg.Receiver](2)
  child3.data = [&Msg.Receiver](3)
  child4.data = [&Msg.Receiver](4)
  simple:query(F.vec3(3.0f,4.0f,0.0f), F.vec3(0.0f,0.0f,1.0f), 3, verifywalk)

  escape
    local vals = {3, 4, 2, 1, 0}
    for i,v in ipairs(vals) do
      emit(`self:Test(verify_num[ [i-1] ], [v]))
    end
    emit(`self:Test(verify_num[ [#vals] ], -1))
  end

  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(0f, 0f, 1f), F.vec3(1f, 1f, 1f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(0f, 1f, 0f), F.vec3(1f, 1f, 1f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(1f, 0f, 0f), F.vec3(1f, 1f, 1f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(2f, 2f, 2f), F.vec3(0f, 0f, -1f), F.vec3(1f, 1f, 1f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(2f, 2f, 2f), F.vec3(-1f, -1f, -1f), F.vec3(1f, 1f, 1f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(4f, 4f, 4f), F.vec3(-1f, -1f, -1f), F.vec3(1f, 1f, 1f)), 3f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), 0f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(0f, 1f, 0f), F.vec3(10f, 10f, 0f)), 10f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 0f), F.vec3(1f, 0f, 0f), F.vec3(10f, 10f, 0f)), 10f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 100f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), -100f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, 100f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), 100f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, -10f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), 10f)
  self:Test(TestRayBoxIntersect(F.vec3(9.9f, 0f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), 0f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 0f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(9.9f, 9.9f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), 0f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 9.9f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 10.1f, 0f), F.vec3(0f, 0f, 1f), F.vec3(10f, 10f, 0f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(0f, 0f, -10f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), -10f)
  self:Test(TestRayBoxIntersect(F.vec3(9.9f, 0f, 0f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), 0f)
  self:Test(TestRayBoxIntersect(F.vec3(9.9f, 0f, 1f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 0f, 1f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(9.9f, 9.9f, 1f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), 1f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 9.9f, 1f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), -1f)
  self:Test(TestRayBoxIntersect(F.vec3(10.1f, 10.1f, 1f), F.vec3(0f, 0f, -1f), F.vec3(10f, 10f, 0f)), -1f)
end

local SUBTESTLEN = 11

local terra PrintColumns(a : rawstring, b : rawstring, c : rawstring) : {}
  C.printf("%-*s %-*s %-*s\n", 24, a, [SUBTESTLEN], b, 8, c);
end

terra main(argc : int, argv : &rawstring) : int
  C.printf("Feather v%i.%i.%i Test Utility\n\n", [F.Version[1]], [F.Version[2]], [F.Version[3]])

  var harness = TestHarness{ }
  
  PrintColumns("Internal Tests", "Subtests", "Pass/Fail")
  PrintColumns("--------------", "--------", "---------")

  escape
    for k, v in pairs(TestHarness.methods) do
      if not terralib.ismacro(v) and #v.type.parameters == 1 then
        emit(quote
          harness.passed, harness.total = 0, 0
          harness:[k]()
          var buf : int8[SUBTESTLEN + 1]
          C.snprintf(buf, [SUBTESTLEN + 1], "%u/%u", harness.passed, harness.total);
          PrintColumns(k, buf, terralib.select(harness.passed == harness.total, "PASS", "FAIL"))
        end)
      end
    end
  end

  return 0;
end

local targetname = "fgTests"
local clangargs = {}
if isdebug then
  table.insert(clangargs, "-g")
  table.insert(clangargs, "/DEBUG")
  targetname = targetname .. "_d"
end

if jit.os == "Windows" then
  targetname = targetname .. ".exe"
end

if jit.os == "Linux" then
  table.insert(clangargs, "-ldl")
  table.insert(clangargs, "-lm")
end

print(clangargs[0])
if terralib.saveobj("bin-"..jit.arch.."/"..targetname, "executable", {main = main}, clangargs) ~= nil then
  return -1
end

main(0, nil)
