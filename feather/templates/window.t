local core = require 'feather.core'
local F = require 'feather.shared'
local override = require 'feather.util'.override
local messages = require 'feather.messages'
local Msg = require 'feather.message'
local Virtual = require 'feather.virtual'

local gen_window_node = terralib.memoize(function(body_type, rtree_node, window_base)
  local struct window_node(Virtual.extends(window_base)) {
    window: &opaque
    body: body_type
    node : rtree_node
    color: F.Color
  }
  return window_node
  end)
  
return core.raw_template {
  color = `F.Color{0},
  core.body
} (
  function(self, context, type_environment)
    if context.window then
      error "NYI: instantiating a window inside another window"
    end
    local rtree_type = context.rtree
    local context_ = override(context, {window = &opaque, transform = &core.transform})
    local body_fns, body_type = type_environment[core.body](context_, type_environment)
    
    local function make_context(self, ui)
      return {
        rtree = `self.rtree,
        rtree_node = `self.rtree.root,
        allocator = `self.rtree.allocator,
        backend = `ui.backend,
        window = `self.window,
        transform = `core.transform.identity(),
      }
    end

    local function override_context(self, context)
      return override(context, {
        rtree = `self.rtree,
        rtree_node = `self.rtree.root,
        allocator = `self.rtree.allocator,
        window = `self.window,
        transform = `core.transform.identity(),
      })
    end

    local fn_table = {
      enter = function(self, context, environment)
        return quote
          self.vftable = [self:gettype()].virtual_initializer
          var pos = F.Vec{array(0f, 0f)}
          var size = F.Vec{array(800f, 600f)}
          var transform = core.transform.identity()
          var zero = [F.Vec3D] {array(0.0f, 0.0f, 0.0f)}
          var zindex = [F.Veci] {array(0, 0)}
          self.rtree:init()
          self.node = self.rtree:create(nil, &zero, &zero, &zero, &zindex)
          self.node.data = &self.super.super
          self.window = [context.backend]:CreateWindow(self.node.data, nil, &pos, &size, "feather window", messages.Window.RESIZABLE, nil)
          self.color = environment.color
          [body_fns.enter(`self.body, override_context(self, context), environment)]
        end
      end,
      update = function(self, context, environment)
        return quote
          var transform = core.transform.identity()
          self.color = environment.color
          [body_fns.update(`self.body, override_context(self, context), environment)]
          [context.backend]:DirtyRect(self.window, nil) --TODO: make this smart enough to only call for a redraw if something changed.
        end
      end,
      exit = function(self, context)
        return quote
          if self.window ~= nil then
            var transform = core.transform.identity()
            [body_fns.exit(`self.body, override_context(self, context))]
            [context.backend]:DestroyWindow(self.window)
            self.window = nil
            self.rtree:destruct()
            end
          end
      end,
      render = function(self, context)
        return quote
            var transform = core.transform.identity()
            [body_fns.render(`self.body, override_context(self, context))]
          end
      end
    }

    local window_node = gen_window_node(body_type, context.rtree_node, context.window_base)
    
    terra window_node:Draw(ui : &opaque) : F.Err
      [&context.ui](ui).backend:Clear(self.window, self.color)
      [body_fns.render(`self.body, make_context(self, `[&context.ui](ui)))]
      return 0
    end
    
    terra window_node:Action(ui : &opaque, subkind : int) : F.Err
      switch subkind do
        case 1 then
          [fn_table.exit(`self, make_context(self, `[&context.ui](ui)))]
        end
      end
    end
  
    -- Everything rejects messages by default, but a standard window has to consume mouse events by default
    terra window_node:MouseDown(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return 0 end
    terra window_node:MouseDblClick(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return 0 end
    terra window_node:MouseUp(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return 0 end
    terra window_node:MouseOn(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:MouseOff(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:MouseMove(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:MouseScroll(ui : &opaque, x : float, y : float, delta : float, hdelta : float) : F.Err return 0 end
    terra window_node:TouchBegin(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:TouchMove(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:TouchEnd(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return 0 end
    terra window_node:KeyUp(ui : &opaque, key : uint8, modkeys : uint8, scancode : uint16) : F.Err return 0 end
    terra window_node:KeyDown(ui : &opaque, key : uint8, modkeys : uint8, scancode : uint16) : F.Err return 0 end
    terra window_node:KeyChar(ui : &opaque, unicode : int32, modkeys : uint8) : F.Err return 0 end

    return fn_table, window_node
  end
  )
