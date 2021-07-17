local core = require 'feather.core'
local F = require 'feather.shared'
local override = require 'feather.util'.override
local messages = require 'feather.messages'
local Msg = require 'feather.message'
local Virtual = require 'feather.virtual'
local C = terralib.includecstring [[
#include <stdio.h>
]]

local gen_window_node = terralib.memoize(function(body_type, rtree_node, window_base)
  local struct window_node(Virtual.extends(window_base)) {
    window: &Msg.Window
    body: body_type
    node : rtree_node
    color: F.Color
  }
  return window_node
  end)
  
return core.raw_template {
  color = `F.Color{0},
  pos = `F.vec(0f, 0f),
  size = `F.vec(800f, 600f),
  core.body
} (
  function(self, type_context, type_environment)
    if type_context.window then
      error "NYI: instantiating a window inside another window"
    end
    local rtree_type = type_context.rtree
    local body_fns, body_type = type_environment[core.body](override(type_context, {window = &Msg.Window, transform = &core.transform}), type_environment)
    
    local function make_context(self, ui, accumulated)
      return {
        rtree = `self.rtree,
        rtree_node = `self.rtree.root,
        allocator = `self.rtree.allocator,
        backend = `ui.backend,
        window = `self.window,
        transform = accumulated,
        accumulated_parent_transform = accumulated
      }
    end

    local function override_context(self, context)
      return override(context, {
        rtree = `self.rtree,
        rtree_node = `self.rtree.root,
        allocator = `self.rtree.allocator,
        window = `self.window,
        transform = `core.transform.identity()
      })
    end

    local fn_table = {
      enter = function(self, context, environment)
        return quote
          self.vftable = [self:gettype()].virtual_initializer
          var pos = [environment.pos]
          var size = [environment.size]
          var ext = F.vec3(size.x / 2f, size.y / 2f, 0f)
          var local_transform = core.transform.translate(ext)
          --var local_transform = core.transform.identity()
          self.rtree:init()
          self.node = self.rtree:create(nil, &local_transform, ext, F.zero)
          self.node.data = &self.super.super
          self.window = [context.backend]:CreateWindow(self.node.data, nil, &pos, &size, "feather window", messages.WindowFlag.RESIZABLE)
          self.color = [environment.color]
          [body_fns.enter(`self.body, override_context(self, context), environment)]
        end
      end,
      update = function(self, context, environment)
        return quote
          self.color = environment.color
          [body_fns.update(`self.body, override_context(self, context), environment)]
          [context.backend]:DirtyRect(self.window, nil) --TODO: make this smart enough to only call for a redraw if something changed.
        end
      end,
      layout = function(self, context, environment)
        return quote
          [body_fns.layout(`self.body, override_context(self, context), environment)]
        end
      end,
      exit = function(self, context)
        return quote
          if self.window ~= nil then
            [body_fns.exit(`self.body, override_context(self, context))]
            [context.backend]:DestroyWindow(self.window)
            self.window = nil
            self.rtree:destruct()
          end
        end
      end,
      render = function(self, context)
        return quote
            [body_fns.render(`self.body, override(context, {
              rtree = `self.rtree,
              rtree_node = `self.rtree.root,
              allocator = `self.rtree.allocator,
              window = `self.window,
              transform = `self.rtree.root.transform,
              accumulated_parent_transform = `self.rtree.root.transform
            }))]
          end
      end
    }

    local window_node = gen_window_node(body_type, type_context.rtree_node, type_context.window_base)
    
    terra window_node:Draw(ui : &opaque) : F.Err
      [&type_context.ui](ui).backend:Clear(self.window, self.color)
      [body_fns.render(`self.body, make_context(self, `[&type_context.ui](ui), `self.rtree.root.transform))]
      return 0
    end
    
    terra window_node:Action(ui : &opaque, subkind : int) : F.Err
      switch subkind do
        case 1 then
          [fn_table.exit(`self, make_context(self, `[&type_context.ui](ui)))]
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
