local core = require 'feather.core'
local F = require 'feather.shared'
local Msg = require 'feather.message'
local B = require 'feather.backend'
local override = require 'feather.util'.override
local Virtual = require 'feather.virtual'
local C = terralib.includecstring [[#include <stdio.h>]]

return core.raw_template {
  pos = constant(`[F.Vec3]{array(0.0f, 0.0f, 0.0f)}),
  ext = constant(`[F.Vec3]{array(0.0f, 0.0f, 0.0f)}),
  rot = constant(`[F.Vec3]{array(0.0f, 0.0f, 0.0f)}),
  mousedown = `F.EmptyCallable{},
  mousedblclick = `F.EmptyCallable{},
  mouseup = `F.EmptyCallable{},
  mouseon = `F.EmptyCallable{},
  mouseoff = `F.EmptyCallable{},
  mousemove = `F.EmptyCallable{},
  mousescroll = `F.EmptyCallable{},
  touchbegin = `F.EmptyCallable{},
  touchmove = `F.EmptyCallable{},
  touchend = `F.EmptyCallable{},
} (
  function(self, type_context, type_environment)
    local body_fns, body_type = type_environment[core.body](override(type_context, {window = &opaque, transform = &core.transform}), type_environment)
    
    local fn_table = {
        enter = function(self, context, environment)
          return quote
            self.vftable = [self:gettype()].virtual_initializer
            var pos = [environment.pos]
            var ext = [environment.ext]
            var rot = [environment.rot]
            var z_index = [F.Veci]{array(0, 0)}
            self.node = [context.rtree]:create([context.rtree_node], &pos, &ext, &rot, &z_index)
            self.node.data = &self.super
            self.mousedown = environment.mousedown
            --var local_transform = M.transform{self.node.pos}
            --var transform = [context.transform]:compose(&local_transform)
            --;[body_fns.enter(`self.body, core.override(context, {rtree_node = `self._2, transform = `transform}), environment)]
          end
        end,
        update = function(self, context, environment)
          return quote
            self.node.pos = [environment.pos]
            self.node.extent = [environment.ext]
            self.node.rot = [environment.rot]
          end
        end,
        exit = function(self, context)
          return quote
            [context.rtree]:destroy(self.node)
          end
        end,
        render = function(self, context)
          return quote 
          -- Temporarily draws a white rectangle so you know where the mousearea is
        -- var local_transform = [core.transform]{self.node.pos}
        -- var transform = [context.transform]:compose(&local_transform)
        -- var x, y = transform.r.x, transform.r.y
        -- var w, h = self.node.extent.x, self.node.extent.y
        -- var rect = F.Rect{array(x-w, y-h, x+w, y+h)}
        -- var corners = F.Rect{array(0f, 0f, 0f, 0f)}
        -- var command : B.Command
        -- command.category = B.Category.RECT
        -- command.shape.rect.corners = &corners
        -- command.shape.rect.rotate = 0.0f
        -- command.shape.area = &rect
        -- command.shape.fillColor = [F.Color]{0xffffffff}
        -- command.shape.border = 0.0f
        -- command.shape.borderColor = [F.Color]{0xffffffff}
        -- command.shape.blur = 0.0f
        -- command.shape.asset = nil
        -- command.shape.z = 0.0f
        -- [context.backend]:Draw([context.window], &command, 1, nil)
          end
        end
      }

    local struct mousearea(Virtual.extends(Msg.Receiver)) {
      node : type_context.rtree_node
      body: body_type
    }
    -- Store all callables. The default empty ones are zero-size types, which take up no space.
    for i,v in ipairs(self.params.names) do
      mousearea.entries:insert {field = v, type = type_environment[v]}
    end


    terra mousearea:MouseDown(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mousedown() end
    terra mousearea:MouseDblClick(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mousedblclick() end
    terra mousearea:MouseUp(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mouseup() end
    terra mousearea:MouseOn(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return self.mouseon() end
    terra mousearea:MouseOff(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return self.mouseoff() end
    terra mousearea:MouseMove(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err --[[C.printf("TEST %g %g\n", x, y)]] return self.mousemove() end
    terra mousearea:MouseScroll(ui : &opaque, x : float, y : float, delta : float, hdelta : float) : F.Err return self.mousescroll() end
    terra mousearea:TouchBegin(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchbegin() end
    terra mousearea:TouchMove(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchmove() end
    terra mousearea:TouchEnd(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchend() end

    return fn_table, mousearea
  end
  )
