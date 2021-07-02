local core = require 'feather.core'
local F = require 'feather.shared'
local Msg = require 'feather.message'
local Messages = require 'feather.messages'
local B = require 'feather.backend'
local override = require 'feather.util'.override
local Virtual = require 'feather.virtual'
local DefaultLayout = require 'feather.layouts.basic'
local C = terralib.includecstring [[#include <stdio.h>]]

return core.raw_template {
  layout = constant(`DefaultLayout.zero),
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
  function(this, type_context, type_environment)
    local body_fns, body_type = type_environment[core.body](override(type_context, {window = &opaque, transform = &core.transform}), type_environment)
    
    local fn_table = {
        enter = function(self, context, environment)
          return quote
            self.vftable = [self:gettype()].virtual_initializer
            var local_transform = [context.transform]
            self.node = [context.rtree]:create([context.rtree_node], &local_transform, F.zero, F.zero)
            self.node.data = &self.super
            escape
              for i,v in ipairs(this.params.names) do
                emit(quote self.[v] = environment.[v] end)
              end
            end
            --;[body_fns.enter(`self.body, core.override(context, {rtree_node = `self._2, transform = `transform}), environment)]
          end
        end,
        update = function(self, context, environment)
          return quote
            self.node.data = &self.super
            escape
              for i,v in ipairs(this.params.names) do
                emit(quote self.[v] = environment.[v] end)
              end
            end
          end
        end,
        layout = function(self, context)
          return quote
            var local_transform = [context.transform]
            self.layout:apply(self.node, &local_transform, context.rtree_node)
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
         --[[var local_transform = [core.transform]{self.node.pos}
         var transform = [context.transform]:compose(&local_transform)
         var x, y = transform.pos.x, transform.pos.y
         var w, h = self.node.extent.x, self.node.extent.y
         var rect = F.Rect{array(x-w, y-h, x+w, y+h)}
         var corners = F.Rect{array(0f, 0f, 0f, 0f)}
         var command : B.Command
         command.category = B.Category.RECT
         command.shape.rect.corners = &corners
         command.shape.rect.rotate = 0.0f
         command.shape.area = &rect
         command.shape.fillColor = [F.Color]{0xffffffff}
         command.shape.border = 0.0f
         command.shape.borderColor = [F.Color]{0xffffffff}
         command.shape.blur = 0.0f
         command.shape.asset = nil
         command.shape.z = 0.0f
         [context.backend]:Draw([context.window], &command, 1, nil)]]
          end
        end
      }

    local struct mousearea(Virtual.extends(Msg.Receiver)) {
      node : type_context.rtree_node
      layout : DefaultLayout
      body: body_type
    }
    -- Store all callables. The default empty ones are zero-size types, which take up no space.
    for i,v in ipairs(this.params.names) do
      if v ~= "layout" then
        mousearea.entries:insert {field = v, type = type_environment[v]}
      end
    end

    terra mousearea:MouseDown(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mousedown(Messages.CreateMouseEvent(x, y, all, modkeys, button)) end
    terra mousearea:MouseDblClick(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mousedblclick(Messages.CreateMouseEvent(x, y, all, modkeys, button)) end
    terra mousearea:MouseUp(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8, button : uint8) : F.Err return self.mouseup(Messages.CreateMouseEvent(x, y, all, modkeys, button)) end
    terra mousearea:MouseOn(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return self.mouseon(Messages.CreateMouseEvent(x, y, all, modkeys, 0)) end
    terra mousearea:MouseOff(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return self.mouseoff(Messages.CreateMouseEvent(x, y, all, modkeys, 0)) end
    terra mousearea:MouseMove(ui : &opaque, x : float, y : float, all : uint8, modkeys : uint8) : F.Err return self.mousemove(Messages.CreateMouseEvent(x, y, all, modkeys, 0)) end
    terra mousearea:MouseScroll(ui : &opaque, x : float, y : float, delta : float, hdelta : float) : F.Err return self.mousescroll(Messages.CreateMouseDelta(x, y, delta, hdelta)) end
    terra mousearea:TouchBegin(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchbegin(Messages.CreateTouchEvent(x, y, z, r, pressure, index, flags, modkeys)) end
    terra mousearea:TouchMove(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchmove(Messages.CreateTouchEvent(x, y, z, r, pressure, index, flags, modkeys)) end
    terra mousearea:TouchEnd(ui : &opaque, x : float, y : float, z : float, r : float, pressure : float, index : uint16, flags : uint8, modkeys : uint8) : F.Err return self.touchend(Messages.CreateTouchEvent(x, y, z, r, pressure, index, flags, modkeys)) end

    return fn_table, mousearea
  end
  )
