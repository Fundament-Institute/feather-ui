local core = require 'feather.core'
local F = require 'feather.shared'
local B = require 'feather.backend'
local Math = require 'std.math'
local Virtual = require 'feather.virtual'
local Msg = require 'feather.message'

terra minpos(p : {F.Vec, F.Vec}) : F.Vec3
  return F.vec3(Math.min(p._0.x, p._1.x), Math.min(p._0.y, p._1.y), 0.0f)
end

terra maxpos(p : {F.Vec, F.Vec}) : F.Vec3
  return F.vec3(Math.max(p._0.x, p._1.x), Math.max(p._0.y, p._1.y), 0.0f)
end

return core.raw_template {
  color = `[F.Color]{0},
  points = `core.required
} (
  function(self, type_context, type_environment)
    local struct storage_type {
      node : type_context.rtree_node
      color: F.Color
      p0 : F.Vec3
      p1 : F.Vec3
    }

    Virtual.extends(Msg.Receiver)(storage_type)

    local fn_table = {
        enter = function(self, context, environment)
          return quote
            self.vftable = [self:gettype()].virtual_initializer
            var pos = minpos(environment.points)
            var ext = (maxpos(environment.points) - pos):component_div(2.0f)
            pos = pos + ext
            var local_transform = core.transform.translate(pos)
            local_transform = [context.transform]:compose(&local_transform)
            self.node = [context.rtree]:create([context.rtree_node], &local_transform, ext, F.zero)
            self.node.data = &self.super
            self.color = [environment.color]
            self.p0 = [environment.points]._0 - pos
            self.p1 = [environment.points]._1 - pos
          end
        end,
        update = function(self, context, environment)
          return quote
            var pos = minpos(environment.points)
            var ext = (maxpos(environment.points) - pos):component_div(2.0f)
            pos = pos + ext
            var local_transform = core.transform.translate(pos)
            self.node.transform = [context.transform]:compose(&local_transform)
            self.node.extent = ext
            self.color = [environment.color]
            self.p0 = [environment.points]._0 - pos
            self.p1 = [environment.points]._1 - pos
          end
        end,
        exit = function(self, context)
          return quote
            [context.rtree]:destroy(self.node)
          end
        end,
        render = function(self, context)
          return quote 
          var local_transform = [context.accumulated_parent_transform]:compose(&self.node.transform)
          -- Temporarily draws a white rectangle so you know where it is
          --[[do
            var x, y =  self.node.transform.pos.x, self.node.transform.pos.y
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
            [context.backend]:Draw([context.window], &command, 1, nil)
          end]]

         var local_p0 = core.transform.translate(self.p0)
         var local_p1 = core.transform.translate(self.p1)
         var p0 = local_transform:compose(&local_p0)
         var p1 = local_transform:compose(&local_p1)
         var points = {F.vec(p0.pos.x, p0.pos.y), F.vec(p1.pos.x, p1.pos.y)}
          var command : B.Command
          command.category = B.Category.LINES
          command.lines.points = &points._0
          command.lines.count = 2
          command.lines.color = self.color
          [context.backend]:Draw([context.window], &command, 1, nil)
          end
        end
      }

    return fn_table, storage_type
  end
  )
