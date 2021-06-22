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
      p0 : F.Vec
      p1 : F.Vec
    }

    Virtual.extends(Msg.Receiver)(storage_type)

    local fn_table = {
        enter = function(self, context, environment)
          return quote
            self.vftable = [self:gettype()].virtual_initializer
            var pos = minpos(environment.points)
            var ext = maxpos(environment.points) - pos
            var z_index = [F.Veci]{array(0, 0)}
            self.node = [context.rtree]:create([context.rtree_node], &pos, &ext, F.vec3(0.0f,0.0f,0.0f), &z_index)
            self.node.data = &self.super
            self.color = [environment.color]
            self.p0 = [environment.points]._0
            self.p1 = [environment.points]._1
          end
        end,
        update = function(self, context, environment)
          return quote
            self.node.pos = minpos(environment.points)
            self.node.extent = maxpos(environment.points) - self.node.pos
            self.color = [environment.color]
            self.p0 = [environment.points]._0
            self.p1 = [environment.points]._1
          end
        end,
        exit = function(self, context)
          return quote
            [context.rtree]:destroy(self.node)
          end
        end,
        render = function(self, context)
          return quote 
          -- Temporarily draws a white rectangle so you know where it is
         --[[var local_transform = [core.transform]{self.node.pos}
         var transform = [context.transform]:compose(&local_transform)
         var x, y = transform.r.x, transform.r.y
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
         var local_p0 = [core.transform]{ self.p0 };
         var local_p1 = [core.transform]{ self.p1 };
         var p0 = [context.transform]:compose(&local_p0)
         var p1 = [context.transform]:compose(&local_p1)
         var points = {F.vec(p0.r.x, p0.r.y), F.vec(p1.r.x, p1.r.y)}
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
