local core = require 'feather.core'
local shared = require 'feather.shared'
local B = require 'feather.backend'

return core.basic_template {
  color = `[shared.Color]{0},
  outline = `[shared.Color]{0},
  angles = `shared.Vec{array(0f, 6.284f)}, -- specified in (start angle, end angle - start angle)
  border = `0.0f,
  innerRadius = `0.0f,
  blur = `0.0f
} (function(context, params)
    return quote
        var x, y = context.transform.pos.x, context.transform.pos.y
        var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
        var rect = shared.rect(x-w, y-h, x+w, y+h)
        var command : B.Command
        command.category = B.Category.ARC
        command.shape.arc.innerRadius = [params.innerRadius]
        command.shape.arc.angles = [params.angles]
        command.shape.area = &rect
        command.shape.fillColor = [params.color]
        command.shape.border = [params.border]
        command.shape.borderColor = [params.outline]
        command.shape.blur = [params.blur]
        command.shape.asset = nil
        command.shape.z = context.transform.pos.z
        [context.backend]:Draw([context.window], &command, 1, nil)
      end
   end
  )
