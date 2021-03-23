local core = require 'feather.core'
local shared = require 'feather.shared'
local B = require 'feather.backend'

return core.basic_template {
  color = `[shared.Color]{0},
  outline = `[shared.Color]{0},
  corners = `shared.Rect{array(0f, 0f, 0f, 0f)},
  border = `0.0f,
  blur = `0.0f
} (function(context, params)
    return quote
        var x, y = context.transform.r.x , context.transform.r.y
        var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
        var rect = shared.rect(x-w, y-h, x+w, y+h)
        var corners = [params.corners] -- copy so we get an l-value
        var command : B.Command
        command.category = B.Category.RECT
        command.shape.rect.corners = &corners
        command.shape.rect.rotate = 0.0f
        command.shape.area = &rect
        command.shape.fillColor = [params.color]
        command.shape.border = [params.border]
        command.shape.borderColor = [params.outline]
        command.shape.blur = [params.blur]
        command.shape.asset = nil
        command.shape.z = context.transform.r.z
        [context.backend]:Draw([context.window], &command, 1, nil)
      end
   end
  )