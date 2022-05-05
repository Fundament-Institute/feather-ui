local core = require 'feather.core'
local shared = require 'feather.shared'

return core.basic_template {
  font = `shared.Font{nil},
  layout = `[shared.Asset]{nil},
  color = `[shared.Color]{0},
  blur = `0.0f,
} (function(context, params)
    return quote
        var x, y = context.transform.pos.x , context.transform.pos.y
        var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
        var rect = shared.rect(x-w, y-h, x+w, y+h)
        var corners = [params.corners] -- copy so we get an l-value
        [context.backend]:DrawText([context.window], font, layout, &rect, [params.color], [params.blur], 0, 0, nil)
      end
   end
  )