local core = require 'feather.core'
local shared = require 'feather.shared'

return core.basic_template {
  color = `[shared.Color]{0},
  outline = `[shared.Color]{0},
  corners = `array(0f, 0f, 0f),
  offset = `0.0f,
  border = `0.0f,
  blur = `0.0f
} (function(context, params)
    return quote
        var x, y = context.transform.r.x, context.transform.r.y
        var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
        var rect = shared.Rect{array(x-w, y-h, x+w, y+h)}
        var corners = shared.Rect{array([params.corners][0], [params.corners][1], [params.corners][2], [params.offset])} -- copy so we get an l-value
        [context.backend]:DrawTriangle([context.window], &rect, &corners, [params.color], [params.border], [params.outline], [params.blur], nil, 0, 0, nil)
      end
   end
  )