local core = require 'feather.core'
local shared = require 'feather.shared'

return core.basic_template {
  color = `[shared.Color]{0}
} (function(context, params)
    return quote
      var x, y = context.transform.r.x, context.transform.r.y
      var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
      var rect = shared.Rect{array(x, y, x+w, y+h)}
      var corners = shared.Rect{array(0f, 0f, 0f, 0f)}
      [context.backend]:DrawRect([context.window], &rect, &corners, [params.color], 0, [params.color], 0, nil, 0, 0, nil)
           end
   end
  )
