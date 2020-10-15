local core = require 'feather.core'
local shared = require 'feather.shared'

return core.basic_template {
  color = `[shared.Color]{0}
} (function(context, params)
    return quote
      var x, y = context.transform.r.x, context.transform.r.y
      var w, h = context.rtree_node.extent.x, context.rtree_node.extent
      var rect: shared.Rect{x, y, x+w, y+h}
      var corners: shared.Rect{0, 0, 0, 0}
      [context.backend]:DrawRect([context.window], &rect, &corners, &[params.color], 0, &[params.color], 0, nil, 0, 0)
           end
   end
  )
