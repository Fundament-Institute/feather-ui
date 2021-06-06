local core = require 'feather.core'
local shared = require 'feather.shared'
local B = require 'feather.backend'

return core.basic_template {
  color = `[shared.Color]{0},
  points = `core.required
} (function(context, params)
    return quote
        var x, y = context.transform.r.x, context.transform.r.y
        var w, h = context.rtree_node.extent.x, context.rtree_node.extent.y
        var rect = shared.rect(x-w, y-h, x+w, y+h)
        var command : B.Command
        command.category = B.Category.LINES
        var points = [params.points]
        command.lines.points = &points._0
        command.lines.count = 2
        command.lines.color = [params.color]
        [context.backend]:Draw([context.window], &command, 1, nil)
      end
   end
  )
