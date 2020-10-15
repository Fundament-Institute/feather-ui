local core = require 'feather.core'
local shared = require 'feather.shared'
local override = require 'feather.util'.override
local messages = require 'feather.messages'

return core.raw_template {
} (
  function(context, type_environment)
    if context.window then
      error "NYI: instantiating a window inside another window"
    end
    local context_ = override(context, {window = &opaque, transform = &core.transform})
    local body_fns, body_type = type_environment[core.body](context_, type_environment)
    local struct window_node {
      window: &opaque
      body: body_type
                             }

    return {
      enter = function(self, context, environment)
        print("printing body context")
        print(self)
        terralib.printraw(context)
        terralib.printraw(environment)
        return quote
            var pos = shared.Vec{array(0f, 0f)}
          var size = shared.Vec{array(800f, 600f)}
          var transform = core.transform.identity()
          self.window = [context.backend]:CreateWindow(nil, nil, &pos, &size, "feather window", messages.Window.RESIZEABLE, nil)
          ;[body_fns.enter(`self.body, override(context, {window = `self.window, transform = `transform}), environment)]
               end
      end,
      update = function(self, context, environment)
        return quote
            var transform = core.transform.identity()
          ;[body_fns.enter(`self.body, override(context, {window = `self.window, transform = `transform}), environment)]
               end
      end,
      exit = function(self, context)
        return quote
            var transform = core.transform.identity()
          ;[body_fns.exit(`self.body, override(context, {window = `self.window, transform = `transform}))]
          [context.backend]:DestroyWindow(self.window)
               end
      end,
      render = function(self, context)
        return quote
            var transform = core.transform.identity()
          ;[body_fns.render(`self.body, override(context, {window = `self.window, transform = `transform}))]
               end
      end
    }, window_node
  end
  )
