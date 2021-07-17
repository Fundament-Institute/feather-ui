local f = require 'feather'
local dynarray = require 'feather.dynarray'
local scaffolding = require 'feather.scaffolding'
local Messages = require 'feather.messages'

local C = terralib.includec "stdio.h"

import 'feather/bind'

local struct application {
  shape: int
                         }
terra application:init(argc: int, argv: &rawstring)
  self.shape = 0
end
terra application:update() end
terra application:exit() end


local stroke_color = bind f.Color { 0xffcccccc }
local clickable_box = f.template {
  pos = `f.vec3(0f, 0f, 0f),
  ext = f.required,
  fill = `f.Color {0xff000000},
  onclick = f.requiredevent()
} {
  f.rect {
    pos = bind pos,
    ext = bind ext,
    outline = stroke_color,
    color = bind fill,
    border = bind 10
  },
  f.mousearea {
    pos = bind pos,
    ext = bind ext,
    mousedown = bindevent(me: Messages.MouseEvent)
      C.printf("in clickable_box mousedown\n")
      onclick()
      return 1
    end
  }
}

local ui = f.ui {
  application = &application,
  queries = {},
  f.window {
    clickable_box {
      pos = bind f.vec3(60, 60, 0),
      ext = bind f.vec3(50, 50, 0),
      onclick = bindevent()
        app.shape = app.shape + 1
        C.printf("app shape is now %d\n", app.shape)
      end
    },
    clickable_box {
      pos = bind f.vec3(170, 60, 0),
      ext = bind f.vec3(50, 50, 0),
      onclick = bindevent()
        app.shape = app.shape - 1
        C.printf("app shape is now %d\n", app.shape)
      end
    },
    f.If (bind app.shape == 0) {
      f.rect {
        pos = bind f.vec3(280, 60, 0),
        ext = bind f.vec3(50, 50, 0),
        outline = bind f.Color {0xffdd3333},
        border = bind 10,
      }
    } :ElseIf (bind app.shape == 1) {
      f.circle {
        pos = bind f.vec3(280, 60, 0),
        ext = bind f.vec3(50, 50, 0),
        outline = bind f.Color {0xff22ee44},
        border = bind 10
      }
    } :Else {
      f.rect {
        pos = bind f.vec3(280, 60, 0),
        ext = bind f.vec3(50, 20, 0),
        outline = bind f.Color {0xff4433ee},
        border = bind 10
      }
    }
  }
}

return scaffolding.simple_application("if-operator", application, ui, {})
