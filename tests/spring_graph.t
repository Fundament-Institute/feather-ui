local C = terralib.includec "stdio.h"
local cond = require 'std.cond'

local f = require 'feather'
local scaffolding = require 'feather.scaffolding'
local spring = require 'feather.layouts.springgraph'
local dynarray = require 'feather.dynarray'

import 'feather/bind'


local struct springapp {
  nodes: dynarray(int)
  edges: dynarray(tuple(int,int))
                      }

terra springapp:update()
  
end

terra springapp:init(argc: int, argv: &rawstring)
  self.nodes:init()
  self.edges:init()
  for i = 0,3 do
    self.nodes:add(i)
  end
  self.edges:add({0,1})
  self.edges:add({1,2})
  
  self:update()
end
terra springapp:exit() end

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
    mousedown = bindevent()
      C.printf("in clickable_box mousedown\n")
      onclick()
      return 1
    end
  }
}

local ui = f.ui {
  application = &springapp,
  queries = {},
  f.window {
    spring.graph {
      gravity = 0.1,
      drag = 0.0,
      pos = bind f.vec3(200, 200, 0),
      f.each "node" (bind app.nodes) {
        clickable_box {
          pos = --[[bind f.vec3(node * 50, node * 50, 0)]]spring.node{
            id = bind node,
            mass = 1.0,
            charge = 50
            },
          ext = bind f.vec3(50f, 50f, 0f),
          onclick = bindevent()
            C.printf("in enablebutton onclick\n")
          end
        }
      },
      f.each "link" (bind app.edges) {
        f.line {
          points = spring.spring {
            pair = bind link,
            strength = 0.1,
            length = 200.0,
          },
          color = stroke_color
        },
      }
    }
    ,
  } 
}

return scaffolding.simple_application("springz", springapp, ui, {})
