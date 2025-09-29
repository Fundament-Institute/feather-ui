-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: 2025 Fundament Research Institute <https://fundament.institute>

local f = require("feather")
local abs, rel, px, NONE = f.abs, f.rel, f.px, f.NONE

local clockring = f.component {
	radius = f.required,
	width = f.required,
	progress = f.required,
	pos = f.required,
	color = f.required,
}(
	function(args)
		return f.shape.arc {
			props = {
				area = abs(args.pos.x, args.pos.y, args.pos.x + args.radius * 2.0, args.pos.y + args.radius * 2.0),
				anchor = abs(0.5, 0.5),
			},
			fill = args.color,
			innerRadius = args.radius - args.width,
			angles = { 0, args.progress * math.pi * 2.0 },
		}
	end
)

local clock = f.component {
	times = f.required,
	radius = f.required,
	pos = f.required,
	color = f.required,
}(function(args)
	return f.region {
		props = {
			area = f.FILL,
		},
		f.each("times", function(k, v)
			local radius = args.radius * 1.0 * v[2]

			return clockring {
				radius = radius,
				pos = args.pos,
				width = radius * 0.1,
				progress = v[1],
				color = args.color,
			}
		end, pairs(args.times)),
	}
end)

local function app(appstate)
	local clockcolor = 0xccccccff
	local w = f.window {
		title = "clock",
		resizable = true,
		clock {
			times = {
				{ appstate.time_hour / 24.0, 0.7 },
				{ appstate.time_min / 60.0, 0.85 },
				{ appstate.time_sec / 60.0, 1.0 },
			},
			radius = 200,
			pos = f.abs(100, 100),
			color = clockcolor,
		},
	}

	return appstate, w
end

return app, nil
