fn linearstep (low high x)
    (clamp ((x - low) / (high - low)) 0.0 1.0)

do
    let linearstep
    locals;