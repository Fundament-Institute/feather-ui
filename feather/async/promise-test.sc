


using import .eventloop
using import .promise


fn can-wait()
    local l = (EventLoop)
    local s = (WaitScope l)

    let p =
        evalLater
            capture {}()
                2 + 2
    let q =
        'then p
            capture {}(x)
                2 * x

    let v =
        'wait q s

    assert (v == 8)

can-wait;
