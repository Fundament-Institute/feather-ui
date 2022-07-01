using import .json

let val =
    parse "{\"foo\" : [ 1, 2, 3 ]}"

# let val =
    parse "[ 1, 2, 3 ]"

# print val


let ser =
    serialize val

let foo =
    literal
        {
            "foo" : [ 3, 2, 1 ],
            "bar" : [ 4, 5, 6 ]
        }

# print ser
# print (serialize foo)

json-match (view foo)
case {"foo" : (f as array) }
    print f
default
    print "not found"
