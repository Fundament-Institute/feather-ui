using import Array
using import Map
using import String
using import enum
using import spicetools
using import itertools

let cjson =
    include
        """"
            #include "cjson/cJSON.h"

load-library "libcjson.so"

# for x in json.extern do
    print x
enum json
let json-string = String
let json-array = (GrowingArray json)
let json-object = (Map Symbol json)
enum json
    array : json-array
    object : json-object
    string : json-string
    number : f64
    boolean : bool
    null

    inline __rimply (T cls)
        static-match T
        case json-array
            inline (value) (this-type.array value)
        case json-object
            inline (value) (this-type.object value)
        case json-string
            inline (value) (this-type.string value)
        case f64
            inline (value) (this-type.number value)
        case f32
            inline (value) (this-type.number value)
        case i64
            inline (value) (this-type.number (value imply f64))
        case i32
            inline (value) (this-type.number value)
        case bool
            inline (value) (this-type.boolean value)
        case NullType
            inline (value) (this-type.null)
        default ()

sugar json-square-list (args...)
    fn do-match (args...)
        returning list
        sugar-match args...
        case (val ', rest...)
            cons val (this-function rest...)
        case (val)
            cons val '()
        default
            error@ unknown-anchor "while expanding json square list" "json syntax is a comma separated list of values"
    list
        json.array
        cons
            json-array
            do-match args...

sugar json-curly-list (args...)
    let obj = (Symbol "#json-object-symbol")
    fn do-match (args...)
        returning list
        sugar-match args...
        case ((name as string) ': val ', rest...)
            cons
                list ''set obj (list sugar-quote (Symbol name)) val
                this-function rest...
        case ((name as string) ': val)
            list
                list ''set obj (list sugar-quote (Symbol name)) val
        default
            error@ unknown-anchor "while expanding json curly list to object" "json object syntax is a comma separated list of key : value pairs"
    list
        json.object
        ..
            cons
                do
                list local obj '= (list json-object)
                do-match args...
            list obj

fn gen-json-matcher (failfunc expr scope params)
    returning Value Scope
    # report params
    if (('typeof params) == Symbol)
        if ('constant? params)
            let id = (params as Symbol)
            if ('variadic? id)
                error "NYI: json variadic symbols matching"
            else
                _
                    expr
                    'bind scope id expr
        else
            error "nonconstant symbol in syntax?"
    elseif (('kind params) == value-kind-const-string)
        let lit = (params as string)
        _
            spice-quote
                if (expr != lit)
                    failfunc;
            scope
    elseif (('typeof params) == list)

        params as:= list
        let matcher-recurse = this-function
        sugar-match params
        case ('curly-list rest...)
            fn object-pattern (failfunc expr scope fields unpackexprs)
                returning Value Scope
                sugar-match fields
                case ((name as string) ': val ', rest...)
                    let fieldexpr =
                        spice-quote
                            dispatch expr
                            case object (obj)
                                try
                                    'get obj [(Symbol name)]
                                else
                                    failfunc;
                            default
                                failfunc;
                    let subunpackexprs scope2 =
                        matcher-recurse failfunc fieldexpr scope val
                    sc_expression_append unpackexprs subunpackexprs
                    let unpackexprs scope3 =
                        this-function failfunc expr scope2 rest... unpackexprs
                    _ unpackexprs scope3
                case ((name as string) ': val)
                    let fieldexpr =
                        spice-quote
                            dispatch expr
                            case object (obj)
                                try
                                    'get obj [(Symbol name)]
                                else
                                    failfunc;
                            default
                                failfunc;
                    let subunpackexprs scope2 =
                        matcher-recurse failfunc fieldexpr scope val
                    sc_expression_append unpackexprs subunpackexprs
                    _ unpackexprs scope2
                default
                    error "json object syntax is a comma separated list of key : value pairs"
            object-pattern failfunc expr scope rest... (sc_expression_new)
        case ('square-list rest...)
            error "NYI: json list matching"
        case ((id as Symbol) 'as (kind as Symbol))
            sugar-eval
                let fields = '(array object string number boolean)
                let matchers =
                    # vvv report
                    ..
                        vvv list
                        qq
                            switch kind

                        vvv 'reverse
                        ->> (fields as Generator)
                            map
                                inline (name)
                                    qq
                                        case (sugar-quote [name])
                                            let extractor =
                                                spice-quote
                                                    dispatch expr
                                                    case [name] (arg)
                                                        arg
                                                    default
                                                        failfunc;
                                            _ extractor ('bind scope id extractor)
                            'cons-sink '()
                        vvv list
                        qq
                            default
                                error [("unknown json field kind for match as; must be one of " .. (tostring fields))]

                cons 'do matchers


        default
            error "NYI: json matching for that pattern"
    else
        error "json match pattern element must be a symbol, a constant string, a json list pattern, or a json object pattern"

run-stage;
do
    let json json-object json-array

    fn parse(str)
        let cj =
            cjson.extern.cJSON_Parse str
        defer cjson.extern.cJSON_Delete cj
        fn convert(cj)
            returning (uniqueof json -1)
            raising Error
            if (cj.type == 0)
                error "invalid cjson value"
            elseif (cj.type == 1)
                json.boolean false
            elseif (cj.type == 2)
                json.boolean true
            elseif (cj.type == 4)
                json.null;
            elseif (cj.type == 8)
                json.number cj.valuedouble
            elseif (cj.type == 16)
                json.string (json-string cj.valuestring)
            elseif (cj.type == 32)
                json.array
                    do
                        local arr = (json-array)
                        loop (elem = cj.child)
                            if (elem == null)
                                break arr
                            else
                                'append arr (this-function elem)
                                repeat elem.next
            elseif (cj.type == 64)
                json.object
                    do
                        local obj = (json-object)
                        loop (elem = cj.child)
                            if (elem == null)
                                break obj
                            else
                                'set obj (Symbol elem.string) (this-function elem)
                                _ elem.next
            elseif (cj.type == 128)
                error "unable to convert cjson raw value"
            else
                error "unknown cjson value type"
        convert cj

    fn serialize(j)
        fn convert(j)
            viewing j
            returning (mutable@ cjson.typedef.cJSON)
            dispatch j
            case array (arr)
                let cj-arr = (cjson.extern.cJSON_CreateArray)
                if (cj-arr == null)
                    error "unable to allocate cjson object"
                for item in arr
                    let cj-item = (this-function item)
                    # TODO: make a more efficient version of this
                    cjson.extern.cJSON_AddItemToArray cj-arr cj-item
                cj-arr
            case object (obj)
                let cj-obj = (cjson.extern.cJSON_CreateObject)
                for name value in obj
                    # TODO: make a more efficient version of this
                    cjson.extern.cJSON_AddItemToObject cj-obj (name as string as rawstring) (this-function value)
                cj-obj
            case string (str)
                cjson.extern.cJSON_CreateString str
            case number (n)
                cjson.extern.cJSON_CreateNumber n
            case boolean (b)
                cjson.extern.cJSON_CreateBool b
            default
                unreachable;

        let cj = (convert j)
        let str = (cjson.extern.cJSON_Print cj)
        defer free str
        String str

    sugar literal (body...)
        cons
            do
            list let 'square-list '= json-square-list
            list let 'curly-list '= json-curly-list
            body...

    let json-match =
        gen-match-block-parser gen-json-matcher


    locals;
