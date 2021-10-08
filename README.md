# Thorin
The Higher-Order Intermediate Representation

## Build Instructions

See [Build Instructions](https://anydsl.github.io/Build-Instructions).

## Documentation

See our [AnyDSL/Thorin Website](https://anydsl.github.io/Thorin).

## Syntax

```ebnf
(* nominals *)
n = lam ID ":" e ["=" e "," e] ";"
  | sig ID ":" e (["=" e "," ... "," e]) | ("(" N ")") ";"

e = e e                             (* application *)
  | ex(e, e)                        (* extract *)
  | ins(e, e, e)                    (* insert *)
  | "[" e "," ... "," e "]"         (* sigma *)
  | "(" e "," ... "," e")" [":" e]  (* tuple *)
  | ID: e = e; e                    (* let *)
```
