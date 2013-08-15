type var = int
type value = int
type u32 = int
type frame = unit
type u1 = int
type mem = var -> value
val emp : mem = fn _ => 0
fun print_res (a:mem, b:u32) = print (Int.toString b ^ "\n")
fun select (a, b, c) = if a <> 0 then b else c
fun add (a, b) = a + b
fun mul (a, b) = a * b
fun cmp_eq (a, b) = if a = b then 1 else 0
