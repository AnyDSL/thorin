; Module generic-thorin runtime decls

declare i8* @thorin_malloc(i64);
declare void @thorin_free(i8*);
declare i64 @map_memory(i32, i32, i8*, i32, i32, i32, i32, i32, i32);
declare void @unmap_memory(i32, i32, i64);

declare i64 @parallel_create(i32, i8*, i64, i8*);
declare void @parallel_join(i64);

