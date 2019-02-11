namespace thorin {
    enum class KernelArgType : uint8_t { Val = 0, Ptr, Struct };

    static const char* runtime_definitions = R"(
        ; Module anydsl runtime decls
        declare noalias [0 x i8]* @anydsl_alloc(i32, i64);
        declare noalias [0 x i8]* @anydsl_alloc_unified(i32, i64);
        declare void @anydsl_release(i32, i8*);
        declare void @anydsl_launch_kernel(i32, i8*, i8*, i32*, i32*, i8**, i32*, i32*, i8*, i32);
        declare void @anydsl_parallel_for(i32, i32, i32, i8*, i8*);
        declare i32  @anydsl_spawn_thread(i8*, i8*);
        declare void @anydsl_sync_thread(i32);
        declare i32  @anydsl_create_graph();
        declare i32  @anydsl_create_task(i32, { i8*, i64 });
        declare void @anydsl_create_edge(i32, i32);
        declare void @anydsl_execute_graph(i32, i32);
    )";
}
