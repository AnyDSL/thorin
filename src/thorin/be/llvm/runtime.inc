namespace thorin {
    enum class KernelArgType : uint8_t { Val = 0, Ptr, Struct, LSM };

    static const char* runtime_definitions = R"(
        ; Module anydsl runtime decls
        declare noalias ptr @anydsl_alloc(i32, i64);
        declare noalias ptr @anydsl_alloc_unified(i32, i64);
        declare void @anydsl_release(i32, ptr);
        declare void @anydsl_launch_kernel(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32);
        declare void @anydsl_parallel_for(i32, i32, i32, ptr, ptr);
        declare void @anydsl_fibers_spawn(i32, i32, i32, ptr, ptr);
        declare i32  @anydsl_spawn_thread(ptr, ptr);
        declare void @anydsl_sync_thread(i32);
        declare i32  @anydsl_create_graph();
        declare i32  @anydsl_create_task(i32, { ptr, i64 });
        declare void @anydsl_create_edge(i32, i32);
        declare void @anydsl_execute_graph(i32, i32);
    )";
}
