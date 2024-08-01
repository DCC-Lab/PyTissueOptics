// ----------------- TEST KERNELS -----------------

 __kernel void test_global_id(__global int *buffer){
    int id = get_global_id(0);
    buffer[id] = id;
}

 __kernel void test_local_id(__global int *buffer){
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    buffer[gid] = lid;
}


 __kernel void test_extract_many_ids(__global int *buffer){
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lgid = get_group_id(0);
    int ls = get_local_size(0);
    int offset = get_global_offset(0);
    buffer[gid] = gid+lid*100+ls*10000+lgid*1000000+offset*100000000;
}

 __kernel void test_compute_global_id_from_local_id(__global int *buffer){
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lgid = get_group_id(0);
    int ls = get_local_size(0);
    int offset = get_global_offset(0);
    buffer[gid] = ls * lgid + lid ;
}

 __kernel void test_compute_global_id_from_local_id_nonuniform(__global int *buffer, __global int *local_sizes){
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lgid = get_group_id(0);
    int ls = get_local_size(0);
    local_sizes[lgid] = ls;
    int i = 0;
    int total_previous = 0;
    for (i = 0; i < lgid; i++) {
        total_previous += local_sizes[i];
    }
    int offset = get_global_offset(0);
    buffer[gid] = total_previous + lid + offset;
}
