
struct Intersection {
    int status;
    float distance;
};

typedef struct Intersection Intersection;

struct Ray {
    float4 origin;
    float4 direction;
    float distance;
};

typedef struct Ray Ray;


void _findBBoxIntersectingSolids(Ray ray,
        __global Solid *solids, __global BBoxIntersection *bboxIntersections, uint gid){
    // uint id = gid + solidID * nSolids;
    uint id = gid;
    bboxIntersections[id].distance = 1.5;
    bboxIntersections[id].solidID = 7;
    printf("nSolids = %d\n", sizeof(solids) / 8);
    printf("solid bbox = (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n", solids[0].bbox_min[0], solids[0].bbox_min[1], solids[0].bbox_min[2], solids[0].bbox_max[0], solids[0].bbox_max[1], solids[0].bbox_max[2]);
    // for each solid, if no bbox, make sure to reset the result to default 'none'.
}

Intersection findIntersection(Ray ray,
        __global Solid *solids, __global BBoxIntersection *bboxIntersections, uint gid) {
    // need BBoxIntersectionResultBuffer of size (n_work_units * n_solids)
    _findBBoxIntersectingSolids(ray,
                                solids, bboxIntersections, gid);
    // >>> locally sort bboxIntersectionResultBuffer (using kernel gid)
    // this will require custom sort algo to sort the correct buffer IDs ...

//    for (i=0; i++; i<n_solids){
//        Intersection intersection = _findClosestPolygonIntersection(ray, solid_id, scene);
//        if (intersection.status == 1){
//            return _composeIntersection(ray, intersection);
//        }
//    }

    Intersection intersection;
    intersection.status = 0;
    intersection.distance = 1.5;
    return intersection;
}

// ----------------- TEST KERNELS -----------------

__kernel void findIntersections(uint n_work_units, __global Ray *rays, __global Solid *solids,
        __global BBoxIntersection *bboxIntersections, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    intersections[gid] = findIntersection(rays[gid], solids, bboxIntersections, gid);
}
