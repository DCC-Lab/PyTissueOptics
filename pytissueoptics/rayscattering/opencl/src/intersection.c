
struct Intersection {
    int status;
    float distance;
};

typedef struct Intersection Intersection;

struct Ray {
    float4 origin;
    float4 direction;
    float length;
};

typedef struct Ray Ray;

struct GemsBoxIntersection {
    bool rayIsInside;
    bool exists;
    float3 position;
};

typedef struct GemsBoxIntersection GemsBoxIntersection;


GemsBoxIntersection _getBBoxIntersection(Ray ray, float3 minCorner, float3 maxCorner) {
    GemsBoxIntersection intersection;
    intersection.rayIsInside = true;
    intersection.exists = false;

    int quadrant[3];
    float3 candidatePlanes;
    for (uint i = 0; i < 3; i++) {
        if (ray.origin[i] < minCorner[i]) {
            quadrant[i] = 0;
            candidatePlanes[i] = minCorner[i];
            intersection.rayIsInside = false;
        } else if (ray.origin[i] > maxCorner[i]) {
            quadrant[i] = 1;
            candidatePlanes[i] = maxCorner[i];
            intersection.rayIsInside = false;
        } else {
            quadrant[i] = 2;
        }
    }

    if (intersection.rayIsInside) {
        return intersection;
    }

    float3 maxT;
    for (uint i = 0; i < 3; i++) {
        if (quadrant[i] != 2 && ray.direction[i] != 0.0f) {
            maxT[i] = (candidatePlanes[i] - ray.origin[i]) / ray.direction[i];
        } else {
            maxT[i] = -1.0f;
        }
    }

    uint plane = 0;
    for (uint i = 1; i < 3; i++) {
        if (maxT[plane] < maxT[i]) {
            plane = i;
        }
    }

    if (maxT[plane] < 0.0f) {
        return intersection;
    }
    if (maxT[plane] > ray.length) {
        return intersection;
    }
    for (uint i = 0; i < 3; i++) {
        if (plane != i) {
            intersection.position[i] = ray.origin[i] + maxT[plane] * ray.direction[i];
            if (intersection.position[i] < minCorner[i] || intersection.position[i] > maxCorner[i]) {
                return intersection;
            }
        } else {
            intersection.position[i] = candidatePlanes[i];
        }
    }

    intersection.exists = true;
    return intersection;
}

void _findBBoxIntersectingSolids(Ray ray,
        __global Solid *solids, __global BBoxIntersection *bboxIntersections, uint gid){

    const uint nSolids = sizeof(solids) / 8;

    for (uint i = 0; i < nSolids; i++) {
        GemsBoxIntersection bboxIntersection = _getBBoxIntersection(ray, solids[i].bbox_min, solids[i].bbox_max);
        printf("Intersection with Solid ID %d : (isInside=%d, exists=%d, position=(%f, %f, %f))\n",
                i, bboxIntersection.rayIsInside, bboxIntersection.exists, bboxIntersection.position.x, bboxIntersection.position.y, bboxIntersection.position.z);
    }

    // uint id = gid + solidID * nSolids;
//    uint id = gid;
//    bboxIntersections[id].distance = 1.5;
//    bboxIntersections[id].solidID = 7;
//    printf("nSolids = %d\n", sizeof(solids) / 8);
//    printf("solid bbox = (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)\n",
//            solids[0].bbox_min[0], solids[0].bbox_min[1], solids[0].bbox_min[2],
//            solids[0].bbox_max[0], solids[0].bbox_max[1], solids[0].bbox_max[2]);
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
