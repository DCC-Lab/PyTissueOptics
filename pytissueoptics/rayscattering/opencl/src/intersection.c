__constant float EPS_CATCH = 1e-7f;
__constant float EPS_BACK_CATCH = 2e-6f;
__constant float EPS_PARALLEL = 1e-6f;
__constant float EPS_SIDE = 3e-6f;
__constant float EPS = 1e-7;

struct Intersection {
    uint exists;
    float distance;
    float3 position;
    float3 normal;
    uint surfaceID;
    uint polygonID;
    float distanceLeft;
    bool isSmooth;
    float3 rawNormal;
};

typedef struct Intersection Intersection;

struct Ray {
    float3 origin;
    float3 direction;
    float length;
};

typedef struct Ray Ray;

struct GemsBoxIntersection {
    bool rayIsInside;
    bool exists;
    float3 position;
};

typedef struct GemsBoxIntersection GemsBoxIntersection;

// BVH node layout. Hand-written here (rather than generated from TreeNodeCL) so the Scene
// struct can reference TreeNode regardless of whether a particular kernel passes the BVH
// buffers. Must remain byte-compatible with TreeNodeCL.STRUCT_DTYPE in
// pytissueoptics/rayscattering/opencl/buffers/treeCL.py.
typedef struct {
  float3 bbox_min;
  float3 bbox_max;
  uint polygonCount;
  uint offset;
} TreeNode;

struct Scene{
    uint nSolids;
    __global Solid *solids;
    __global Surface *surfaces;
    __global Triangle *triangles;
    __global Vertex *vertices;
    __global SolidCandidate *solidCandidates;
    // BVH (optional). When nNodes == 0 the flat per-solid AABB path is used.
    uint nNodes;
    __global TreeNode *treeNodes;
    __global uint *leafPolygons;
};

typedef struct Scene Scene;

// Sentinel matching NO_CHILD in pytissueoptics/rayscattering/opencl/buffers/treeCL.py.
__constant uint TREE_NO_CHILD = 0xFFFFFFFFu;
// Maximum BVH depth we will encode per work-item in private memory. Must >= BVH_MAX_DEPTH on
// the Python side. 32 covers the default of 20 with comfortable headroom.
#define BVH_STACK_SIZE 32

GemsBoxIntersection _getBBoxIntersection(Ray ray, float3 minCornerVector, float3 maxCornerVector) {
    GemsBoxIntersection intersection;
    intersection.rayIsInside = true;
    intersection.exists = false;

    float rayOrigin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    float rayDirection[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    float minCorner[3] = {minCornerVector.x, minCornerVector.y, minCornerVector.z};
    float maxCorner[3] = {maxCornerVector.x, maxCornerVector.y, maxCornerVector.z};
    int quadrant[3];
    float candidatePlanes[3];

    for (uint i = 0; i < 3; i++) {
        if (rayOrigin[i] < minCorner[i]) {
            quadrant[i] = 0;
            candidatePlanes[i] = minCorner[i];
            intersection.rayIsInside = false;
        } else if (rayOrigin[i] > maxCorner[i]) {
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

    float maxT[3];
    for (uint i = 0; i < 3; i++) {
        if (quadrant[i] != 2 && rayDirection[i] != 0.0f) {
            maxT[i] = (candidatePlanes[i] - rayOrigin[i]) / rayDirection[i];
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

    float intersectionPoint[3];
    for (uint i = 0; i < 3; i++) {
        if (plane != i) {
            intersectionPoint[i] = rayOrigin[i] + maxT[plane] * rayDirection[i];
            if (intersectionPoint[i] < minCorner[i] || intersectionPoint[i] > maxCorner[i]) {
                return intersection;
            }
        } else {
            intersectionPoint[i] = candidatePlanes[i];
        }
    }

    intersection.exists = true;
    intersection.position = (float3)(intersectionPoint[0], intersectionPoint[1], intersectionPoint[2]);
    return intersection;
}

void _findBBoxIntersectingSolids(Ray ray, Scene *scene, uint gid, uint photonSolidID, uint ignoreSolidID) {

    for (uint i = 0; i < scene->nSolids; i++) {
        uint boxGID = gid * scene->nSolids + i;
        uint solidID = i + 1;
        scene->solidCandidates[boxGID].solidID = solidID;

        if (solidID == ignoreSolidID) {
            scene->solidCandidates[boxGID].distance = -1;
            continue;
        }

        if (solidID == photonSolidID) {
            scene->solidCandidates[boxGID].distance = 0;
            continue;
        }

        GemsBoxIntersection gemsIntersection = _getBBoxIntersection(ray, scene->solids[i].bbox_min, scene->solids[i].bbox_max);
        if (gemsIntersection.rayIsInside) {
            scene->solidCandidates[boxGID].distance = 0;
        } else if (!gemsIntersection.exists) {
            scene->solidCandidates[boxGID].distance = -1;
        } else {
            scene->solidCandidates[boxGID].distance = length(gemsIntersection.position - ray.origin);
        }
    }
}

void _sortSolidCandidates(Scene *scene, uint gid) {
    /*
    Simple bubble sort algorithm (kernel-friendly) to sort the solid candidates by distance.
    */
    for (uint i = 0; i < scene->nSolids; i++) {
        uint boxGID = gid * scene->nSolids + i;
        for (uint j = i + 1; j < scene->nSolids; j++) {
            uint boxGID2 = gid * scene->nSolids + j;
            if (scene->solidCandidates[boxGID].distance > scene->solidCandidates[boxGID2].distance) {
                SolidCandidate tmp = scene->solidCandidates[boxGID];
                scene->solidCandidates[boxGID] = scene->solidCandidates[boxGID2];
                scene->solidCandidates[boxGID2] = tmp;
            }
        }
    }
}

struct HitPoint {
    bool exists;
    float distance;
    float3 position;
};

typedef struct HitPoint HitPoint;

HitPoint _getTriangleIntersection(Ray ray, float3 v1, float3 v2, float3 v3, float3 normal) {
    HitPoint hitPoint;
    hitPoint.exists = false;

    float3 edgeA = v2 - v1;
    float3 edgeB = v3 - v1;
    float3 pVector = cross(ray.direction, edgeB);
    float det = dot(edgeA, pVector);

    bool rayIsParallel = fabs(det) < EPS_PARALLEL;
    if (rayIsParallel) {
        return hitPoint;
    }

    float invDet = 1.0f / det;
    float3 tVector = ray.origin - v1;
    float u = dot(tVector, pVector) * invDet;
    if (u < -EPS_SIDE || u > 1.0f) {
        return hitPoint;
    }

    float3 qVector = cross(tVector, edgeA);
    float v = dot(ray.direction, qVector) * invDet;
    if (v < -EPS_SIDE || u + v > 1.0f + EPS_SIDE) {
        return hitPoint;
    }

    float t = dot(edgeB, qVector) * invDet;
    hitPoint.distance = t;
    hitPoint.position = ray.origin + t * ray.direction;

    // Check if the intersection is slightly outside the triangle.
    float error = 0;
    if (u < -EPS){
        error -= u;
    }
    if (v < -EPS){
        error -= v;
    }
    if (u + v > 1.0 + EPS){
        error += u + v - 1.0;
    }
    if (error > 0){
        // Move the hit point towards the triangle center by this error factor.
        float3 correction = v1 + v2 + v3 - hitPoint.position * 3;
        hitPoint.position += 2.0f * error * correction;
    }

    if (t >= 0 && ray.length >= t){
        hitPoint.exists = true;
        return hitPoint;
    }

    float dt;
    if (t <= 0) {
        dt = t;
    } else {
        dt = t - ray.length;
    }
    float dt_T = fabs(dot(normal, ray.direction) * dt);

    if (t > ray.length && dt_T < EPS_CATCH) {
        // Forward catch.
        hitPoint.exists = true;
        return hitPoint;
    }

    if (t < 0 && (t > -EPS_BACK_CATCH || dt_T < EPS_CATCH)) {
        // Backward catch.
        hitPoint.exists = true;

        // If ray lies on the triangle, return a distance of 0 to prioritize this intersection.
        if (dt_T < EPS) {
            hitPoint.distance = 0;
        }
        return hitPoint;
    }

    return hitPoint;
}

void _testPolygonIntersection(Ray ray, uint polygonID, uint photonSolidID, uint ignoreSolidID,
                              __global Surface *surfaces, __global Triangle *triangles, __global Vertex *vertices,
                              Intersection *intersection, float *minSameSolidDistance) {
    // Tests one polygon and updates `intersection` (closest hit so far) and `minSameSolidDistance`
    // (used to cancel backward-catch hits when a same-solid hit is farther). Skips polygons whose
    // surface environment does not include the photon's current solid, or whose owning solid is
    // the one we have been asked to ignore (e.g. the last detector hit, to avoid re-detection).
    // ignoreSolidID == 0 disables the ignore-filter, matching the flat-list path's convention.
    uint surfaceID = triangles[polygonID].surfaceID;
    if (ignoreSolidID != 0 && surfaces[surfaceID].insideSolidID == (int)ignoreSolidID) {
        return;
    }
    if (photonSolidID != surfaces[surfaceID].insideSolidID && photonSolidID != surfaces[surfaceID].outsideSolidID) {
        return;
    }

    uint v0 = triangles[polygonID].vertexIDs[0];
    uint v1 = triangles[polygonID].vertexIDs[1];
    uint v2 = triangles[polygonID].vertexIDs[2];
    HitPoint hitPoint = _getTriangleIntersection(ray, vertices[v0].position, vertices[v1].position, vertices[v2].position, triangles[polygonID].normal);
    if (!hitPoint.exists) {
        return;
    }

    bool isGoingInside = dot(ray.direction, triangles[polygonID].normal) < 0;
    uint nextSolidID = isGoingInside ? surfaces[surfaceID].insideSolidID : surfaces[surfaceID].outsideSolidID;
    if (nextSolidID == photonSolidID) {
        if (hitPoint.distance > *minSameSolidDistance) {
            *minSameSolidDistance = hitPoint.distance;
        }
        return;
    }

    if (fabs(hitPoint.distance) < fabs(intersection->distance)) {
        intersection->exists = true;
        intersection->distance = hitPoint.distance;
        intersection->position = hitPoint.position;
        intersection->normal = triangles[polygonID].normal;
        intersection->surfaceID = surfaceID;
        intersection->polygonID = polygonID;
    }
}

void _resolveBackCatch(Intersection *intersection, float minSameSolidDistance) {
    if (intersection->distance == 0 && minSameSolidDistance == 0){
        // Cancel back catch. Surface overlap.
        intersection->exists = false;
    } else if (intersection->distance < 0) {
        // Cancel backward catch if the same-solid intersect distance is greater.
        if (minSameSolidDistance > intersection->distance + 1e-7) {
            intersection->exists = false;
        }
    }
}

Intersection _findClosestPolygonIntersection(Ray ray, uint solidID,
                                            __global Solid *solids, __global Surface *surfaces,
                                            __global Triangle *triangles, __global Vertex *vertices,
                                            uint photonSolidID) {
    Intersection intersection;
    intersection.exists = false;
    intersection.distance = INFINITY;

    float minSameSolidDistance = -INFINITY;

    for (uint s = solids[solidID-1].firstSurfaceID; s <= solids[solidID-1].lastSurfaceID; s++) {
        // When an interface joins a side surface, an outside photon could try to intersect with the interface
        //  while this is not allowed. So we skip these tests (where surface environments dont match the photon).
        if (photonSolidID != surfaces[s].insideSolidID && photonSolidID != surfaces[s].outsideSolidID) {
            continue;
        }

        for (uint p = surfaces[s].firstPolygonID; p <= surfaces[s].lastPolygonID; p++) {
            _testPolygonIntersection(ray, p, photonSolidID, 0, surfaces, triangles, vertices,
                                     &intersection, &minSameSolidDistance);
        }
    }

    _resolveBackCatch(&intersection, minSameSolidDistance);
    return intersection;
}

bool _bboxNodeWorthExploring(Ray ray, float3 bboxMin, float3 bboxMax, float closestDistance) {
    // Mirror of FastIntersectionFinder._nodeIsWorthExploring: keep the node if the ray hits its
    // bbox, AND the hit is at least as close as anything we have found so far. When the ray
    // origin is already inside the bbox we always recurse (the bbox intersect cannot return
    // a meaningful distance in that case).
    GemsBoxIntersection bb = _getBBoxIntersection(ray, bboxMin, bboxMax);
    if (bb.rayIsInside) {
        return true;
    }
    if (!bb.exists) {
        return false;
    }
    float bboxDistance = length(bb.position - ray.origin);
    return bboxDistance <= closestDistance;
}

Intersection _findIntersectionBVH(Ray ray, Scene *scene, uint photonSolidID, uint ignoreSolidID) {
    /*
    OpenCL counterpart of the Python FastIntersectionFinder, walked iteratively with a private
    DFS stack. The BVH is the flattened SpacePartition built by CLScene; each TreeNode is either
    a leaf (polygonCount > 0; offset indexes into leafPolygons) or an internal node (left child
    at idx+1; right child at offset, or absent if offset == TREE_NO_CHILD).
    */
    Intersection intersection;
    intersection.exists = false;
    intersection.distance = INFINITY;
    float minSameSolidDistance = -INFINITY;

    if (scene->nNodes == 0) {
        return intersection;
    }

    uint stack[BVH_STACK_SIZE];
    int stackTop = 0;
    stack[0] = 0;

    while (stackTop >= 0) {
        uint nodeIdx = stack[stackTop--];
        TreeNode node = scene->treeNodes[nodeIdx];

        if (!_bboxNodeWorthExploring(ray, node.bbox_min, node.bbox_max, intersection.distance)) {
            continue;
        }

        if (node.polygonCount > 0) {
            for (uint i = 0; i < node.polygonCount; i++) {
                uint polygonID = scene->leafPolygons[node.offset + i];
                _testPolygonIntersection(ray, polygonID, photonSolidID, ignoreSolidID,
                                         scene->surfaces, scene->triangles, scene->vertices,
                                         &intersection, &minSameSolidDistance);
            }
            continue;
        }

        // Internal node: push right first so left is popped first (DFS pre-order).
        uint rightChild = node.offset;
        if (rightChild != TREE_NO_CHILD && stackTop + 1 < BVH_STACK_SIZE) {
            stackTop++;
            stack[stackTop] = rightChild;
        }
        if (stackTop + 1 < BVH_STACK_SIZE) {
            stackTop++;
            stack[stackTop] = nodeIdx + 1;
        }
    }

    _resolveBackCatch(&intersection, minSameSolidDistance);
    return intersection;
}

float _cotangent(float3 v0, float3 v1, float3 v2) {
    float3 edge0 = v0 - v1;
    float3 edge1 = v2 - v1;
    float lengthCross = length(cross(edge1, edge0));
    if (lengthCross < EPS_SIDE) {
        lengthCross = EPS_SIDE;
    }
    return dot(edge1, edge0) / lengthCross;
}

void setSmoothNormal(Intersection *intersection, __global Triangle *triangles, __global Vertex *vertices, Ray *ray) {
    float3 newNormal;
    bool newNormalSet = false;

    // Check edge case where the intersection is directly on a vertex, in which case we just return the vertex normal.
    for (uint i = 0; i < 3; i++) {
        float3 vertex = vertices[triangles[intersection->polygonID].vertexIDs[i]].position;
        if (length(intersection->position - vertex) < EPS_SIDE) {
            newNormal = vertices[triangles[intersection->polygonID].vertexIDs[i]].normal;
            newNormalSet = true;
            break;
        }
    }
    
    if (!newNormalSet) {
        // Compute the new smooth normal as a weighted average of the vertex normals.
        float weights[3];
        for (uint i = 0; i < 3; i++) {
            float3 vertex = vertices[triangles[intersection->polygonID].vertexIDs[i]].position;
            float3 prevVertex = vertices[triangles[intersection->polygonID].vertexIDs[(i + 2) % 3]].position;
            float3 nextVertex = vertices[triangles[intersection->polygonID].vertexIDs[(i + 1) % 3]].position;
            float cotPrev = _cotangent(intersection->position, vertex, prevVertex);
            float cotNext = _cotangent(intersection->position, vertex, nextVertex);
            float d = length(vertex - intersection->position);
            weights[i] = (cotPrev + cotNext) / (d * d);
        }

        float sum = weights[0] + weights[1] + weights[2];
        for (uint i = 0; i < 3; i++) {
            weights[i] /= sum;
        }

        newNormal = weights[0] * vertices[triangles[intersection->polygonID].vertexIDs[0]].normal +
                    weights[1] * vertices[triangles[intersection->polygonID].vertexIDs[1]].normal +
                    weights[2] * vertices[triangles[intersection->polygonID].vertexIDs[2]].normal;
    }

    // Do not allow the new smooth normal to have a different dot product with ray direction. 
    // This is a rare edge case that can happen when the ray direction is approximately parallel to the surface. More common in low resolution meshes (like icosphere of order 1)
    // Not accounting for this can lead to a photon slightly going inside another solid mesh, but being considered as leaving the other solid (during FresnelIntersection calculations).
    // Which would result in the wrong next environment being set as well as the wrong step correction being applied after refraction.
    if (dot(newNormal, ray->direction) * dot(intersection->normal, ray->direction) < 0) {
        intersection->isSmooth = false;
        return;
    }
    intersection->normal = normalize(newNormal);

    intersection->isSmooth = true;
    intersection->rawNormal = triangles[intersection->polygonID].normal;
}

void _composeIntersection(Intersection *intersection, Ray *ray, Scene *scene) {
    if (!intersection->exists) {
        return;
    }

    intersection->isSmooth = false;
    if (scene->surfaces[intersection->surfaceID].toSmooth) {
        setSmoothNormal(intersection, scene->triangles, scene->vertices, ray);
    }
    intersection->distanceLeft = ray->length - intersection->distance;
}

Intersection findIntersection(Ray ray, Scene *scene, uint gid, uint photonSolidID, uint ignoreSolidID) {
    /*
    Dispatch to BVH traversal (FastIntersectionFinder-equivalent) when CLScene built a tree;
    otherwise fall back to the flat per-solid AABB sort (SimpleIntersectionFinder-equivalent).
    The selection is made on the host side via CLScene; see opencl/CLScene.py.
    */
    if (scene->nNodes > 0) {
        Intersection bvhIntersection = _findIntersectionBVH(ray, scene, photonSolidID, ignoreSolidID);
        _composeIntersection(&bvhIntersection, &ray, scene);
        return bvhIntersection;
    }

    _findBBoxIntersectingSolids(ray, scene, gid, photonSolidID, ignoreSolidID);
    _sortSolidCandidates(scene, gid);

    Intersection closestIntersection;
    closestIntersection.exists = false;
    closestIntersection.distance = INFINITY;
    if (scene->nSolids == 0) {
        return closestIntersection;
    }

    for (uint i = 0; i < scene->nSolids; i++) {
        uint boxGID = gid * scene->nSolids + i;
        if (scene->solidCandidates[boxGID].distance == -1) {
            // Default buffer value -1 means that there is no intersection with this solid
            continue;
        }

        if (scene->solidCandidates[boxGID].distance > closestIntersection.distance) {
            // The solid candidates are sorted by distance, so we can break early if the BBox distance
            // is greater than the closest intersection found so far.
            break;
        }

        uint solidID = scene->solidCandidates[boxGID].solidID;
        Intersection intersection = _findClosestPolygonIntersection(ray, solidID, scene->solids, scene->surfaces, scene->triangles, scene->vertices, photonSolidID);
        if (intersection.exists && intersection.distance < closestIntersection.distance) {
            closestIntersection = intersection;
        }
    }

    _composeIntersection(&closestIntersection, &ray, scene);
    return closestIntersection;
}

// ----------------- TEST KERNELS -----------------

__kernel void findIntersections(__global Ray *rays, uint nSolids, __global Solid *solids, __global Surface *surfaces,
        __global Triangle *triangles, __global Vertex *vertices, __global SolidCandidate *solidCandidates,
        uint photonSolidID, uint ignoreSolidID, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    // Trailing fields (nNodes/treeNodes/leafPolygons) are zero-initialized by C99 brace
    // semantics, so dispatch in findIntersection() picks the flat path.
    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates};
    intersections[gid] = findIntersection(rays[gid], &scene, gid, photonSolidID, ignoreSolidID);
}


__kernel void findIntersectionsBVH(__global Ray *rays, uint nSolids, __global Solid *solids, __global Surface *surfaces,
        __global Triangle *triangles, __global Vertex *vertices, __global SolidCandidate *solidCandidates,
        uint nNodes, __global TreeNode *treeNodes, __global uint *leafPolygons,
        uint photonSolidID, uint ignoreSolidID, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates,
                   nNodes, treeNodes, leafPolygons};
    intersections[gid] = findIntersection(rays[gid], &scene, gid, photonSolidID, ignoreSolidID);
}


__kernel void setSmoothNormals(__global Intersection *intersections, __global Triangle *triangles, __global Vertex *vertices, __global Ray *rays) {
    uint gid = get_global_id(0);
    Intersection intersection = intersections[gid];
    Ray ray = rays[gid];

    setSmoothNormal(&intersection, triangles, vertices, &ray);

    intersections[gid] = intersection;
    rays[gid] = ray;
}
