
struct Intersection {
    uint exists;
    float distance;
    float3 position;
    float3 normal;
    uint surfaceID;
    float distanceLeft;
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


GemsBoxIntersection _getSolidCandidate(Ray ray, float3 minCorner, float3 maxCorner) {
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

void _findBBoxIntersectingSolids(Ray ray, uint nSolids,
        __global Solid *solids, __global SolidCandidate *solidCandidates, uint gid){

    for (uint i = 0; i < nSolids; i++) {
        uint boxGID = gid * nSolids + i;
        solidCandidates[boxGID].solidID = i;

        GemsBoxIntersection gemsIntersection = _getSolidCandidate(ray, solids[i].bbox_min, solids[i].bbox_max);
//        printf("Intersection with Solid ID %d : (isInside=%d, exists=%d, position=(%f, %f, %f))\n",
//                i, gemsIntersection.rayIsInside, gemsIntersection.exists, gemsIntersection.position.x, gemsIntersection.position.y, gemsIntersection.position.z);
        if (gemsIntersection.rayIsInside) {
            solidCandidates[boxGID].distance = 0;
        } else if (!gemsIntersection.exists) {
            solidCandidates[boxGID].distance = -1;
        } else {
            solidCandidates[boxGID].distance = length(gemsIntersection.position - ray.origin.xyz);
        }
    }
}

void _sortSolidCandidates(__global SolidCandidate *solidCandidates, uint gid, uint nSolids) {
    for (uint i = 0; i < nSolids; i++) {
        uint boxGID = gid * nSolids + i;
        for (uint j = i + 1; j < nSolids; j++) {
            uint boxGID2 = gid * nSolids + j;
            if (solidCandidates[boxGID].distance > solidCandidates[boxGID2].distance) {
                SolidCandidate tmp = solidCandidates[boxGID];
                solidCandidates[boxGID] = solidCandidates[boxGID2];
                solidCandidates[boxGID2] = tmp;
            }
        }
    }
}

struct HitPoint {
    bool exists;
    float3 position;
};

typedef struct HitPoint HitPoint;

HitPoint _getTriangleIntersection(Ray ray, float3 v1, float3 v2, float3 v3) {
    float3 rayDirection = ray.direction.xyz;
    float3 rayOrigin = ray.origin.xyz;
    float EPSILON = 0.000001f;

    HitPoint hitPoint;
    hitPoint.exists = false;

    float3 edgeA = v2 - v1;
    float3 edgeB = v3 - v1;
    float3 pVector = cross(rayDirection, edgeB);
    float det = dot(edgeA, pVector);

    bool rayIsParallel = fabs(det) < EPSILON;
    if (rayIsParallel) {
        return hitPoint;
    }

    float invDet = 1.0f / det;
    float3 tVector = rayOrigin - v1;
    float u = dot(tVector, pVector) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return hitPoint;
    }

    float3 qVector = cross(tVector, edgeA);
    float v = dot(rayDirection, qVector) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return hitPoint;
    }

    float t = dot(edgeB, qVector) * invDet;
    if (t < EPSILON) {
        return hitPoint;
    }

    if (t > ray.length) {
        return hitPoint;
    }

    hitPoint.exists = true;
    hitPoint.position = rayOrigin + t * rayDirection;
    return hitPoint;
}

Intersection _findClosestPolygonIntersection(Ray ray, uint solidID,
                                            __global Solid *solids, __global Surface *surfaces,
                                            __global Triangle *triangles, __global Vertex *vertices) {
        Intersection intersection;
        intersection.exists = false;
        intersection.distance = INFINITY;
//        printf("This solid (%d) has %d surfaces (ID %d to %d)\n", solidID, solids[solidID].lastSurfaceID + 1 - solids[solidID].firstSurfaceID,
//                solids[solidID].firstSurfaceID, solids[solidID].lastSurfaceID);
        for (uint s = solids[solidID].firstSurfaceID; s <= solids[solidID].lastSurfaceID; s++) {
//            printf("    Surface %d has %d polygons (ID %d to %d)\n", s, surfaces[s].lastPolygonID + 1 - surfaces[s].firstPolygonID,
//                    surfaces[s].firstPolygonID, surfaces[s].lastPolygonID);
            for (uint p = surfaces[s].firstPolygonID; p <= surfaces[s].lastPolygonID; p++) {
//                printf("        Triangle %d has 3 vertices (ID: %d, %d, %d)\n", p, triangles[p].vertexIDs[0], triangles[p].vertexIDs[1], triangles[p].vertexIDs[2]);
                uint vertexIDs[3] = {triangles[p].vertexIDs[0], triangles[p].vertexIDs[1], triangles[p].vertexIDs[2]};
                HitPoint hitPoint = _getTriangleIntersection(ray, vertices[vertexIDs[0]].position, vertices[vertexIDs[1]].position, vertices[vertexIDs[2]].position);
                if (!hitPoint.exists) {
                    continue;
                }
                float distance = length(hitPoint.position - ray.origin.xyz);
                if (distance < intersection.distance) {
                    intersection.exists = true;
                    intersection.distance = distance;
                    intersection.position = hitPoint.position;
                    intersection.normal = triangles[p].normal;
                    intersection.surfaceID = s;
                }
            }
        }
//        printf("Intersection with Solid ID %d : (exists=%d, distance=%f, position=(%f, %f, %f), polygonID=%d)\n",
//                solidID, intersection.exists, intersection.distance, intersection.position.x, intersection.position.y, intersection.position.z, intersection.polygonID);
        return intersection;
}

void _composeIntersection(Intersection *intersection, Ray *ray) {
    if (!intersection->exists) {
        return;
    }
    // todo: smoothing & environments
    intersection->distanceLeft = ray->length - intersection->distance;
}

Intersection findIntersection(Ray ray, uint nSolids, __global Solid *solids,
        __global Surface *surfaces, __global Triangle *triangles, __global Vertex *vertices,
        __global SolidCandidate *solidCandidates, uint gid) {
    _findBBoxIntersectingSolids(ray,
                                nSolids, solids, solidCandidates, gid);
    _sortSolidCandidates(solidCandidates, gid, nSolids);

    Intersection closestIntersection;
    closestIntersection.exists = false;
    closestIntersection.distance = INFINITY;
    for (uint i = 0; i < nSolids; i++) {
        uint boxGID = gid * nSolids + i;
        if (solidCandidates[boxGID].distance == -1) {
//            printf("Skipping Solid %d\n", solidCandidates[boxGID].solidID);
            continue;
        }
        bool contained = solidCandidates[boxGID].distance == 0;
        if (!contained && closestIntersection.exists) {
            break;
        }

        uint solidID = solidCandidates[boxGID].solidID;
//        printf("Testing polygons of Solid %d\n", solidID);
        Intersection intersection = _findClosestPolygonIntersection(ray, solidID, solids, surfaces, triangles, vertices);
        if (intersection.exists  && intersection.distance < closestIntersection.distance) {
            closestIntersection = intersection;
        }
    }

    _composeIntersection(&closestIntersection, &ray);
    return closestIntersection;
}

// ----------------- TEST KERNELS -----------------

__kernel void findIntersections(uint nSolids, __global Ray *rays, __global Solid *solids, __global Surface *surfaces,
        __global Triangle *triangles, __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    intersections[gid] = findIntersection(rays[gid], nSolids, solids, surfaces, triangles, vertices, solidCandidates, gid);
}
