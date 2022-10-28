
struct Intersection {
    uint exists;
    float distance;
    float3 position;
    float3 normal;
    uint surfaceID;
    uint polygonID;
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
    float EPSILON = 0.00001f;

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
        for (uint s = solids[solidID].firstSurfaceID; s <= solids[solidID].lastSurfaceID; s++) {
            for (uint p = surfaces[s].firstPolygonID; p <= surfaces[s].lastPolygonID; p++) {
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
                    intersection.polygonID = p;
                }
            }
        }
        return intersection;
}

float _cotangent(float3 v0, float3 v1, float3 v2) {
    float3 edge0 = v0 - v1;
    float3 edge1 = v2 - v1;
    return dot(edge1, edge0) / length(cross(edge1, edge0));
}

void setSmoothNormal(Intersection *intersection, __global Triangle *triangles, __global Vertex *vertices) {
    float3 weights;
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
    weights /= sum;

    intersection->normal = weights[0] * vertices[triangles[intersection->polygonID].vertexIDs[0]].normal +
                           weights[1] * vertices[triangles[intersection->polygonID].vertexIDs[1]].normal +
                           weights[2] * vertices[triangles[intersection->polygonID].vertexIDs[2]].normal;
    intersection->normal = normalize(intersection->normal);
}

void _composeIntersection(Intersection *intersection, Ray *ray, __global Surface *surfaces,
                            __global Triangle *triangles, __global Vertex *vertices) {
    if (!intersection->exists) {
        return;
    }

    if (surfaces[intersection->surfaceID].toSmooth) {
        setSmoothNormal(intersection, triangles, vertices);
    }
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
    if (nSolids == 0) {
        return closestIntersection;
    }

    for (uint i = 0; i < nSolids; i++) {
        uint boxGID = gid * nSolids + i;
        if (solidCandidates[boxGID].distance == -1) {
            continue;
        }
        bool contained = solidCandidates[boxGID].distance == 0;
        if (!contained && closestIntersection.exists) {
            break;
        }

        uint solidID = solidCandidates[boxGID].solidID;
        Intersection intersection = _findClosestPolygonIntersection(ray, solidID, solids, surfaces, triangles, vertices);
        if (intersection.exists  && intersection.distance < closestIntersection.distance) {
            closestIntersection = intersection;
        }
    }

    _composeIntersection(&closestIntersection, &ray, surfaces, triangles, vertices);
    return closestIntersection;
}

// ----------------- TEST KERNELS -----------------

__kernel void findIntersections(uint nSolids, __global Ray *rays, __global Solid *solids, __global Surface *surfaces,
        __global Triangle *triangles, __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    intersections[gid] = findIntersection(rays[gid], nSolids, solids, surfaces, triangles, vertices, solidCandidates, gid);
}
