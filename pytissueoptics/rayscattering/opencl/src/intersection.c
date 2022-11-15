const float EPSILON = 0.00001f;

struct Intersection {
    uint exists;
    uint isTooClose;
    float distance;
    float3 position;
    float3 normal;
    uint surfaceID;
    uint polygonID;
    float distanceLeft;
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

struct Scene{
    uint nSolids;
    __global Solid *solids;
    __global Surface *surfaces;
    __global Triangle *triangles;
    __global Vertex *vertices;
    __global SolidCandidate *solidCandidates;
};

typedef struct Scene Scene;

GemsBoxIntersection _getSolidCandidate(Ray ray, float3 minCornerVector, float3 maxCornerVector) {
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

void _findBBoxIntersectingSolids(Ray ray, Scene *scene, uint gid){

    for (uint i = 0; i < scene->nSolids; i++) {
        uint boxGID = gid * scene->nSolids + i;
        scene->solidCandidates[boxGID].solidID = i + 1;

        GemsBoxIntersection gemsIntersection = _getSolidCandidate(ray, scene->solids[i].bbox_min, scene->solids[i].bbox_max);
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
    bool isTooClose;
    float3 position;
};

typedef struct HitPoint HitPoint;

HitPoint _getTriangleIntersection(Ray ray, float3 v1, float3 v2, float3 v3) {
    HitPoint hitPoint;
    hitPoint.exists = false;
    hitPoint.isTooClose = false;

    float3 edgeA = v2 - v1;
    float3 edgeB = v3 - v1;
    float3 pVector = cross(ray.direction, edgeB);
    float det = dot(edgeA, pVector);

    bool rayIsParallel = fabs(det) < EPSILON;
    if (rayIsParallel) {
        return hitPoint;
    }

    float invDet = 1.0f / det;
    float3 tVector = ray.origin - v1;
    float u = dot(tVector, pVector) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return hitPoint;
    }

    float3 qVector = cross(tVector, edgeA);
    float v = dot(ray.direction, qVector) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return hitPoint;
    }

    float t = dot(edgeB, qVector) * invDet;

    if (t < 0.0f) {
        return hitPoint;
    }

    if (t > (ray.length + 10 * EPSILON)) {
        // No Intersection, it's too far away
        return hitPoint;
    } else if (t > ray.length) {
        // Just a bit too far away. There is no intersection, but we cannot accept photon to land here.
        // hitPoint will also be returned with exists = true in order to consider this event and possibly process it if it was the closest "hit".
        hitPoint.isTooClose = true;
    }

    hitPoint.exists = true;
    hitPoint.position = ray.origin + t * ray.direction;
    return hitPoint;
}

Intersection _findClosestPolygonIntersection(Ray ray, uint solidID,
                                            __global Solid *solids, __global Surface *surfaces,
                                            __global Triangle *triangles, __global Vertex *vertices) {
    Intersection intersection;
    intersection.exists = false;
    intersection.distance = INFINITY;
    for (uint s = solids[solidID-1].firstSurfaceID; s <= solids[solidID-1].lastSurfaceID; s++) {
        for (uint p = surfaces[s].firstPolygonID; p <= surfaces[s].lastPolygonID; p++) {
            uint vertexIDs[3] = {triangles[p].vertexIDs[0], triangles[p].vertexIDs[1], triangles[p].vertexIDs[2]};
            HitPoint hitPoint = _getTriangleIntersection(ray, vertices[vertexIDs[0]].position, vertices[vertexIDs[1]].position, vertices[vertexIDs[2]].position);
            if (!hitPoint.exists) {
                continue;
            }
            float distance = length(hitPoint.position - ray.origin);
            if (distance < intersection.distance) {
                intersection.exists = true;
                intersection.isTooClose = hitPoint.isTooClose;
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

    intersection->normal = weights[0] * vertices[triangles[intersection->polygonID].vertexIDs[0]].normal +
                           weights[1] * vertices[triangles[intersection->polygonID].vertexIDs[1]].normal +
                           weights[2] * vertices[triangles[intersection->polygonID].vertexIDs[2]].normal;
    intersection->normal = normalize(intersection->normal);
}

void _composeIntersection(Intersection *intersection, Ray *ray, Scene *scene) {
    if (!intersection->exists) {
        return;
    }

    if (scene->surfaces[intersection->surfaceID].toSmooth) {
        setSmoothNormal(intersection, scene->triangles, scene->vertices);
    }
    intersection->distanceLeft = ray->length - intersection->distance;
}

Intersection findIntersection(Ray ray, Scene *scene, uint gid) {
    _findBBoxIntersectingSolids(ray, scene, gid);
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
            continue;
        }
        bool contained = scene->solidCandidates[boxGID].distance == 0;
        if (!contained && closestIntersection.exists) {
            break;
        }

        uint solidID = scene->solidCandidates[boxGID].solidID;
        Intersection intersection = _findClosestPolygonIntersection(ray, solidID, scene->solids, scene->surfaces, scene->triangles, scene->vertices);
        if (intersection.exists  && intersection.distance < closestIntersection.distance) {
            closestIntersection = intersection;
        }
    }

    _composeIntersection(&closestIntersection, &ray, scene);
    return closestIntersection;
}

// ----------------- TEST KERNELS -----------------

__kernel void findIntersections(__global Ray *rays, uint nSolids, __global Solid *solids, __global Surface *surfaces,
        __global Triangle *triangles, __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global Intersection *intersections) {
    uint gid = get_global_id(0);
    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates};
    intersections[gid] = findIntersection(rays[gid], &scene, gid);
}
