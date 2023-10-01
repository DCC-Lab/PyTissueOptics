
struct FresnelIntersection {
    float3 incidencePlane;
    uint isReflected;
    float angleDeflection;
    uint nextMaterialID;
    int nextSolidID;
};

typedef struct FresnelIntersection FresnelIntersection;


float _getReflectionCoefficient(float n1, float n2, float thetaIn) {
    if (n1 == n2) {
        return 0;
    }
    if (thetaIn == 0) {
        float R = (n2 - n1) / (n2 + n1);
        return R * R;
    }

    float sa1 = sin(thetaIn);

    float sa2 = sa1 * n1 / n2;
    if (sa2 > 1) {
        return 1;
    }

    float ca1 = sqrt(1 - sa1 * sa1);
    float ca2 = sqrt(1 - sa2 * sa2);

    float cap = ca1 * ca2 - sa1 * sa2;
    float cam = ca1 * ca2 + sa1 * sa2;
    float sap = sa1 * ca2 + ca1 * sa2;
    float sam = sa1 * ca2 - ca1 * sa2;

    return 0.5 * sam * sam * (cap * cap + cam * cam) / (sap * sap * cam * cam);
}

bool _getIsReflected(float nIn, float nOut, float thetaIn, __global uint *seeds, uint gid) {
    float R = _getReflectionCoefficient(nIn, nOut, thetaIn);
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (R > randomFloat) {
        return true;
    }
    return false;
}

float _getReflectionDeflection(float thetaIn) {
    return 2 * thetaIn - M_PI_F;
}

float _getRefractionDeflection(float nIn, float nOut, float thetaIn) {
    float sinThetaOut = nIn / nOut * sin(thetaIn);
    float thetaOut = asin(sinThetaOut);
    return thetaIn - thetaOut;
}

void _createFresnelIntersection(FresnelIntersection* fresnelIntersection,
                                float nIn, float nOut, float thetaIn, __global uint *seeds, uint gid) {
    fresnelIntersection->isReflected = _getIsReflected(nIn, nOut, thetaIn, seeds, gid);

    if (fresnelIntersection->isReflected) {
        fresnelIntersection->angleDeflection = _getReflectionDeflection(thetaIn);
    } else {
        fresnelIntersection->angleDeflection = _getRefractionDeflection(nIn, nOut, thetaIn);
    }
}

FresnelIntersection computeFresnelIntersection(float3 rayDirection, Intersection *intersection,
        __constant Material *materials, __global Surface *surfaces, __global uint *seeds, uint gid) {
    FresnelIntersection fresnelIntersection;
    float3 normal = intersection->normal;

    float nIn;
    float nOut;

    bool goingInside = dot(rayDirection, normal) < 0;
    if (goingInside) {
        normal *= -1;
        nIn = materials[surfaces[intersection->surfaceID].outsideMaterialID].n;
        nOut = materials[surfaces[intersection->surfaceID].insideMaterialID].n;
        fresnelIntersection.nextMaterialID = surfaces[intersection->surfaceID].insideMaterialID;
        fresnelIntersection.nextSolidID = surfaces[intersection->surfaceID].insideSolidID;
    } else {
        nIn = materials[surfaces[intersection->surfaceID].insideMaterialID].n;
        nOut = materials[surfaces[intersection->surfaceID].outsideMaterialID].n;
        fresnelIntersection.nextMaterialID = surfaces[intersection->surfaceID].outsideMaterialID;
        fresnelIntersection.nextSolidID = surfaces[intersection->surfaceID].outsideSolidID;
    }

    fresnelIntersection.incidencePlane = cross(rayDirection, normal);
    if (length(fresnelIntersection.incidencePlane) < 0.0000001f) {
        fresnelIntersection.incidencePlane = getAnyOrthogonal(&rayDirection);
    }
    fresnelIntersection.incidencePlane = normalize(fresnelIntersection.incidencePlane);

    float thetaIn = acos(clamp(dot(normal, rayDirection), -1.0f, 1.0f));

    _createFresnelIntersection(&fresnelIntersection, nIn, nOut, thetaIn, seeds, gid);

    return fresnelIntersection;
}

// --------------- TEST KERNEL ---------------

Intersection getLocalIntersection(__global Intersection *intersections, uint gid) {
    Intersection intersection;
    intersection.exists = intersections[gid].exists;
    intersection.isTooClose = intersections[gid].isTooClose;
    intersection.distance = intersections[gid].distance;
    intersection.position = intersections[gid].position;
    intersection.normal = intersections[gid].normal;
    intersection.surfaceID = intersections[gid].surfaceID;
    intersection.polygonID = intersections[gid].polygonID;
    intersection.distanceLeft = intersections[gid].distanceLeft;
    return intersection;
}

__kernel void computeFresnelIntersectionKernel(float3 rayDirection, __global Intersection *intersections,
        __constant Material *materials, __global Surface *surfaces, __global uint *seeds,
        __global FresnelIntersection *fresnelIntersections) {
    uint gid = get_global_id(0);
    Intersection localIntersection = getLocalIntersection(intersections, gid);
    fresnelIntersections[gid] = computeFresnelIntersection(rayDirection, &localIntersection, materials, surfaces, seeds, gid);
}

struct FloatContainer {
    float value;
};
typedef struct FloatContainer FloatContainer;

__kernel void getReflectionCoefficientKernel(float n1, float n2, float thetaIn, __global FloatContainer *results) {
    uint gid = get_global_id(0);
    results[gid].value = _getReflectionCoefficient(n1, n2, thetaIn);
}
