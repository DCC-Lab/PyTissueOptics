
struct FresnelIntersection {
    float3 incidencePlane;
    uint isReflected;
    float angleDeflection;
    uint nextMaterialID;
};

typedef struct FresnelIntersection FresnelIntersection;


void _createFresnelIntersection(FresnelIntersection* fresnelIntersection, float thetaIn) {
    // todo: implement refraction
    fresnelIntersection->isReflected = true;
    fresnelIntersection->angleDeflection = 2 * thetaIn - M_PI_F;
}


FresnelIntersection computeFresnelIntersection(float3 rayDirection, Intersection *intersection,
        __constant Material *materials, __global Surface *surfaces) {
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
    } else {
        nIn = materials[surfaces[intersection->surfaceID].insideMaterialID].n;
        nOut = materials[surfaces[intersection->surfaceID].outsideMaterialID].n;
        fresnelIntersection.nextMaterialID = surfaces[intersection->surfaceID].outsideMaterialID;
    }

    fresnelIntersection.incidencePlane = cross(rayDirection, normal);
    if (length(fresnelIntersection.incidencePlane) < 0.0000001f) {
        fresnelIntersection.incidencePlane = getAnyOrthogonal(&rayDirection);
    }
    fresnelIntersection.incidencePlane = normalize(fresnelIntersection.incidencePlane);

    float thetaIn = acos(dot(normal, rayDirection));

    _createFresnelIntersection(&fresnelIntersection, thetaIn);

    return fresnelIntersection;
}