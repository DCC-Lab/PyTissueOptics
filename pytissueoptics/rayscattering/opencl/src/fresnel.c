
struct FresnelIntersection {
    float3 incidencePlane;
    uint isReflected;
    float angleDeflection;
};

typedef struct FresnelIntersection FresnelIntersection;

FresnelIntersection computeFresnelIntersection(float3 rayDirection, Intersection *intersection,
        __constant Material *materials, __global Surface *surfaces) {
    FresnelIntersection fresnelIntersection;
    uint inMaterialID = surfaces[intersection->surfaceID].insideMaterialID;
    uint outMaterialID = surfaces[intersection->surfaceID].outsideMaterialID;

    // If each polygon as notion of Surface ID
    // and each surface has a single inside and outside material...
    // then we could store environments only inside surfaces
    // lets do this. This will save a lot of space when dealing with many polygons
    return fresnelIntersection;
}