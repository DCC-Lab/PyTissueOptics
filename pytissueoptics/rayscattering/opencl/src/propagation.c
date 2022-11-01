#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"
#include "intersection.c"
#include "fresnel.c"

const int NO_SOLID_ID = -1;
const int NO_SURFACE_ID = -1;

void moveBy(float distance, __global Photon *photons, uint photonId){
    photons[photonId].position += (distance * photons[photonId].direction);
}

void scatterBy(float phi, float theta, __global Photon *photons, uint photonId){
    rotateAroundAxisGlobal(&photons[photonId].er, &photons[photonId].direction, phi);
    rotateAroundAxisGlobal(&photons[photonId].direction, &photons[photonId].er, theta);
}

void decreaseWeightBy(float delta_weight, __global Photon *photons, uint photonId){
    photons[photonId].weight -= delta_weight;
}

void interact(__global Photon *photons, __constant Material *materials, __global DataPoint *logger,
              uint logIndex, uint photonId){
    float delta_weight = photons[photonId].weight * materials[photons[photonId].materialID].albedo;
    decreaseWeightBy(delta_weight, photons, photonId);
    logger[logIndex].x = photons[photonId].position.x;
    logger[logIndex].y = photons[photonId].position.y;
    logger[logIndex].z = photons[photonId].position.z;
    logger[logIndex].delta_weight = delta_weight;
    logger[logIndex].solidID = photons[photonId].solidID;
    logger[logIndex].surfaceID = NO_SURFACE_ID;
}

void scatter(__global Photon *photons, __constant Material *materials, __global uint *seeds, __global DataPoint *logger,
             uint *logIndex, uint gid, uint photonId){

    float rndPhi = getRandomFloatValue(seeds, gid);
    float rndTheta = getRandomFloatValue(seeds, gid);
    ScatteringAngles angles = getScatteringAngles(rndPhi, rndTheta, photons, materials, photonId);

    scatterBy(angles.phi, angles.theta, photons, photonId);
    interact(photons, materials, logger, *logIndex, photonId);
    (*logIndex)++;
}

void roulette(float weightThreshold, __global Photon *photons, __global uint *seeds, uint gid, uint photonId){
    if (photons[photonId].weight >= weightThreshold || photons[photonId].weight == 0){
        return;
    }
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (randomFloat < 0.1){
        photons[photonId].weight /= 0.1;
    }
    else{
        photons[photonId].weight = 0;
    }
}

void reflect(__global Photon *photons, FresnelIntersection *fresnelIntersection, uint photonId){
    rotateAround(&photons[photonId].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void refract(__global Photon *photons, FresnelIntersection *fresnelIntersection, uint photonId){
    rotateAround(&photons[photonId].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void logIntersection(Intersection *intersection, __global Photon *photons, __global Surface *surfaces,
                    __global DataPoint *logger, uint *logIndex, uint photonId){
    uint logID = *logIndex;
    logger[logID].x = photons[photonId].position.x;
    logger[logID].y = photons[photonId].position.y;
    logger[logID].z = photons[photonId].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = surfaces[intersection->surfaceID].insideSolidID;

    bool isLeavingSurface = dot(photons[photonId].direction.xyz, intersection->normal) > 0;
    int sign = isLeavingSurface ? 1 : -1;
    logger[logID].delta_weight = sign * photons[photonId].weight;
    (*logIndex)++;

    int outsideSolidID = surfaces[intersection->surfaceID].outsideSolidID;
    if (outsideSolidID == NO_SOLID_ID){
        return;
    }
    logID++;
    logger[logID].x = photons[photonId].position.x;
    logger[logID].y = photons[photonId].position.y;
    logger[logID].z = photons[photonId].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = outsideSolidID;
    logger[logID].delta_weight = -sign * photons[photonId].weight;
    (*logIndex)++;
}

float reflectOrRefract(__global Photon *photons, __constant Material *materials,
        __global Surface *surfaces, Intersection *intersection, __global DataPoint *logger,
        uint *logIndex, __global uint *seeds, uint gid, uint photonId){
    FresnelIntersection fresnelIntersection = computeFresnelIntersection(photons[photonId].direction.xyz, intersection,
                                                                         materials, surfaces, seeds, gid);

    if (fresnelIntersection.isReflected) {
        reflect(photons, &fresnelIntersection, photonId);
    }
    else {
        logIntersection(intersection, photons, surfaces, logger, logIndex, photonId);
        refract(photons, &fresnelIntersection, photonId);

        float mut1 = materials[photons[photonId].materialID].mu_t;
        float mut2 = materials[fresnelIntersection.nextMaterialID].mu_t;
        if (mut1 == 0) {
            intersection->distanceLeft = 0;
        } else if (mut2 != 0) {
            intersection->distanceLeft *= mut1 / mut2;
        } else {
            intersection->distanceLeft = INFINITY;
        }
        photons[photonId].materialID = fresnelIntersection.nextMaterialID;
        photons[photonId].solidID = fresnelIntersection.nextSolidID;
    }

    return intersection->distanceLeft;
}

float propagateStep(float distance, __global Photon *photons, __constant Material *materials, Scene *scene,
                    __global uint *seeds, __global DataPoint *logger, uint *logIndex, uint gid, uint photonId){

    if (distance == 0) {
        float mu_t = materials[photons[photonId].materialID].mu_t;
        float randomNumber = getRandomFloatValue(seeds, gid);
        distance = getScatteringDistance(mu_t, randomNumber);
    }

    Ray stepRay = {photons[photonId].position, photons[photonId].direction, distance};
    Intersection intersection = findIntersection(stepRay, scene, gid);  // todo: make sure gid (not photonID) is used correctly by solid candidates

    float distanceLeft = 0;

    if (intersection.exists){
        moveBy(intersection.distance, photons, photonId);
        distanceLeft = reflectOrRefract(photons, materials, scene->surfaces, &intersection, logger, logIndex, seeds, gid, photonId);
        moveBy(0.00001f, photons, photonId);  // move a little bit to help avoid bad intersection check
    } else {
        if (distance == INFINITY){
            photons[photonId].weight = 0;
            return 0;
        }
        moveBy(distance, photons, photonId);
        scatter(photons, materials, seeds, logger, logIndex, gid, photonId);
    }

    return distanceLeft;
}

__kernel void propagate(uint maxPhotons, uint maxInteractions, float weightThreshold, uint workUnitsAmount, __global Photon *photons,
            __constant Material *materials, uint nSolids, __global Solid *solids, __global Surface *surfaces, __global Triangle *triangles,
            __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global uint *seeds, __global DataPoint *logger){
    // todo: rename photonId to photonID, gid to globalID
    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates};

    uint gid = get_global_id(0);
    uint logIndex = gid * maxInteractions;
    uint maxLogIndex = logIndex + maxInteractions;

    uint photonCount = 0;

    while ((logIndex < maxLogIndex -1) && (photonCount < maxPhotons)){  // todo: not sure the first condition is needed (duplicate)
        uint currentPhotonIndex = gid + (photonCount * workUnitsAmount);  // todo: I would prefer to use subsequent IDs (convention)

        float distance = 0;
        float4 er = getAnyOrthogonalGlobal(&photons[currentPhotonIndex].direction);  // todo: refactor everything to float3
        photons[currentPhotonIndex].er = er;

        while (photons[currentPhotonIndex].weight != 0){
            if (logIndex >= (maxLogIndex -1)){  // Added -1 to avoid potential overflow when intersection logs twice
                return;
            }
            distance = propagateStep(distance, photons, materials, &scene,
                                     seeds, logger, &logIndex, gid, currentPhotonIndex);
            roulette(weightThreshold, photons, seeds, gid, currentPhotonIndex);
            }
        photonCount++;
    }
}
