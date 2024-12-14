#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"
#include "intersection.c"
#include "fresnel.c"

__constant int NO_SOLID_ID = -1;
__constant int NO_SURFACE_ID = -1;
__constant float MIN_ANGLE = 0.0001f;

void moveBy(float distance, __global Photon *photons, uint photonID){
    photons[photonID].position += (distance * photons[photonID].direction);
}

void moveTo(float3 position, __global Photon *photons, uint photonID){
    photons[photonID].position = position;
}

void scatterBy(float phi, float theta, __global Photon *photons, uint photonID){
    rotateAroundAxisGlobal(&photons[photonID].er, &photons[photonID].direction, phi);
    rotateAroundAxisGlobal(&photons[photonID].direction, &photons[photonID].er, theta);
    photons[photonID].er = getAnyOrthogonalGlobal(&photons[photonID].direction);
}

void decreaseWeightBy(float delta_weight, __global Photon *photons, uint photonID){
    photons[photonID].weight -= delta_weight;
}

void interact(__global Photon *photons, __constant Material *materials, __global DataPoint *logger,
              uint logIndex, uint photonID){
    float delta_weight = photons[photonID].weight * materials[photons[photonID].materialID].albedo;
    decreaseWeightBy(delta_weight, photons, photonID);
    logger[logIndex].x = photons[photonID].position.x;
    logger[logIndex].y = photons[photonID].position.y;
    logger[logIndex].z = photons[photonID].position.z;
    logger[logIndex].delta_weight = delta_weight;
    logger[logIndex].solidID = photons[photonID].solidID;
    logger[logIndex].surfaceID = NO_SURFACE_ID;
}

void scatter(__global Photon *photons, __constant Material *materials, __global uint *seeds, __global DataPoint *logger,
             uint *logIndex, uint gid, uint photonID){

    float rndPhi = getRandomFloatValue(seeds, gid);
    float rndTheta = getRandomFloatValue(seeds, gid);
    ScatteringAngles angles = getScatteringAngles(rndPhi, rndTheta, photons, materials, photonID);

    scatterBy(angles.phi, angles.theta, photons, photonID);
    interact(photons, materials, logger, *logIndex, photonID);
    (*logIndex)++;
}

void roulette(float weightThreshold, __global Photon *photons, __global uint *seeds, uint gid, uint photonID){
    if (photons[photonID].weight >= weightThreshold || photons[photonID].weight == 0){
        return;
    }
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (randomFloat < 0.1){
        photons[photonID].weight /= 0.1;
    }
    else{
        photons[photonID].weight = 0;
    }
}

void reflect(FresnelIntersection *fresnelIntersection, __global Photon *photons, uint photonID){
    rotateAround(&photons[photonID].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void refract(FresnelIntersection *fresnelIntersection, __global Photon *photons, uint photonID){
    rotateAround(&photons[photonID].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void logIntersection(Intersection *intersection, __global Photon *photons, __global Surface *surfaces,
                    __global DataPoint *logger, uint *logIndex, uint photonID){
    uint logID = *logIndex;
    logger[logID].x = photons[photonID].position.x;
    logger[logID].y = photons[photonID].position.y;
    logger[logID].z = photons[photonID].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = surfaces[intersection->surfaceID].insideSolidID;

    bool isLeavingSurface = dot(photons[photonID].direction, intersection->normal) > 0;
    int sign = isLeavingSurface ? 1 : -1;
    logger[logID].delta_weight = sign * photons[photonID].weight;
    (*logIndex)++;

    int outsideSolidID = surfaces[intersection->surfaceID].outsideSolidID;
    if (outsideSolidID == NO_SOLID_ID){
        return;
    }
    logID++;
    logger[logID].x = photons[photonID].position.x;
    logger[logID].y = photons[photonID].position.y;
    logger[logID].z = photons[photonID].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = outsideSolidID;
    logger[logID].delta_weight = -sign * photons[photonID].weight;
    (*logIndex)++;
}

float reflectOrRefract(Intersection *intersection, __global Photon *photons, __constant Material *materials,
        __global Surface *surfaces, __global DataPoint *logger, uint *logIndex, __global uint *seeds, uint gid, uint photonID){
    FresnelIntersection fresnelIntersection = computeFresnelIntersection(photons[photonID].direction, intersection,
                                                                         materials, surfaces, seeds, gid);

    if (fresnelIntersection.isReflected) {
        if (intersection->isSmooth) {
            // Prevent reflection from crossing the raw surface.
            float smoothAngle = acos(dot(intersection->normal, intersection->rawNormal));
            float minDeflectionAngle = smoothAngle + fabs(fresnelIntersection.angleDeflection) / 2 + MIN_ANGLE;
            if (fabs(fresnelIntersection.angleDeflection) < minDeflectionAngle) {
                fresnelIntersection.angleDeflection = sign(fresnelIntersection.angleDeflection) * minDeflectionAngle;
            }
        }
        reflect(&fresnelIntersection, photons, photonID);
    }
    else {
        logIntersection(intersection, photons, surfaces, logger, logIndex, photonID);
        if (intersection->isSmooth) {
            // Prevent refraction from not crossing the raw surface.
            float maxDeflectionAngle = fabs(M_PI_F / 2 - acos(dot(intersection->rawNormal, photons[photonID].direction))) - MIN_ANGLE;
            if (fabs(fresnelIntersection.angleDeflection) > maxDeflectionAngle) {
                fresnelIntersection.angleDeflection = sign(fresnelIntersection.angleDeflection) * maxDeflectionAngle;
            }
        }
        refract(&fresnelIntersection, photons, photonID);

        float mut1 = materials[photons[photonID].materialID].mu_t;
        float mut2 = materials[fresnelIntersection.nextMaterialID].mu_t;
        if (mut1 == 0) {
            intersection->distanceLeft = 0;
        } else if (mut2 != 0) {
            intersection->distanceLeft *= mut1 / mut2;
        } else {
            intersection->distanceLeft = INFINITY;
        }
        photons[photonID].materialID = fresnelIntersection.nextMaterialID;
        photons[photonID].solidID = fresnelIntersection.nextSolidID;
    }

    return intersection->distanceLeft;
}

float propagateStep(float distance, __global Photon *photons, __constant Material *materials, Scene *scene,
                    __global uint *seeds, __global DataPoint *logger, uint *logIndex, uint gid, uint photonID){

    if (distance <= 0) {
        float mu_t = materials[photons[photonID].materialID].mu_t;
        float randomNumber = getRandomFloatValue(seeds, gid);
        distance += getScatteringDistance(mu_t, randomNumber);
        if (distance < 0){
            // Not really possible until mu_t is very high (> 1000) and intense smoothing is applied (order-1 spheres).
            distance = 0;
        }
    }

    Ray stepRay = {photons[photonID].position, photons[photonID].direction, distance};
    Intersection intersection = findIntersection(stepRay, scene, gid, photons[photonID].solidID);

    float distanceLeft = 0;

    if (intersection.exists){
        moveTo(intersection.position, photons, photonID);
        distanceLeft = reflectOrRefract(&intersection, photons, materials, scene->surfaces, logger, logIndex, seeds, gid, photonID);
    } else {
        if (distance == INFINITY){
            photons[photonID].weight = 0;
            return 0;
        }

        moveBy(distance, photons, photonID);

        scatter(photons, materials, seeds, logger, logIndex, gid, photonID);
    }

    return distanceLeft;
}

__kernel void propagate(uint maxPhotons, uint maxInteractions, float weightThreshold, uint workUnitsAmount, __global Photon *photons,
            __constant Material *materials, uint nSolids, __global Solid *solids, __global Surface *surfaces, __global Triangle *triangles,
            __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global uint *seeds, __global DataPoint *logger){
    /*
    OpenCL implementation of the Python module Photon.
    See the Python module documentation for more details.
    */

    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates};

    uint gid = get_global_id(0);
    uint logIndex = gid * maxInteractions;
    uint maxLogIndex = logIndex + maxInteractions;

    uint photonCount = 0;

    while (photonCount < maxPhotons){
        uint currentPhotonIndex = gid + (photonCount * workUnitsAmount);
        photons[currentPhotonIndex].er = getAnyOrthogonalGlobal(&photons[currentPhotonIndex].direction);

        float distance = 0;
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


// ---------------------------- TEST KERNELS ----------------------------


__kernel void moveByKernel(float distance, __global Photon *photons, uint photonID){
    moveBy(distance, photons, photonID);
}

__kernel void scatterByKernel(float phi, float theta, __global Photon *photons, uint photonID){
    photons[photonID].er = getAnyOrthogonalGlobal(&photons[photonID].direction);
    scatterBy(phi, theta, photons, photonID);
}

__kernel void decreaseWeightByKernel(float delta_weight, __global Photon *photons, uint photonID){
    decreaseWeightBy(delta_weight, photons, photonID);
}

__kernel void rouletteKernel(float weightThreshold, __global uint *seeds, __global Photon *photons, uint photonID){
    roulette(weightThreshold, photons, seeds, photonID, photonID);
}

__kernel void reflectKernel(float3 incidencePlane, float angleDeflection, __global Photon *photons, uint photonID){
    FresnelIntersection fresnelIntersection;
    fresnelIntersection.incidencePlane = incidencePlane;
    fresnelIntersection.angleDeflection = angleDeflection;
    reflect(&fresnelIntersection, photons, photonID);
}

__kernel void refractKernel(float3 incidencePlane, float angleDeflection, __global Photon *photons, uint photonID){
    FresnelIntersection fresnelIntersection;
    fresnelIntersection.incidencePlane = incidencePlane;
    fresnelIntersection.angleDeflection = angleDeflection;
    refract(&fresnelIntersection, photons, photonID);
}

__kernel void interactKernel(__constant Material *materials, __global DataPoint *logger,
                             uint logIndex, __global Photon *photons, uint photonID){
    interact(photons, materials, logger, logIndex, photonID);
}

__kernel void logIntersectionKernel(float3 normal, int surfaceID, __global Surface *surfaces,
                    __global DataPoint *logger, uint logIndex, __global Photon *photons, uint photonID){
    Intersection intersection;
    intersection.normal = normal;
    intersection.surfaceID = surfaceID;
    logIntersection(&intersection, photons, surfaces, logger, &logIndex, photonID);
}

__kernel void reflectOrRefractKernel(float3 normal, int surfaceID, float distanceLeft,
                                     __constant Material *materials, __global Surface *surfaces,
                                     __global DataPoint *logger, uint logIndex, __global uint *seeds,
                                     __global Photon *photons, uint photonID){
    Intersection intersection;
    intersection.normal = normal;
    intersection.surfaceID = surfaceID;
    intersection.distanceLeft = distanceLeft;
    reflectOrRefract(&intersection, photons, materials, surfaces, logger, &logIndex, seeds, photonID, photonID);
}

__kernel void propagateStepKernel(float distance, __constant Material *materials, __global Surface *surfaces,
                    __global uint *seeds, __global DataPoint *logger, uint logIndex,
                    __global Photon *photons, uint photonID){
    Scene scene;
    scene.surfaces = surfaces;
    uint gid = photonID;
    propagateStep(distance, photons, materials, &scene, seeds, logger, &logIndex, gid, photonID);
}
