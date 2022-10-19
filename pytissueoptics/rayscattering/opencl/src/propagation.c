#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"
#include "intersection.c"
#include "fresnel.c"


void moveBy(__global Photon *photons, float distance, uint gid){
    photons[gid].position += (distance * photons[gid].direction);
}

void scatterBy(__global Photon *photons, float phi, float theta, uint gid){
    rotateAroundAxisGlobal(&photons[gid].er, &photons[gid].direction, phi);
    rotateAroundAxisGlobal(&photons[gid].direction, &photons[gid].er, theta);
}

void decreaseWeightBy(__global Photon *photons, float delta_weight, uint gid){
    photons[gid].weight -= delta_weight;
}

void interact(__global Photon *photons, __constant Material *materials, __global DataPoint *logger, uint gid, uint logIndex){
    float delta_weight = photons[gid].weight * materials[photons[gid].material_id].albedo;
    decreaseWeightBy(photons, delta_weight, gid);
    logger[logIndex].x = photons[gid].position.x;
    logger[logIndex].y = photons[gid].position.y;
    logger[logIndex].z = photons[gid].position.z;
    logger[logIndex].delta_weight = delta_weight;
}

void scatter(uint gid, uint logIndex,
           __global Photon *photons, __constant Material *materials, __global DataPoint *logger,
           __global float *randomNumbers, __global uint *seeds){
    ScatteringAngles angles = getScatteringAngles(gid, photons, materials, randomNumbers, seeds);

    scatterBy(photons, angles.phi, angles.theta, gid);
    interact(photons, materials, logger, gid, logIndex);
}

void roulette(uint gid, float weightThreshold, __global Photon *photons, __global uint * randomSeedBuffer){
    if (photons[gid].weight >= weightThreshold){
        return;
    }
    float randomFloat = getRandomFloatValue(randomSeedBuffer, gid);
    if (randomFloat < 0.1){
        photons[gid].weight /= 0.1;
    }
    else{
        photons[gid].weight = 0;
    }
}

void reflect(__global Photon *photons, FresnelIntersection *fresnelIntersection, uint gid){
    rotateAround(&photons[gid].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void refract(__global Photon *photons, FresnelIntersection *fresnelIntersection, uint gid){
    rotateAround(&photons[gid].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

float reflectOrRefract(__global Photon *photons, __constant Material *materials,
        __global Surface *surfaces, Intersection *intersection, uint gid){
    FresnelIntersection fresnelIntersection = computeFresnelIntersection(photons[gid].direction.xyz, intersection,
                                                                         materials, surfaces);

    if (fresnelIntersection.isReflected) {
        reflect(photons, &fresnelIntersection, gid);
    }
    else {
        // todo: logIntersection()
        refract(photons, &fresnelIntersection, gid);

        float mut1 = materials[photons[gid].material_id].mu_t;
        float mut2 = materials[fresnelIntersection.nextMaterialID].mu_t;
        if (mut1 == 0) {
            intersection->distanceLeft = 0;
        } else if (mut2 != 0) {
            intersection->distanceLeft += mut1 / mut2;
        } else {
            intersection->distanceLeft = INFINITY;
        }
        photons[gid].material_id = fresnelIntersection.nextMaterialID;
    }

    return intersection->distanceLeft;
}

float propagateStep(float distance, uint gid, uint logIndex,
           __global Photon *photons, __constant Material *materials, __global DataPoint *logger,
           __global float *randomNumbers, __global uint *seeds,
           uint nSolids, __global Solid *solids, __global Surface *surfaces, __global Triangle *triangles,
            __global Vertex *vertices, __global SolidCandidate *solidCandidates){

    if (distance == 0) {
        randomNumbers[gid] = getRandomFloatValue(seeds, gid);
        float mu_t = materials[photons[gid].material_id].mu_t;
        distance = getScatteringDistance(randomNumbers, mu_t, gid);
    }

    Ray stepRay = {photons[gid].position, photons[gid].direction, distance};
    Intersection intersection = findIntersection(stepRay, nSolids, solids, surfaces, triangles, vertices,
                                        solidCandidates, gid);

    float distanceLeft = 0;

    if (intersection.exists){
        moveBy(photons, intersection.distance, gid);
        distanceLeft = reflectOrRefract(photons, materials, surfaces, &intersection, gid);
        moveBy(photons, 0.00001f, gid);  // move a little bit to help avoid bad intersection check
    } else {
        moveBy(photons, distance, gid);
        scatter(gid, logIndex, photons, materials, logger, randomNumbers, seeds);
    }

    return distanceLeft;
}

__kernel void propagate(uint dataSize, uint maxInteractions, float weightThreshold,
            __global Photon *photons, __constant Material *materials, __global DataPoint *logger,
            __global float *randomNumbers, __global uint *seeds,
            uint nSolids, __global Solid *solids, __global Surface *surfaces, __global Triangle *triangles,
            __global Vertex *vertices, __global SolidCandidate *solidCandidates){
    // todo: maybe simplify args with SceneStruct with ptrs to ptrs?
    uint gid = get_global_id(0);
    uint stepIndex = 0;
    uint logIndex = 0;
    float4 er = getAnyOrthogonalGlobal(&photons[gid].direction);
    photons[gid].er = er;

    float distance = 0;

    while (photons[gid].weight != 0){
        if (stepIndex == maxInteractions){
            printf("Warning: Out of logger memory for photon %d who could not propagate totally.\n", gid);
            break;
        }

        logIndex = gid + stepIndex * dataSize;
        distance = propagateStep(distance, gid, logIndex,
                                photons, materials, logger, randomNumbers, seeds,
                                nSolids, solids, surfaces, triangles, vertices, solidCandidates);
        roulette(gid, weightThreshold, photons, seeds);
        stepIndex++;
    }
}
