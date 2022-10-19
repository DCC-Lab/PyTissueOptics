#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"


void moveBy(float distance, __global Photon *photons, uint gid){
    photons[gid].position += (distance * photons[gid].direction);
}

void scatterBy(float phi, float theta, __global Photon *photons, uint gid){
    rotateAroundAxisGlobal(&photons[gid].er, &photons[gid].direction, phi);
    rotateAroundAxisGlobal(&photons[gid].direction, &photons[gid].er, theta);
}

void decreaseWeightBy(float delta_weight, __global Photon *photons,  uint gid){
    photons[gid].weight -= delta_weight;
}

void interact(__global Photon *photons, __constant Material *materials, __global DataPoint *logger,
              uint logIndex, uint gid){
    float delta_weight = photons[gid].weight * materials[photons[gid].material_id].albedo;
    decreaseWeightBy(delta_weight, photons, gid);
    logger[logIndex].x = photons[gid].position.x;
    logger[logIndex].y = photons[gid].position.y;
    logger[logIndex].z = photons[gid].position.z;
    logger[logIndex].delta_weight = delta_weight;
}

void scatter(__global Photon *photons, __constant Material *materials, __global uint *seeds, __global DataPoint *logger,
             uint logIndex, uint gid){

    float rndPhi = getRandomFloatValue(seeds, gid);
    float rndTheta = getRandomFloatValue(seeds, gid);
    ScatteringAngles angles = getScatteringAngles(rndPhi, rndTheta, photons, materials, gid);
    scatterBy(angles.phi, angles.theta, photons, gid);
    interact(photons, materials, logger, logIndex, gid);
}

void roulette(float weightThreshold, __global Photon *photons, __global uint *seeds, uint gid){
    if (photons[gid].weight >= weightThreshold){
        return;
    }
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (randomFloat < 0.1){
        photons[gid].weight /= 0.1;
    }
    else{
        photons[gid].weight = 0;
    }
}

bool getIntersection(float distance) {
    return false;
}

float propagateStep(float distance, __global Photon *photons, __constant Material *materials, __global uint *seeds,
                    __global DataPoint *logger, uint logIndex, uint gid){

    if (distance == 0) {
        float mu_t = materials[photons[gid].material_id].mu_t;
        float randomNumber = getRandomFloatValue(seeds, gid);
        distance = getScatteringDistance(mu_t, randomNumber);
    }

    float distanceLeft = 0;

    // requires intersection struct instead of bool
    bool intersects = getIntersection(distance);

    if (intersects){

    } else {
        moveBy(distance, photons, gid);
        scatter(photons, materials, seeds, logger, logIndex, gid);
    }

    return distanceLeft;
}

__kernel void propagate(uint maxPhotons, uint maxInteractions, float weightThreshold, uint workUnitsAmount, __global Photon *photons,
                        __constant Material *materials, __global uint *seeds, __global DataPoint *logger){
    uint gid = get_global_id(0);
    uint photonCount = 0;
    uint interactionCount = 0;
    float4 er = getAnyOrthogonalGlobal(&photons[gid].direction);
    photons[gid].er = er;


    while ((interactionCount < maxInteractions) && (photonCount < maxPhotons)){
        Photon currentPhoton = photons[gid + (photonCount * workUnitsAmount)];
        float distance = 0;
        while (currentPhoton.weight != 0){
            if (interactionCount == maxInteractions){
                return;}

            uint logIndex = gid + (interactionCount * workUnitsAmount);
            distance = propagateStep(distance, photons, materials, seeds, logger, logIndex, gid);
            roulette(weightThreshold, photons, seeds, gid);
            interactionCount++;
            }
        photonCount++;

    }
}
