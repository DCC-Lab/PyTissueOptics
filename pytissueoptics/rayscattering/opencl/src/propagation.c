#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"


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
    float delta_weight = photons[photonId].weight * materials[photons[photonId].material_id].albedo;
    decreaseWeightBy(delta_weight, photons, photonId);
    logger[logIndex].x = photons[photonId].position.x;
    logger[logIndex].y = photons[photonId].position.y;
    logger[logIndex].z = photons[photonId].position.z;
    logger[logIndex].delta_weight = delta_weight;
}

void scatter(__global Photon *photons, __constant Material *materials, __global uint *seeds, __global DataPoint *logger,
             uint logIndex, uint gid, uint photonId){

    float rndPhi = getRandomFloatValue(seeds, gid);
    float rndTheta = getRandomFloatValue(seeds, gid);
    ScatteringAngles angles = getScatteringAngles(rndPhi, rndTheta, photons, materials, photonId);
    scatterBy(angles.phi, angles.theta, photons, photonId);
    interact(photons, materials, logger, logIndex, photonId);
}

void roulette(float weightThreshold, __global Photon *photons, __global uint *seeds, uint gid, uint photonId){
    if (photons[photonId].weight >= weightThreshold){
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

bool getIntersection(float distance) {
    return false;
}

float propagateStep(float distance, __global Photon *photons, __constant Material *materials, __global uint *seeds,
                    __global DataPoint *logger, uint logIndex, uint gid, uint photonId){

    if (distance == 0) {
        float mu_t = materials[photons[photonId].material_id].mu_t;
        float randomNumber = getRandomFloatValue(seeds, gid);
        distance = getScatteringDistance(mu_t, randomNumber);
    }

    float distanceLeft = 0;

    // requires intersection struct instead of bool
    bool intersects = getIntersection(distance);

    if (intersects){

    } else {
        moveBy(distance, photons, photonId);
        scatter(photons, materials, seeds, logger, logIndex, gid, photonId);
    }

    return distanceLeft;
}

__kernel void propagate(uint maxPhotons, uint maxInteractions, float weightThreshold, uint workUnitsAmount, __global Photon *photons,
                        __constant Material *materials, __global uint *seeds, __global DataPoint *logger){
    uint gid = get_global_id(0);
    uint photonCount = 0;
    uint interactionCount = 0;


    while ((interactionCount < maxInteractions) && (photonCount < maxPhotons)){
        uint currentPhotonIndex = gid + (photonCount * workUnitsAmount);
        //printf("gid: %d, pid: %d, photonCount: %d / %d \n", gid, currentPhotonIndex, photonCount, maxPhotons);
        //printf("gid: %d, pid: %d, interactionCount: %d / %d\n\n", gid, currentPhotonIndex, photonCount, maxInteractions);

        float distance = 0;
        float4 er = getAnyOrthogonalGlobal(&photons[currentPhotonIndex].direction);
        photons[currentPhotonIndex].er = er;

        while (photons[currentPhotonIndex].weight != 0){
            if (interactionCount == maxInteractions){
            //printf("Max interactions reached");
                return;}

            uint logIndex = (gid * workUnitsAmount) + interactionCount;
            distance = propagateStep(distance, photons, materials, seeds, logger, logIndex, gid, currentPhotonIndex);
            roulette(weightThreshold, photons, seeds, gid, currentPhotonIndex);
            interactionCount++;
            //printf("photon %d: %f\n",currentPhotonIndex, photons[currentPhotonIndex].weight);
            }
        photonCount++;

    }
    //printf("End of loop. Filled interactions or propagated enough photons");
}
