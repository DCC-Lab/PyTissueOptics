#include "random.c"
#include "vectorOperators.c"


void decreaseWeightBy(__global photonStruct *photons, float delta_weight, uint gid){
    photons[gid].weight -= delta_weight;
}

void interact(__global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger, uint gid, uint logIndex){
    float delta_weight = photons[gid].weight * materials[photons[gid].material_id].albedo;
    decreaseWeightBy(photons, delta_weight, gid);
    logger[logIndex].x = photons[gid].position.x;
    logger[logIndex].y = photons[gid].position.y;
    logger[logIndex].z = photons[gid].position.z;
    logger[logIndex].delta_weight = delta_weight;
}

void moveBy(__global photonStruct *photons, float distance, uint gid){
    photons[gid].position += (distance * photons[gid].direction);
}

float getScatteringDistance(__global float * randomNums, float mu_t, uint gid){
    return -log(randomNums[gid]) / mu_t;
}

float getScatteringAnglePhi(__global float * randomNums, uint gid){
    float phi = 2.0f * M_PI * randomNums[gid];
    return phi;
}

float getScatteringAngleTheta(__global float * randomNums, float g, uint gid){
    if (g == 0){
        return acos(2.0f * randomNums[gid] - 1.0f);
    }
    else{
        float temp = (1.0f - g * g) / (1 - g + 2 * g * randomNums[gid]);
        return acos((1.0f + g * g - temp * temp) / (2 * g));
    }
}

void scatterBy(__global photonStruct *photons, float phi, float theta, uint gid){
    rotateAroundAxisGlobal(&photons[gid].er, &photons[gid].direction, phi);
    rotateAroundAxisGlobal(&photons[gid].direction, &photons[gid].er, theta);
}

void roulette(__global photonStruct *photons, __global uint * randomSeedBuffer, uint gid){
    float randomFloat = getRandomFloatValue(randomSeedBuffer, gid);
    if (randomFloat < 0.1f){
        photons[gid].weight /= 0.1f;
    }
    else{
        photons[gid].weight = 0.0f;
    }
}


bool getIntersection(float distance) {
    return false;
}

// move to scatteringMaterial.c
struct ScatteringAngles {
    float phi, theta;
};

typedef struct ScatteringAngles ScatteringAngles;

ScatteringAngles getScatteringAngles(uint gid,
           __global photonStruct *photons, __constant materialStruct *materials,
           __global float *randomNums, __global uint *seedBuffer)
{
    ScatteringAngles angles;
    randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
    angles.phi = getScatteringAnglePhi(randomNums, gid);
    randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
    float g = materials[photons[gid].material_id].g;
    angles.theta = getScatteringAngleTheta(randomNums, g, gid);
    return angles;
}



void scatter(uint gid, uint logIndex,
           __global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger,
           __global float *randomNums, __global uint *seedBuffer){
    ScatteringAngles angles = getScatteringAngles(gid, photons, materials, randomNums, seedBuffer);

    scatterBy(photons, angles.phi, angles.theta, gid);
    interact(photons, materials, logger, gid, logIndex);
}

float propagateStep(float distance, uint gid, uint logIndex,
           __global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger,
           __global float *randomNums, __global uint *seedBuffer){
    if (distance == 0) {
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float mu_t = materials[photons[gid].material_id].mu_t;
        distance = getScatteringDistance(randomNums, mu_t, gid);
    }

    float distanceLeft = 0;

    // requires intersection struct instead of bool
    bool intersects = getIntersection(distance);

    if (intersects){

    } else {
        moveBy(photons, distance, gid);
        scatter(gid, logIndex, photons, materials, logger, randomNums, seedBuffer);
    }

    return distanceLeft;
}

__kernel void propagate(uint dataSize, float weightThreshold,
                        __global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger,
                        __global float *randomNums, __global uint *seedBuffer){
    uint gid = get_global_id(0);
    uint stepIndex = 0;
    uint logIndex = 0;
    float4 er = getAnyOrthogonalGlobal(&photons[gid].direction);
    photons[gid].er = er;

    float distance = 0;

    while (photons[gid].weight >= weightThreshold){
        logIndex = gid + stepIndex * dataSize;
        distance = propagateStep(distance, gid, logIndex,
                                photons, materials, logger, randomNums, seedBuffer);
        stepIndex++;
    }
}

__kernel void getScatteringDistanceKernel(__global float * distanceBuffer, __global float * randomNums, float mu_t){
    uint gid = get_global_id(0);
    distanceBuffer[gid] = getScatteringDistance(randomNums, mu_t, gid);
}

__kernel void getScatteringAnglePhiKernel(__global float * angleBuffer,  __global float * randomNums){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAnglePhi(randomNums, gid);
}

__kernel void getScatteringAngleThetaKernel(__global float * angleBuffer,  __global float * randomNums, float g){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAngleTheta(randomNums, g, gid);
}
