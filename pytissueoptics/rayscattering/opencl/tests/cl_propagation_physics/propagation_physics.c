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
    photons[gid].position += distance * photons[gid].direction;
}

float getScatteringDistance(__global photonStruct *photons,__constant materialStruct *materials, __global float * randomNums, uint gid){
    return -log(randomNums[gid]) / materials[photons[gid].material_id].mu_t;
}

float getScatteringAnglePhi(__global float * randomNums, uint gid){
    float phi = 2.0f * M_PI * randomNums[gid];
    return phi;
}

float getScatteringAngleTheta(__global photonStruct *photons,__constant materialStruct *materials,  __global float * randomNums, uint gid){
    materialStruct material = materials[photons[gid].material_id];
    if (material.g == 0){
        return acos(2.0f * randomNums[gid] - 1.0f);
    }
    else{
        float temp = (1.0f - material.g * material.g) / (1 - material.g + 2 * material.g * randomNums[gid]);
        return acos((1.0f + material.g * material.g - temp * temp) / (2 * material.g));
    }
}

void scatterBy(__global photonStruct *photons, float phi, float theta, uint gid){
    rotateAroundGlobal(&photons[gid].er, &photons[gid].direction, phi);
    rotateAroundGlobal(&photons[gid].direction, &photons[gid].er, theta);
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

__kernel void propagate(uint dataSize, float weightThreshold, __global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger, __global float *randomNums, __global uint *seedBuffer){
    uint gid = get_global_id(0);
    uint stepIndex = 0;
    uint logIndex = 0;
    while (photons[gid].weight >= weightThreshold){
        logIndex = gid + stepIndex * dataSize;
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float distance = getScatteringDistance(photons, materials, randomNums, gid);
        moveBy(photons, distance, gid);
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float phi = getScatteringAnglePhi(photons, randomNums, gid);
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float theta = getScatteringAngleTheta(photons, materials, randomNums, gid);
        scatterBy(photons, phi, theta, gid);
        interact(photons, materials, logger, gid, logIndex);
        stepIndex++;
        //if (photons[gid].weight <= weightThreshold){
         //   roulette(photons, seedBuffer, gid);
//        }
    }
}
