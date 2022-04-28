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

__kernel void getScatteringDistanceKernel(__global float * distanceBuffer, __global float * randomNums, float mu_t){
    uint gid = get_global_id(0);
    distanceBuffer[gid] = getScatteringDistance(randomNums, mu_t, gid);
}

// GET ANGLES
float getScatteringAnglePhi(__global float * randomNums, uint gid){
    float phi = 2.0f * M_PI * randomNums[gid];
    return phi;
}

__kernel void getScatteringAnglePhiKernel(__global float * angleBuffer,  __global float * randomNums){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAnglePhi(randomNums, gid);
}

float getScatteringAngleTheta(float g,  __global float * randomNums, uint gid){
    if (g == 0){
        return acos(2.0f * randomNums[gid] - 1.0f);
    }
    else{
        float temp = (1.0f - g * g) / (1 - g + 2 * g * randomNums[gid]);
        return acos((1.0f + g * g - temp * temp) / (2 * g));
    }
}

__kernel void getScatteringAngleThetaKernel(__global float * angleBuffer,  __global float * randomNums, float g){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAngleTheta(g, randomNums, gid);
}


// GET DIRECTIONS
void scatterBy(__global photonStruct *photons, float phi, float theta, uint gid){
    rotateAroundGlobal(&photons[gid].er, &photons[gid].direction, phi);
    rotateAroundGlobal(&photons[gid].direction, &photons[gid].er, theta);
}

void roulette(__global photonStruct *photons, __global float  * randomNums, uint gid){
    if (randomNums[gid] < 0.1f){
        photons[gid].weight /= 0.1f;
    }
    else{
        photons[gid].weight = 0.0f;
    }
}
