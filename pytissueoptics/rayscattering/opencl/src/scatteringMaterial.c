struct ScatteringAngles {
    float phi, theta;
};

typedef struct ScatteringAngles ScatteringAngles;

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

// ----------- Test kernels -----------

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
