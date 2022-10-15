struct ScatteringAngles {
    float phi, theta;
};

typedef struct ScatteringAngles ScatteringAngles;

float getScatteringDistance(__global float * randomNumbers, float mu_t, uint gid){
    return -log(randomNumbers[gid]) / mu_t;
}

float getScatteringAnglePhi(__global float * randomNumbers, uint gid){
    float phi = 2.0f * M_PI * randomNumbers[gid];
    return phi;
}

float getScatteringAngleTheta(__global float * randomNumbers, float g, uint gid){
    if (g == 0){
        return acos(2.0f * randomNumbers[gid] - 1.0f);
    }
    else{
        float temp = (1.0f - g * g) / (1 - g + 2 * g * randomNumbers[gid]);
        return acos((1.0f + g * g - temp * temp) / (2 * g));
    }
}

ScatteringAngles getScatteringAngles(uint gid,
           __global Photon *photons, __constant Material *materials,
           __global float *randomNumbers, __global uint *seeds)
{
    ScatteringAngles angles;
    randomNumbers[gid] = getRandomFloatValue(seeds, gid);
    angles.phi = getScatteringAnglePhi(randomNumbers, gid);
    randomNumbers[gid] = getRandomFloatValue(seeds, gid);
    float g = materials[photons[gid].material_id].g;
    angles.theta = getScatteringAngleTheta(randomNumbers, g, gid);
    return angles;
}

// ----------- Test kernels -----------

__kernel void getScatteringDistanceKernel(__global float * distanceBuffer, __global float * randomNumbers, float mu_t){
    uint gid = get_global_id(0);
    distanceBuffer[gid] = getScatteringDistance(randomNumbers, mu_t, gid);
}

__kernel void getScatteringAnglePhiKernel(__global float * angleBuffer,  __global float * randomNumbers){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAnglePhi(randomNumbers, gid);
}

__kernel void getScatteringAngleThetaKernel(__global float * angleBuffer,  __global float * randomNumbers, float g){
    uint gid = get_global_id(0);
    angleBuffer[gid] = getScatteringAngleTheta(randomNumbers, g, gid);
}
