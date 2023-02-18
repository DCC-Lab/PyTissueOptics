struct ScatteringAngles {
    float phi, theta;
};

typedef struct ScatteringAngles ScatteringAngles;

float getScatteringDistance(float mu_t, float randomNumber){
    return -log(randomNumber) / mu_t;
}

float getScatteringAnglePhi(float randomNumber){
    float phi = 2.0f * M_PI * randomNumber;
    return phi;
}

float getScatteringAngleTheta(float g, float randomNumber){
    if (g == 0){
        return acos(2.0f * randomNumber - 1.0f);
    }
    else{
        float temp = (1.0f - g * g) / (1 - g + 2 * g * randomNumber);
        return acos((1.0f + g * g - temp * temp) / (2 * g));
    }
}

ScatteringAngles getScatteringAngles(float rndPhi, float rndTheta,__global Photon *photons,
                                     __constant Material *materials, uint photonID)
{
    ScatteringAngles angles;
    float g = materials[photons[photonID].materialID].g;
    angles.phi = getScatteringAnglePhi(rndPhi);
    angles.theta = getScatteringAngleTheta(g, rndTheta);
    return angles;
}

// ----------- Test kernels -----------

//__kernel void getScatteringDistanceKernel(__global float * distanceBuffer, __global float * randomNumbers, float mu_t){
//    uint gid = get_global_id(0);
//    distanceBuffer[gid] = getScatteringDistance(randomNumbers, mu_t, gid);
//}
//
//__kernel void getScatteringAnglePhiKernel(__global float * angleBuffer,  __global float * randomNumbers){
//    uint gid = get_global_id(0);
//    angleBuffer[gid] = getScatteringAnglePhi(randomNumbers, gid);
//}
//
//__kernel void getScatteringAngleThetaKernel(__global float * angleBuffer,  __global float * randomNumbers, float g){
//    uint gid = get_global_id(0);
//    angleBuffer[gid] = getScatteringAngleTheta(randomNumbers, g, gid);
//}
