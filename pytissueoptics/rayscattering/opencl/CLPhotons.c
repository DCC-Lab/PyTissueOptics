//----------------------------------------------------------------------------------------------------------------------
// RANDOM GENERATOR
//----------------------------------------------------------------------------------------------------------------------

uint wangHash(uint seed){
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float getRandomFloatValue(__global unsigned int *seedBuffer, unsigned int id){
     float result = 0.0f;
     while(result == 0.0f){
         uint rnd_seed = wangHash(seedBuffer[id]);
         seedBuffer[id] = rnd_seed;
         result = (float)rnd_seed / (float)UINT_MAX;
     }
     return result;
    }

 __kernel void fillRandomFloatBuffer(__global unsigned int *seedBuffer, __global float *randomFloatBuffer){
    int id = get_global_id(0);
    randomFloatBuffer[id] = getRandomFloatValue(seedBuffer, id);
    }


//----------------------------------------------------------------------------------------------------------------------
// VECTOR OPERATIONS
//----------------------------------------------------------------------------------------------------------------------

void normalizeVectorLocal(float4 *vector){
    float length = sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
    if (length != 0.0f) {
        vector->x /= length;
        vector->y /= length;
        vector->z /= length;
    }
    }

void normalizeVectorGlobal(__global float4 *vector){

    float length = sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
    if (length != 0.0f) {
        vector->x /= length;
        vector->y /= length;
        vector->z /= length;
    }
    }


void rotateAroundGlobal(__global float4 *mainVector, __global float4 *axisVector, float theta){
    normalizeVectorGlobal(axisVector);
    //normalizeVector(mainVector);
    float sint = sin(theta);
    float cost = cos(theta);
    float one_cost = 1.0f - cost;
    float ux = axisVector->x;
    float uy = axisVector->y;
    float uz = axisVector->z;
    float X = mainVector->x;
    float Y = mainVector->y;
    float Z = mainVector->z;
    float x = (cost + ux * ux * one_cost) * X \
            + (ux * uy * one_cost - uz * sint) * Y \
            + (ux * uz * one_cost + uy * sint) * Z;
    float y = (uy * ux * one_cost + uz * sint) * X \
            + (cost + uy * uy * one_cost) * Y \
            + (uy * uz * one_cost - ux * sint) * Z;
    float z = (uz * ux * one_cost - uy * sint) * X \
            + (uz * uy * one_cost + ux * sint) * Y \
            + (cost + uz * uz * one_cost) * Z;
    mainVector->x = x;
    mainVector->y = y;
    mainVector->z = z;
    }

void rotateAroundLocal(float4 *mainVector, float4 *axisVector, float theta){
    normalizeVectorLocal(axisVector);
    //normalizeVector(mainVector);
    float sint = sin(theta);
    float cost = cos(theta);
    float one_cost = 1.0f - cost;
    float ux = axisVector->x;
    float uy = axisVector->y;
    float uz = axisVector->z;
    float X = mainVector->x;
    float Y = mainVector->y;
    float Z = mainVector->z;
    float x = (cost + ux * ux * one_cost) * X \
            + (ux * uy * one_cost - uz * sint) * Y \
            + (ux * uz * one_cost + uy * sint) * Z;
    float y = (uy * ux * one_cost + uz * sint) * X \
            + (cost + uy * uy * one_cost) * Y \
            + (uy * uz * one_cost - ux * sint) * Z;
    float z = (uz * ux * one_cost - uy * sint) * X \
            + (uz * uy * one_cost + ux * sint) * Y \
            + (cost + uz * uz * one_cost) * Z;
    mainVector->x = x;
    mainVector->y = y;
    mainVector->z = z;
    }

float4 getAnyOrthogonal(float4 *vector){
    if (fabs(vector->z) < fabs(vector->x)){
        float4 r = (float4)(vector->y, -vector->x, 0.0f, 0.0f);
        return r;
        }

    else{
        float4 r = (float4)(0.0f, -vector->z, vector->y, 0.0f);
        return r;}
    }


// ---------------------------------------------------------------------------------------------------------------------
// CREATE PHOTONS
// ---------------------------------------------------------------------------------------------------------------------

__kernel void fillPencilPhotonsBuffer(__global photonStruct *photons, __global unsigned int *randomSeedBuffer, float4 position, float4 direction){
    uint gid = get_global_id(0);
    photons[gid].position = position;
    photons[gid].direction = direction;
    photons[gid].er = getAnyOrthogonal(&direction);
    photons[gid].weight = 1.0f;
    photons[gid].material_id = 0;
}

__kernel void fillIsotropicPhotonsBuffer(__global photonStruct *photons, __global unsigned int *randomSeedBuffer, float4 position){
    uint gid = get_global_id(0);
    photons[gid].position = position;
    float4 direction = (0.0f, 0.0f, 1.0f, 0.0f);
    float4 er = (1.0f, 0.0f, 0.0f, 0.0f);
    float randomFloat = getRandomFloatValue(randomSeedBuffer, gid);
    float phi = 2.0f * M_PI * randomFloat;
    float randomFloat2 = getRandomFloatValue(randomSeedBuffer, gid);
    float theta = acos(2.0f * randomFloat2 - 1.0f);
    rotateAroundLocal(&er, &direction, phi);
    rotateAroundLocal(&direction, &er, theta);
    photons[gid].direction = direction;
    photons[gid].er = er;
    photons[gid].weight = 1.0f;
    photons[gid].material_id = 0;
}


// ------------------------------------------------------------------------------------------------
// PROPAGATION PHYSICS
// ------------------------------------------------------------------------------------------------

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
    float g = materials[0].g;
    float mu_t = materials[0].mu_t;

    while (photons[gid].weight >= weightThreshold){
        logIndex = gid + stepIndex * dataSize;
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float distance = getScatteringDistance(randomNums, mu_t, gid);
        moveBy(photons, distance, gid);
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float phi = getScatteringAnglePhi(randomNums, gid);
        randomNums[gid] = getRandomFloatValue(seedBuffer, gid);
        float g = materials[0].g;
        float theta = getScatteringAngleTheta(randomNums, g, gid);
        scatterBy(photons, phi, theta, gid);
        interact(photons, materials, logger, gid, logIndex);
        stepIndex++;
    }
}
