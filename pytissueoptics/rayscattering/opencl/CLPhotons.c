// RANDOM
uint wang_hash(uint seed)
    {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }


void randomize_buffer_seed(__global unsigned int * rnd_buffer, int id)
    {
     uint rnd_int = wang_hash(id);
     rnd_buffer[id] = rnd_int;
    }

float random_float(__global unsigned int * rnd_buffer, int id)
    {
     uint maxint = 0;
     maxint--;
     uint rnd_int = wang_hash(rnd_buffer[id]);
     rnd_buffer[id] = rnd_int;
     return ((float)rnd_int) / (float)maxint;
    }


 __kernel void randomize_seed_init(__global unsigned int *rnd_buffer)
    {
       int id = get_global_id(0);
       randomize_buffer_seed(rnd_buffer, id);
    }

 __kernel void fill_random_float_buffer(__global unsigned int * rnd_buffer, float * float_buffer)
    {
    int id = get_global_id(0);
    float_buffer[id] = random_float(rnd_buffer, id);
    }


// ------------------------------------------------------------------------------------------------
// PHOTON PHYSICS
// ------------------------------------------------------------------------------------------------

void normalizeVector(float4 *vector)
    {
    float length = sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
    vector->x /= length;
    vector->y /= length;
    vector->z /= length;
    }

void decreaseWeightBy(__global photonStruct *photons, float delta_weight, __global loggerStruct *logger, uint gid)
{
        photons[gid].weight -= delta_weight;
}

void interact(__global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger, uint globalCounter, uint datasize, uint gid)
{
    float delta_weight = photons[gid].weight * materials[photons[gid].material_id].albedo;
    decreaseWeightBy(photons, delta_weight, logger, gid);
    uint logIndex = gid + globalCounter * datasize;
    logger[logIndex].position = photons[gid].position;
    logger[logIndex].delta_weight = delta_weight;

}

void moveBy(__global photonStruct *photons, float distance, uint gid)
{
        photons[gid].position += distance * photons[gid].direction;
}

float getScatteringDistance(__global photonStruct *photons,__constant materialStruct *materials, __global float * randomNums, uint gid)
{
    return -log(randomNums[gid]) / materials[photons[gid].material_id].mu_t;
}

float getScatteringAnglePhi(__global photonStruct *photons, __global float * randomNums, uint gid)
{
    float phi = 2.0f * M_PI * randomNums[gid];
    return phi;
}

float getScatteringAngleTheta(__global photonStruct *photons,__constant materialStruct *materials,  __global float * randomNums, uint gid)
{
    if (materials[gid].g == 0){
        return acos(2.0f * randomNums[gid] - 1.0f);
    }
    else{
        float temp = (1.0f - materials[gid].g * materials[gid].g) / (1 - materials[gid].g + 2 * materials[gid].g * randomNums[gid]);
        return acos((1.0f + materials[gid].g * materials[gid].g - temp * temp) / (2 * materials[gid].g));
    }
}


float4 rotateAround(float4 mainVector, float4 axisVector, float theta)
{
    normalizeVector(&axisVector);
    normalizeVector(&mainVector);
    float sint = sin(theta);
    float cost = cos(theta);
    float one_cost = 1.0f - cost;
    float ux = axisVector.x;
    float uy = axisVector.y;
    float uz = axisVector.z;
    float X = mainVector.x;
    float Y = mainVector.y;
    float Z = mainVector.z;
    float x = (cost + ux * ux * one_cost) * X \
            + (ux * uy * one_cost - uz * sint) * Y \
            + (ux * uz * one_cost + uy * sint) * Z;
    float y = (uy * ux * one_cost + uz * sint) * X \
            + (cost + uy * uy * one_cost) * Y \
            + (uy * uz * one_cost - ux * sint) * Z;
    float z = (uz * ux * one_cost - uy * sint) * X \
            + (uz * uy * one_cost + ux * sint) * Y \
            + (cost + uz * uz * one_cost) * Z;
    return (float4)(x, y, z, 0.0f);
}

void scatterBy(__global photonStruct *photons, float phi, float theta, uint gid)
{
    photons[gid].er = rotateAround(photons[gid].er, photons[gid].direction, phi);
    photons[gid].direction = rotateAround(photons[gid].direction, photons[gid].er, theta);
}
// PROPAGATE KERNELS
__kernel void propagate(uint datasize, __global photonStruct *photons, __constant materialStruct *materials, __global loggerStruct *logger, __global float *randomNums, __global uint *rnd_buffer)
{
    unsigned int gid = get_global_id(0);
    unsigned int globalCounter = 0;
    while(photons[gid].weight > 0.0001)
    {
//    for (int i = 0; i < 30; i++){
        randomNums[gid] = random_float(rnd_buffer, gid);
        float distance = getScatteringDistance(photons, materials, randomNums, gid);
        moveBy(photons, distance, gid);
        randomNums[gid] = random_float(rnd_buffer, gid);
        float phi = getScatteringAnglePhi(photons, randomNums, gid);
        randomNums[gid] = random_float(rnd_buffer, gid);
        float theta = getScatteringAngleTheta(photons, materials, randomNums, gid);
        //scatterBy(photons, phi, theta, gid);
        interact(photons, materials, logger, globalCounter, datasize, gid);
        globalCounter++;
    }
    photons[gid].weight = 0.0f;
}
