
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
    rotateAroundAxisLocal(&er, &direction, phi);
    rotateAroundAxisLocal(&direction, &er, theta);
    photons[gid].direction = direction;
    photons[gid].er = er;
    photons[gid].weight = 1.0f;
    photons[gid].material_id = 0;
}