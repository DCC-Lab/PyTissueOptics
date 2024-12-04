#include "random.c"
#include "vectorOperators.c"
#include "scatteringMaterial.c"
#include "intersection.c"
#include "fresnel.c"

__constant int NO_SOLID_ID = -1;
__constant int NO_SURFACE_ID = -1;
__constant float MIN_ANGLE = 0.0001f;
__constant float MAX_RADIUS = 1.001f;
__constant int DEBUG_ID = -201400;

void moveBy(float distance, __global Photon *photons, uint photonID){
    photons[photonID].position += (distance * photons[photonID].direction);
}

void moveTo(float3 position, __global Photon *photons, uint photonID){
    photons[photonID].position = position;
}

void scatterBy(float phi, float theta, __global Photon *photons, uint photonID){
    rotateAroundAxisGlobal(&photons[photonID].er, &photons[photonID].direction, phi);
    rotateAroundAxisGlobal(&photons[photonID].direction, &photons[photonID].er, theta);
    photons[photonID].er = getAnyOrthogonalGlobal(&photons[photonID].direction);
}

void decreaseWeightBy(float delta_weight, __global Photon *photons, uint photonID){
    photons[photonID].weight -= delta_weight;
}

void interact(__global Photon *photons, __constant Material *materials, __global DataPoint *logger,
              uint logIndex, uint photonID){
    float delta_weight = photons[photonID].weight * materials[photons[photonID].materialID].albedo;
    decreaseWeightBy(delta_weight, photons, photonID);
//    if (photonID != DEBUG_ID){
//        return;
//    }
        // Print if position outside radius of 0.9
//    if (length(photons[photonID].position) > 0.901 && photons[photonID].solidID == 1){
//        printf("LOG photon out MIN RADIUS: %f\n", length(photons[photonID].position));
//        printf("\t ID: %d\n", photonID);
//        printf("\t POSITION: (%f, %f, %f)\n", photons[photonID].position.x, photons[photonID].position.y, photons[photonID].position.z);
//        photons[photonID].weight -= 0.1;
//    }

    logger[logIndex].x = photons[photonID].position.x;
    logger[logIndex].y = photons[photonID].position.y;
    logger[logIndex].z = photons[photonID].position.z;
    // Print if position radius is greater than MAX_RADIUS
    if (length(photons[photonID].position) > MAX_RADIUS){
        printf("LOG PHOTON OUTSIDE SOLID: %f\n", length(photons[photonID].position));
        printf("\t ID: %d\n", photonID);
        printf("\t POSITION: (%f, %f, %f)\n", photons[photonID].position.x, photons[photonID].position.y, photons[photonID].position.z);
        photons[photonID].weight -= 0.1;
    }
    logger[logIndex].delta_weight = delta_weight;
    logger[logIndex].solidID = photons[photonID].solidID;
    logger[logIndex].surfaceID = NO_SURFACE_ID;
}

void scatter(__global Photon *photons, __constant Material *materials, __global uint *seeds, __global DataPoint *logger,
             uint *logIndex, uint gid, uint photonID){

    float rndPhi = getRandomFloatValue(seeds, gid);
    float rndTheta = getRandomFloatValue(seeds, gid);
    ScatteringAngles angles = getScatteringAngles(rndPhi, rndTheta, photons, materials, photonID);

    scatterBy(angles.phi, angles.theta, photons, photonID);
    interact(photons, materials, logger, *logIndex, photonID);
    (*logIndex)++;
}

void roulette(float weightThreshold, __global Photon *photons, __global uint *seeds, uint gid, uint photonID){
    if (photons[photonID].weight >= weightThreshold || photons[photonID].weight == 0){
        return;
    }
    float randomFloat = getRandomFloatValue(seeds, gid);
    if (randomFloat < 0.1){
        photons[photonID].weight /= 0.1;
    }
    else{
        photons[photonID].weight = 0;
    }
}

void reflect(FresnelIntersection *fresnelIntersection, __global Photon *photons, uint photonID){
    rotateAround(&photons[photonID].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void refract(FresnelIntersection *fresnelIntersection, __global Photon *photons, uint photonID){
    rotateAround(&photons[photonID].direction, &fresnelIntersection->incidencePlane, fresnelIntersection->angleDeflection);
}

void logIntersection(Intersection *intersection, __global Photon *photons, __global Surface *surfaces,
                    __global DataPoint *logger, uint *logIndex, uint photonID){
    uint logID = *logIndex;
    logger[logID].x = photons[photonID].position.x;
    logger[logID].y = photons[photonID].position.y;
    logger[logID].z = photons[photonID].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = surfaces[intersection->surfaceID].insideSolidID;

    bool isLeavingSurface = dot(photons[photonID].direction, intersection->normal) > 0;
    int sign = isLeavingSurface ? 1 : -1;
    logger[logID].delta_weight = sign * photons[photonID].weight;
    (*logIndex)++;

    int outsideSolidID = surfaces[intersection->surfaceID].outsideSolidID;
    if (outsideSolidID == NO_SOLID_ID){
        return;
    }
    logID++;
    logger[logID].x = photons[photonID].position.x;
    logger[logID].y = photons[photonID].position.y;
    logger[logID].z = photons[photonID].position.z;
    logger[logID].surfaceID = intersection->surfaceID;
    logger[logID].solidID = outsideSolidID;
    logger[logID].delta_weight = -sign * photons[photonID].weight;
    (*logIndex)++;
}

float reflectOrRefract(Intersection *intersection, __global Photon *photons, __constant Material *materials,
        __global Surface *surfaces, __global DataPoint *logger, uint *logIndex, __global uint *seeds, uint gid, uint photonID){
    FresnelIntersection fresnelIntersection = computeFresnelIntersection(photons[photonID].direction, intersection,
                                                                         materials, surfaces, seeds, gid);
    if (photonID == DEBUG_ID) {
        printf("*Debug Fresnel - photonID: %d\n", photonID);
        printf("\t isReflected: %d\n", fresnelIntersection.isReflected);
        printf("\t isSmooth: %d\n", intersection->isSmooth);
        printf("\t smooth normal: %f %f %f\n", intersection->normal.x, intersection->normal.y, intersection->normal.z);
        printf("\t raw normal: %f %f %f\n", intersection->rawNormal.x, intersection->rawNormal.y, intersection->rawNormal.z);
        printf("\t dot(normal, direction): %f\n", dot(intersection->normal, photons[photonID].direction));
        printf("\t dot(rawNormal, direction): %f\n", dot(intersection->rawNormal, photons[photonID].direction));
        printf("\t angleDeflection: %f\n", fresnelIntersection.angleDeflection);
    }

    // If out angle is close to minAngle used, assert next distance is greater than some epsilon based on direction (so the resulting
    //  distance from triangle is greater than EPS_CATCH)...

    if (fresnelIntersection.isReflected) {
        if (intersection->isSmooth) {
            // Prevent reflection from crossing the raw surface.
            float smoothAngle = acos(dot(intersection->normal, intersection->rawNormal));
            float minDeflectionAngle = smoothAngle + fabs(fresnelIntersection.angleDeflection) / 2 + MIN_ANGLE;
            if (fabs(fresnelIntersection.angleDeflection) < minDeflectionAngle) {
                if (photonID == DEBUG_ID) {
                    printf("Deflection angle: %f\n", fresnelIntersection.angleDeflection);
                    printf("\t minDeflectionAngle: %f\n", minDeflectionAngle);
                }
                fresnelIntersection.angleDeflection = sign(fresnelIntersection.angleDeflection) * minDeflectionAngle;
            }
        }
        reflect(&fresnelIntersection, photons, photonID);
    }
    else {
        logIntersection(intersection, photons, surfaces, logger, logIndex, photonID);
        if (intersection->isSmooth) {
            // Prevent refraction from not crossing the raw surface.
            float maxDeflectionAngle = fabs(M_PI_F / 2 - acos(dot(intersection->rawNormal, photons[photonID].direction))) - MIN_ANGLE;
            if (fabs(fresnelIntersection.angleDeflection) > maxDeflectionAngle) {
                fresnelIntersection.angleDeflection = sign(fresnelIntersection.angleDeflection) * maxDeflectionAngle;
            }
        }
        refract(&fresnelIntersection, photons, photonID);

        float mut1 = materials[photons[photonID].materialID].mu_t;
        float mut2 = materials[fresnelIntersection.nextMaterialID].mu_t;
//        printf("Refracting with mut1: %f, mut2: %f\n", mut1, mut2);
//        printf("\t current material ID: %d\n", photons[photonID].materialID);
//        printf("\t next material ID: %d\n", fresnelIntersection.nextMaterialID);
        if (mut1 == 0) {
            intersection->distanceLeft = 0;
        } else if (mut2 != 0) {
            intersection->distanceLeft *= mut1 / mut2;
        } else {
            intersection->distanceLeft = INFINITY;
        }
        photons[photonID].materialID = fresnelIntersection.nextMaterialID;
        photons[photonID].solidID = fresnelIntersection.nextSolidID;
    }

    // TODO: if distanceLeft < some threshold around 1e-6, move the photon out of the surface by eps?
    //  ...This risks the photon from crossing another surface.
    // Or reconsider back epsilon catch

    return intersection->distanceLeft;
}

float propagateStep(float distance, __global Photon *photons, __constant Material *materials, Scene *scene,
                    __global uint *seeds, __global DataPoint *logger, uint *logIndex, uint gid, uint photonID){

    if (distance <= 0) {
        float mu_t = materials[photons[photonID].materialID].mu_t;
        float randomNumber = getRandomFloatValue(seeds, gid);
//        printf("++ New step (previous: %f)\n", distance);
        distance += getScatteringDistance(mu_t, randomNumber);
//        printf("\t New distance: %f\n", distance);
        if (distance < 0){
            // Not really possible until mu_t if very high (> 1000) and intense smoothing is applied (order 1 spheres).
            printf("! Distance still negative: %f\n", distance);
            distance = 0;
        }
    }

    Ray stepRay = {photons[photonID].position, photons[photonID].direction, distance};
    Intersection intersection = findIntersection(stepRay, scene, gid, photons[photonID].solidID);

//    float3 debugDirection = (float3)(-0.717926, -0.693910, 0.055414);
//    if (length(photons[photonID].direction - debugDirection) < 0.000001f){
//        float3 debugPosition = (float3)(-0.172085, 0.106354, -0.784270);
//        if (length(photons[photonID].position - debugPosition) < 0.000001f){
//            printf("Debugging photon %d\n", photonID);
//            printf("\t Distance: %f\n", distance);
//            printf("\t Solid ID: %d\n", photons[photonID].solidID);
//            printf("\t Intersection exists: %d\n", intersection.exists);
//            printf("\t Intersection distance: %f\n", intersection.distance);
//        }
//    }

    if (photonID == DEBUG_ID) {
        printf("Debug step\n");
        printf("\t Position: %f, %f, %f\n", photons[photonID].position.x, photons[photonID].position.y, photons[photonID].position.z);
        printf("\t Direction: %f, %f, %f\n", photons[photonID].direction.x, photons[photonID].direction.y, photons[photonID].direction.z);
        printf("\t Distance: %f\n", distance);
        printf("\t Solid ID: %d\n", photons[photonID].solidID);
        printf("\t Intersection exists: %d\n", intersection.exists);
        printf("\t Intersection distance: %f\n", intersection.distance);
    }

    float distanceLeft = 0;

    if (intersection.exists){
        moveTo(intersection.position, photons, photonID);
        distanceLeft = reflectOrRefract(&intersection, photons, materials, scene->surfaces, logger, logIndex, seeds, gid, photonID);
        if (photonID == DEBUG_ID){
            printf("[%d] Intersecting with polygon %d\n", photonID, intersection.polygonID);
            printf("\t Distance left: %f\n", distanceLeft);
        }

        // Check if intersection lies too close to a vertex.
        int closeToVertexID = -1;
        for (uint i = 0; i < 3; i++) {
            uint vertexID = scene->triangles[intersection.polygonID].vertexIDs[i];
            if (length(intersection.position - scene->vertices[vertexID].position) < 3e-7) {
                closeToVertexID = vertexID;
                break;
            }
        }

        // If too close to a vertex, move photon away slightly.
        if (closeToVertexID != -1) {
            int stepSign = 1;
            int solidIDTowardsNormal = scene->surfaces[intersection.surfaceID].outsideSolidID;
            if (solidIDTowardsNormal != photons[photonID].solidID) {
                stepSign = -1;
            }
            float3 stepCorrection = stepSign * scene->vertices[closeToVertexID].normal * EPS_CATCH;
            photons[photonID].position += stepCorrection;
            // TODO: force def of vertex normals in unsmoothed scenes.
//            printf("Photon too close to vertex ID %d.", closeToVertexID);
        }

    } else {
        if (distance == INFINITY){
            photons[photonID].weight = 0;
            return 0;
        }

        moveBy(distance, photons, photonID);

        scatter(photons, materials, seeds, logger, logIndex, gid, photonID);
    }

    return distanceLeft;
}

__kernel void propagate(uint maxPhotons, uint maxInteractions, float weightThreshold, uint workUnitsAmount, __global Photon *photons,
            __constant Material *materials, uint nSolids, __global Solid *solids, __global Surface *surfaces, __global Triangle *triangles,
            __global Vertex *vertices, __global SolidCandidate *solidCandidates, __global uint *seeds, __global DataPoint *logger){
    /*
    OpenCL implementation of the Python module Photon.
    See the Python module documentation for more details.
    */

    Scene scene = {nSolids, solids, surfaces, triangles, vertices, solidCandidates};

    uint gid = get_global_id(0);
    uint logIndex = gid * maxInteractions;
    uint maxLogIndex = logIndex + maxInteractions;

    uint photonCount = 0;
    uint maxSteps = 0;

    while (photonCount < maxPhotons){
        uint currentPhotonIndex = gid + (photonCount * workUnitsAmount);
        photons[currentPhotonIndex].er = getAnyOrthogonalGlobal(&photons[currentPhotonIndex].direction);

        float distance = 0;
        int nSteps = 0;
        float lastWeight = photons[currentPhotonIndex].weight;
        while (photons[currentPhotonIndex].weight != 0){
            if (logIndex >= (maxLogIndex -1)){  // Added -1 to avoid potential overflow when intersection logs twice
                return;
            }
            distance = propagateStep(distance, photons, materials, &scene,
                                     seeds, logger, &logIndex, gid, currentPhotonIndex);
            roulette(weightThreshold, photons, seeds, gid, currentPhotonIndex);

//             Safety check for stuck photons.
            // wont work for TIR without scattering
            if (lastWeight == photons[currentPhotonIndex].weight){
                nSteps++;
            } else {
                if (nSteps > maxSteps){
                    maxSteps = nSteps;
                }
                nSteps = 0;
                lastWeight = photons[currentPhotonIndex].weight;
            }

            if (nSteps > 1000){
                printf("!!!!!!! Discarding stuck photon %d\n", currentPhotonIndex);
                printf("\t Position: %f %f %f\n", photons[currentPhotonIndex].position.x, photons[currentPhotonIndex].position.y, photons[currentPhotonIndex].position.z);
                printf("\t Direction: %f %f %f\n", photons[currentPhotonIndex].direction.x, photons[currentPhotonIndex].direction.y, photons[currentPhotonIndex].direction.z);
                photons[currentPhotonIndex].weight = 0;
                break;
            }
        }
        photonCount++;
    }

    if (maxSteps > 10) {
        printf("Max steps: %d\n", maxSteps);
    }
}


// ---------------------------- TEST KERNELS ----------------------------


__kernel void moveByKernel(float distance, __global Photon *photons, uint photonID){
    moveBy(distance, photons, photonID);
}

__kernel void scatterByKernel(float phi, float theta, __global Photon *photons, uint photonID){
    photons[photonID].er = getAnyOrthogonalGlobal(&photons[photonID].direction);
    scatterBy(phi, theta, photons, photonID);
}

__kernel void decreaseWeightByKernel(float delta_weight, __global Photon *photons, uint photonID){
    decreaseWeightBy(delta_weight, photons, photonID);
}

__kernel void rouletteKernel(float weightThreshold, __global uint *seeds, __global Photon *photons, uint photonID){
    roulette(weightThreshold, photons, seeds, photonID, photonID);
}

__kernel void reflectKernel(float3 incidencePlane, float angleDeflection, __global Photon *photons, uint photonID){
    FresnelIntersection fresnelIntersection;
    fresnelIntersection.incidencePlane = incidencePlane;
    fresnelIntersection.angleDeflection = angleDeflection;
    reflect(&fresnelIntersection, photons, photonID);
}

__kernel void refractKernel(float3 incidencePlane, float angleDeflection, __global Photon *photons, uint photonID){
    FresnelIntersection fresnelIntersection;
    fresnelIntersection.incidencePlane = incidencePlane;
    fresnelIntersection.angleDeflection = angleDeflection;
    refract(&fresnelIntersection, photons, photonID);
}

__kernel void interactKernel(__constant Material *materials, __global DataPoint *logger,
                             uint logIndex, __global Photon *photons, uint photonID){
    interact(photons, materials, logger, logIndex, photonID);
}

__kernel void logIntersectionKernel(float3 normal, int surfaceID, __global Surface *surfaces,
                    __global DataPoint *logger, uint logIndex, __global Photon *photons, uint photonID){
    Intersection intersection;
    intersection.normal = normal;
    intersection.surfaceID = surfaceID;
    logIntersection(&intersection, photons, surfaces, logger, &logIndex, photonID);
}

__kernel void reflectOrRefractKernel(float3 normal, int surfaceID, float distanceLeft,
                                     __constant Material *materials, __global Surface *surfaces,
                                     __global DataPoint *logger, uint logIndex, __global uint *seeds,
                                     __global Photon *photons, uint photonID){
    Intersection intersection;
    intersection.normal = normal;
    intersection.surfaceID = surfaceID;
    intersection.distanceLeft = distanceLeft;
    reflectOrRefract(&intersection, photons, materials, surfaces, logger, &logIndex, seeds, photonID, photonID);
}

__kernel void propagateStepKernel(float distance, __constant Material *materials, __global Surface *surfaces,
                    __global uint *seeds, __global DataPoint *logger, uint logIndex,
                    __global Photon *photons, uint photonID){
    Scene scene;
    scene.surfaces = surfaces;
    uint gid = photonID;
    propagateStep(distance, photons, materials, &scene, seeds, logger, &logIndex, gid, photonID);
}
