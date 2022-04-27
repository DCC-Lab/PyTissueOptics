void normalizeVector(__global float4 *vector){
    float length = sqrt(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);
    vector->x /= length;
    vector->y /= length;
    vector->z /= length;
    }




void rotateAroundAxis(__global float4 *mainVector, __global float4 *axisVector, float theta){
    normalizeVector(axisVector);
    normalizeVector(mainVector);
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

__kernel void rotateAroundAxisKernel(__global float4 *vector, __global float4 *axis, __global float *angle){
    int i = get_global_id(0);
    rotateAroundAxis(&vector[i], &axis[i], angle[i]);
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
