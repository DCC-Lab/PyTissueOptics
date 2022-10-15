
struct Intersection {
    int status;
    float distance;
};

typedef struct Intersection Intersection;

Intersection getIntersection(float distance) {
    Intersection intersection;
    intersection.status = 0;
    return intersection;
}
