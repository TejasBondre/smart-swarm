#include <glm/glm.hpp>
#include "math.h"
#include <iostream>

using namespace glm;

class Boid {
 public:
    dvec3 vel, pos;
    Boid (dvec3 position, dvec3 velocity) { pos = position; vel = velocity; }
    void updatePos() { 
        pos = pos + vel;
    }
};

int main (int argc, char *argv[]) {
    dvec3 position = dvec3(rand()%512, rand()%512, rand()%512);
    dvec3 velocity = dvec3(rand()%512, rand()%512, rand()%512);
    Boid* b = new Boid(position, velocity);
    for (int i=0; i<10; i++) {
        b->updatePos();
        std::cout <<"position = ("<< b->pos.x <<", "<< b->pos.y <<", "<< b->pos.z <<
          ")   and   velocity = ("<< b->vel.x <<", "<< b->vel.y <<", "<< b->vel.z <<")"<< std::endl;
    }
    return 0;
}
