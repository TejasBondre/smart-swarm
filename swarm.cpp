#include <iostream>
#include <glm/glm.hpp>
#include "math.h"
#include <GL/glut.h>

using namespace glm;

class Boid {
 public:
    dvec3 vel, pos;
    Boid (dvec3 position, dvec3 velocity) { pos = position; vel = velocity; }
    void updatePos() { 
        pos = pos + vel;
    }
};

const unsigned int width = 512;
const unsigned int height = 512;
Boid* b;
void render(void);

int main (int argc, char *argv[]) {
    dvec3 position = dvec3(255, 255, 255);
    dvec3 velocity = dvec3(rand()%2, rand()%2, rand()%2);
    b = new Boid(position, velocity);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow("swarm");
    glutDisplayFunc(render);
    glutIdleFunc(render);
    glutMainLoop();

    return 0;
}


void render(void) {
    unsigned int image_plane[height][width][3];
    for (int x=0; x<width; x++) { 
        for (int y=0; y<height; y++) { 
            image_plane[y][x][0]=0;
            image_plane[y][x][1]=0;
            image_plane[y][x][2]=0;
        }
    }

    b->updatePos();
    std::cout <<"position = ("<< b->pos.x <<", "<< b->pos.y <<", "<< b->pos.z <<
      ")   and   velocity = ("<< b->vel.x <<", "<< b->vel.y <<", "<< b->vel.z <<")"<< std::endl;

    for (int x=0; x<512; x++) {
        for (int y=0; y<512; y++) {
            if ((x==b->pos.x) && (y==b->pos.y)) {
                image_plane[y][x][0] = (unsigned int) (trunc (255*255*255*(255*0.9)));
                image_plane[y][x][1] = (unsigned int) (trunc (255*255*255*(255*0.9)));
                image_plane[y][x][2] = (unsigned int) (trunc (255*255*255*(255*0.9)));
            }
        }
    }
    glDrawPixels( width, height, GL_RGB, GL_UNSIGNED_INT, image_plane );
    glutSwapBuffers();
}
