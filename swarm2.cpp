/* This is a version without GLM library */
/* since Garima is having some issues with GLM on her machine */

#include <vector>
#include <GL/glut.h>
#include "math.h"
#include <iostream>

//#include <glm/glm.hpp>
//using namespace glm;
using namespace std;

class dvec3{
  public:
    double x, y, z;
    dvec3() { x=0.0; y=0.0; z=0.0; }
    dvec3(double xx, double yy, double zz) {x=xx; y=yy; z=zz; }
    dvec3 negative() { return dvec3 (-x, -y, -z); }
    dvec3 vectAdd (dvec3 v) { return dvec3( x+v.x,  y+v.y,  z+v.z); }
    dvec3 vectMult (double scalar) { return dvec3(x*scalar, y*scalar, z*scalar); }
};

class ivec3{
  public:
    int x, y, z;
    ivec3() { x=0; y=0; z=0; }
    ivec3(int xx, int yy, int zz) {x=xx; y=yy; z=zz;}
    ivec3 negative() { return ivec3 (-x, -y, -z); }
    ivec3 vectAdd (ivec3 v) { return ivec3( x+v.x, y+v.y, z+v.z);}
    ivec3 vectAdd (dvec3 v) { return ivec3( (int)(x+v.x), (int)(y+v.y), (int)(z+v.z));}
    ivec3 vectMult (double scalar) { return ivec3((int)x*scalar, (int)y*scalar, (int)z*scalar); }
};

const unsigned int width = 512;
const unsigned int height = 512;
dvec3 centerOfGravity;

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

class Boid {
 public:
    dvec3 vel;
    dvec3 pos;
    dvec3 col;
    Boid(){ pos = dvec3(rand()%512, rand()%512, rand()%512); 
            vel = dvec3(rand()%4-2, rand()%4-2, rand()%4-2); 
            col = dvec3 ((rand()%10)/10.0, (rand()%10)/10.0, (rand()%10)/10.0);
          }
    Boid(ivec3 position, ivec3 velocity, dvec3 color, float radius) 
          { pos = dvec3(position.x, position.y, position.z); vel = dvec3(velocity.x, velocity.y, velocity.z); col = color; }
    void updatePos() { 
        vel = (vel.vectMult(0.99975)).vectAdd(((centerOfGravity.vectMult(0.001)).vectAdd(pos.negative())).vectMult(0.00025));
        pos = pos.vectAdd(vel);
        if (pos.x > 512) pos.x = pos.x - 512; if (pos.x < 0) pos.x = pos.x + 512;
        if (pos.y > 512) pos.y = pos.y - 512; if (pos.y < 0) pos.y = pos.y + 512;
        if (pos.z > 512) pos.z = pos.z - 512; if (pos.z < 0) pos.z = pos.z + 512;
    }
};

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

const int numBoids=1000;
vector <Boid*> boids;
void render(void);
void update(void);

GLfloat ctrlpoints[numBoids][3]; 

int main (int argc, char *argv[]) {
    for (int i=0; i<numBoids; i++) boids.push_back(new Boid());
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow("swarm");
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_FLAT);    
    glutDisplayFunc(render);
    glutIdleFunc(update);
    glutMainLoop();
    return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
    centerOfGravity = dvec3(0.0, 0.0, 0.0);
    for (int i=0; i<numBoids; i++) centerOfGravity = centerOfGravity.vectAdd(boids.at(i)->pos);
    for (int i=0; i<numBoids; i++) boids.at(i)->updatePos();
    glutPostRedisplay();
}

void render(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i=0; i<numBoids; i++) {
        ctrlpoints[i][0] = (boids.at(i)->pos.x - 256)/256;
        ctrlpoints[i][1] = (boids.at(i)->pos.y - 256)/256;
        ctrlpoints[i][2] = (boids.at(i)->pos.z - 256)/256;
        glPointSize(12.5*((512-boids.at(i)->pos.z)/512));
        glColor3f(boids.at(i)->col.x, boids.at(i)->col.y, boids.at(i)->col.z);
        glBegin(GL_POINTS);
            glVertex3fv(&ctrlpoints[i][0]);
        glEnd();
    }
    glFlush();
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
