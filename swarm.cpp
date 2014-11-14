#include <vector>
#include <GL/glut.h>
#include <glm/glm.hpp>
#include "math.h"
#include <iostream>

using namespace std;
using namespace glm;
const unsigned int width = 512;
const unsigned int height = 512;
dvec3 centerOfGravity;

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

class Boid {
 public:
    dvec3 vel;
    float r;
    dvec3 pos;
    dvec3 col;
    Boid () { pos = dvec3(rand()%512, rand()%512, rand()%512); vel = dvec3(rand()%4-2, rand()%4-2, rand()%4-2); col = dvec3 ((rand()%10)/10.0, (rand()%10)/10.0, (rand()%10)/10.0); r=4; }
    Boid (ivec3 position, ivec3 velocity, dvec3 color, float radius) { pos = position; vel = velocity; r = radius; col = color; }
    void updatePos() { 
    	vel = vel*0.999 + (centerOfGravity*0.002 - pos)*0.001;
        pos = pos + vel;
        if (pos.x > 512) pos.x = pos.x - 512; if (pos.x < 0) pos.x = pos.x + 512;
        if (pos.y > 512) pos.y = pos.y - 512; if (pos.y < 0) pos.y = pos.y + 512;
        if (pos.z > 512) pos.z = pos.z - 512; if (pos.z < 0) pos.z = pos.z + 512;
    }
};

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

int numBoids=500;
vector <Boid*> boids;
void render(void);
void update(void);

int main (int argc, char *argv[]) {
    for (int i=0; i<numBoids; i++) boids.push_back(new Boid());
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow("swarm");
    glutDisplayFunc(render);
    glutIdleFunc(update);
    glutMainLoop();
    return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	centerOfGravity = dvec3(0.0, 0.0, 0.0);
    for (int i=0; i<numBoids; i++) centerOfGravity = centerOfGravity + boids.at(i)->pos;
    for (int i=0; i<numBoids; i++) boids.at(i)->updatePos();
    glutPostRedisplay();
}

void render(void) {
    unsigned int image_plane[height][width][3];
    double zBuffer[height][width];
	for (int x=0; x<width; x++) for (int y=0; y<height; y++) 
		{ zBuffer[y][x]=999.9; image_plane[y][x][0]=0; image_plane[y][x][1]=0; image_plane[y][x][2]=0; }

	glViewport(0,0,512,512);
	glMatrixMode(GL_PROJECTION); glLoadIdentity();
	gluPerspective(90.0, 1.0, 0.0, 512.0); // parameters = (vertical FOV degrees, aspect ratio, near clipping, far clipping)

	glMatrixMode(GL_MODELVIEW); glLoadIdentity();
	gluLookAt(255.0, 255.0, 516.0,    255.0, 255.0, 255.0,   0.0, 1.0, 0.0); // parameters = (eye x-y-z,  center x-y-z,  up_direction x-y-z)

    GLdouble model[4*4], proj[4*4];
    GLint view[2*2]; view[0]=0; view[1]=0; view[2]=512; view[3]=512;
    glGetDoublev(GL_MODELVIEW_MATRIX, model);
    glGetDoublev(GL_PROJECTION_MATRIX, proj);

    for (int i=0; i<numBoids; i++) {
    	int thisObjectMinX=999, thisObjectMaxX=0, thisObjectMinY=999, thisObjectMaxY=0;
        for (int x=boids.at(i)->pos.x-boids.at(i)->r; x<boids.at(i)->pos.x+boids.at(i)->r; x++) {
            for (int y=boids.at(i)->pos.y-boids.at(i)->r; y<boids.at(i)->pos.y+boids.at(i)->r; y++) {
                GLdouble dbx,dby,dbz;
                gluProject(x, y, boids.at(i)->pos.z, model, proj, view, &dbx, &dby, &dbz);
                int projx = (int)round(dbx); int projy = (int)round(dby); 
                if ((projx>=0) && (projx<512) && (projy>=0) && (projy<512) && (zBuffer[projy][projx] > dbz)) {
                	zBuffer[projy][projx] = dbz;
                	if(thisObjectMinX>projx) thisObjectMinX=projx; if(thisObjectMaxX<projx) thisObjectMaxX=projx;
                	if(thisObjectMinY>projy) thisObjectMinY=projy; if(thisObjectMaxY<projy) thisObjectMaxY=projy;
                }
            }
        }
        for(int ix=thisObjectMinX; ix<=thisObjectMaxX; ix++) {
        	for(int iy=thisObjectMinY; iy<=thisObjectMaxY; iy++) {
				image_plane[iy][ix][0] = (unsigned int) (trunc (255*255*255*(255*boids.at(i)->col.x)));
                image_plane[iy][ix][1] = (unsigned int) (trunc (255*255*255*(255*boids.at(i)->col.y)));
                image_plane[iy][ix][2] = (unsigned int) (trunc (255*255*255*(255*boids.at(i)->col.z)));
            }
        }
    }
    glDrawPixels( width, height, GL_RGB, GL_UNSIGNED_INT, image_plane );
    glutSwapBuffers();
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
