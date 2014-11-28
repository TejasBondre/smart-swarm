#include <vector>
#include <GL/glut.h>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;
const int numBoids = 100;
const unsigned int width = 512;
const unsigned int height = 512;
double timeStep = 1.0;
GLfloat ctrlpoints[numBoids][3];
int boidSize = 5;
double zMove = 0.0, xMove = 0.0;
double camzz = 1024.0, camxx = 0.0;
class dvec3{
public:
	double x, y, z;
	dvec3() { x = 0.0; y = 0.0; z = 0.0; }
	dvec3(double xx, double yy, double zz) { x = xx; y = yy; z = zz; }
	dvec3 negative() { return dvec3(-x, -y, -z); }
	dvec3 vectAdd(dvec3 v) { return dvec3(x + v.x, y + v.y, z + v.z); }
	dvec3 vectMult(double scalar) { return dvec3(x*scalar, y*scalar, z*scalar); }
	double length(){ return sqrt(x*x + y*y + z*z); }
};

class ivec3{
public:
	int x, y, z;
	ivec3() { x = 0; y = 0; z = 0; }
	ivec3(int xx, int yy, int zz) { x = xx; y = yy; z = zz; }
	ivec3 negative() { return ivec3(-x, -y, -z); }
	ivec3 vectAdd(ivec3 v) { return ivec3(x + v.x, y + v.y, z + v.z); }
	ivec3 vectAdd(dvec3 v) { return ivec3((int)(x + v.x), (int)(y + v.y), (int)(z + v.z)); }
	ivec3 vectMult(double scalar) { return ivec3((int)x*scalar, (int)y*scalar, (int)z*scalar); }
};

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

class Boid {
public:
	dvec3 vel;
	dvec3 pos;
	dvec3 col;
	dvec3 center;
	int count;
	double cohesionDist, cohesionRatio, alignmentDist, alignmentRatio, separationDist, separationRatio;

	Boid() {
		pos = dvec3(rand() % 512, rand() % 512, rand() % 512);
		vel = dvec3(rand() % 4 - 2, rand() % 4 - 2, rand() % 4 - 2);
		col = dvec3((rand() % 100) / 100.0, (rand() % 100) / 100.0, (rand() % 100) / 100.0);
		cohesionDist=100; cohesionRatio=0.001;
		alignmentDist=70; alignmentRatio=0.02; 
		separationDist=30; separationRatio=-0.005; 
	}

	vector<Boid*> getNeighbours(vector <Boid*> boidArray, dvec3 position, int radius) {
		vector<Boid*> neighbours;
		for (int i = 0; i < numBoids; i++) {
			Boid* other = boidArray[i];
			dvec3 posOther = other->pos;
			double distanceBetween = (posOther.vectAdd(position.negative())).length();
			if (distanceBetween < radius) {
				neighbours.push_back(other);
			}
		}
		return neighbours;
	}

	void updateForces(vector <Boid*> boidArray) {
		dvec3 force = dvec3(0,0,0);
		dvec3 distanceFromCenter;
		vector<Boid*> neighbours;
		//++++ COHESION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		center = dvec3();
		neighbours = getNeighbours(boidArray, pos, cohesionDist);
		for (int j = 0; j< neighbours.size(); j++) {
			center = center.vectAdd(neighbours[j]->pos);
		}
		center = center.vectMult(1 / (double)neighbours.size());
		distanceFromCenter = center.vectAdd(pos.negative());
		force = force.vectAdd(distanceFromCenter.vectMult(cohesionRatio));

		//++++ SEPARATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		center = dvec3();
		neighbours = getNeighbours(boidArray, pos, separationDist);
		for (int j = 0; j< neighbours.size(); j++) {
			center = center.vectAdd(neighbours[j]->pos);
		}
		center = center.vectMult(1 / (double)neighbours.size());
		distanceFromCenter = center.vectAdd(pos.negative());
		force = force.vectAdd(distanceFromCenter.vectMult(separationRatio));

		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		vel = vel.vectAdd(force.vectMult(timeStep));
	}

	void alignVel(vector<Boid*> boidArray) {
		dvec3 avgVel = dvec3(0, 0, 0);
		dvec3 velocityDifference;
		vector<Boid*> neighbours;
		//++++ ALIGNMENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		avgVel = dvec3();
		neighbours = getNeighbours(boidArray, pos, alignmentDist);
		for (int j = 0; j< neighbours.size(); j++) {
			avgVel = avgVel.vectAdd(neighbours[j]->vel);
		}
		avgVel = avgVel.vectMult(1 / (double)neighbours.size());
		velocityDifference = avgVel.vectAdd(vel.negative());
		vel = vel.vectAdd(velocityDifference.vectMult(alignmentRatio));

		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		updatePos();
	}

	void updatePos(void){ 
		pos = pos.vectAdd(vel.vectMult(timeStep)); 
	}
};
vector <Boid*> boids;

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	for (int i = 0; i<numBoids; i++) boids.at(i)->updateForces(boids);
	for (int i = 0; i<numBoids; i++) boids.at(i)->alignVel(boids);
	glutPostRedisplay();
}

void render(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();  // glFrustum(-1.0, 1.0, -1.0, 1.0, 0.01, 512.0);
	gluPerspective(45.0, 1.0, 0.0, 1.0);  				// parameters = (vertical FOV degrees, aspect ratio, near clipping, far clipping)
	camzz = camzz - zMove;
	camxx = camxx - xMove;
	gluLookAt(0, 0, camzz,    0, 0, 255.0,   0.0, 1.0, 0.0);  // parameters = (eye x-y-z,  center x-y-z,  up_direction x-y-z)

	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i<numBoids; i++) {
		ctrlpoints[i][0] = (((int)(boids.at(i)->pos.x)) % 512) - 256;  if(ctrlpoints[i][0] < -256.0) ctrlpoints[i][0]=ctrlpoints[i][0] +512.0;
		ctrlpoints[i][1] = (((int)(boids.at(i)->pos.y)) % 512) - 256;  if(ctrlpoints[i][1] < -256.0) ctrlpoints[i][1]=ctrlpoints[i][1] +512.0;
		ctrlpoints[i][2] = ((int)(boids.at(i)->pos.z)) % 512;          if(ctrlpoints[i][2] < 0)      ctrlpoints[i][2]=ctrlpoints[i][2] +512.0;

		glColor3f(boids.at(i)->col.x, boids.at(i)->col.y, boids.at(i)->col.z);
		glBegin(GL_TRIANGLES);
		glVertex3f(ctrlpoints[i][0]-boidSize, ctrlpoints[i][1]-boidSize, ctrlpoints[i][2]);
		glVertex3f(ctrlpoints[i][0]+boidSize, ctrlpoints[i][1]-boidSize, ctrlpoints[i][2]);
		glVertex3f(ctrlpoints[i][0]-boidSize, ctrlpoints[i][1]+boidSize, ctrlpoints[i][2]);
		glEnd();

		glLineWidth(1.0);
		if(1) {		// display bounding box
			glColor3f(1.0, 0.0, 0.0);
			glBegin(GL_LINES); glVertex3f(-256, 256, 0);    glVertex3f(256, 256, 0);  glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256, 0);      glVertex3f(256, -256, 0);    glEnd();
			glBegin(GL_LINES); glVertex3f(-256, 256, 0);    glVertex3f(-256, -256, 0);      glEnd();
			glBegin(GL_LINES); glVertex3f(256, 256, 0);  glVertex3f(256, -256, 0);    glEnd();
			glColor3f(0.0, 1.0, 0.0);
			glBegin(GL_LINES); glVertex3f(-256, 256, 0);    glVertex3f(-256, 256, 512);  glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256, 0);      glVertex3f(-256, -256, 512);    glEnd();
			glBegin(GL_LINES); glVertex3f(256, -256, 0);    glVertex3f(256, -256, 512);  glEnd();
			glBegin(GL_LINES); glVertex3f(256, 256, 0);  glVertex3f(256, 256, 512);glEnd();
			glColor3f(0.0, 0.0, 1.0);
			glBegin(GL_LINES); glVertex3f(-256, 256, 512);  glVertex3f(256, 256, 512);glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256, 512);    glVertex3f(256, -256, 512);  glEnd();
			glBegin(GL_LINES); glVertex3f(-256, 256, 512);  glVertex3f(-256, -256, 512);    glEnd();
			glBegin(GL_LINES); glVertex3f(256, 256, 512);glVertex3f(256, -256, 512);  glEnd();
		}
		else {		// display co-ordinate axes
			glColor3f(1.0, 0.0, 0.0); glBegin(GL_LINES); glVertex3f(256, 0, 0); glVertex3f(-256, 0, 0); glEnd();
			glColor3f(0.0, 1.0, 0.0); glBegin(GL_LINES); glVertex3f(0, 256, 0); glVertex3f(0, -256, 0); glEnd();
			glColor3f(0.0, 0.0, 1.0); glBegin(GL_LINES); glVertex3f(0, 0, 512); glVertex3f(0, 0, 0); glEnd();
		}
	}
	glFlush();
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

double xDeltaAngle = 0.0;
double yDeltaAngle = 0.0;
int isDragging = 0;
int xDragStart = 0;
int yDragStart = 0;
void mouseMove(int xx, int yy) {
    if (isDragging) {
        xDeltaAngle = (xx - xDragStart) * 0.5;
        yDeltaAngle = (yy - yDragStart) * 0.5;
    }
}
void mouseButton(int button, int state, int xx, int yy) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            isDragging = 1;
            xDragStart = xx;
            yDragStart = yy;
        }
        else  {
            glMatrixMode(GL_MODELVIEW);
            glTranslatef(0, 0, 255);
			glRotatef(xDeltaAngle, 0, 1, 0);
			glRotatef(yDeltaAngle, 1, 0, 0);
			glTranslatef(0, 0, -255);
            isDragging = 0;
        }
    }
    if (button == 3) { camzz = camzz - 15.0; }
    if (button == 4) { camzz = camzz + 15.0; }
}

void pressKey(int key, int xx, int yy) {
    switch (key) {
        case GLUT_KEY_UP : 
            glMatrixMode(GL_MODELVIEW);
            glTranslatef(0, 0, 255);
			glRotatef(9, 1, 0, 0);
			glTranslatef(0, 0, -255);
        break;
        case GLUT_KEY_DOWN :
            glMatrixMode(GL_MODELVIEW);
            glTranslatef(0, 0, 255);
			glRotatef(-9, 1, 0, 0);
			glTranslatef(0, 0, -255);
        break;
        case GLUT_KEY_LEFT : 
            glMatrixMode(GL_MODELVIEW);
            glTranslatef(0, 0, 255);
			glRotatef(9, 0, 1, 0);
			glTranslatef(0, 0, -255);
        break;
        case GLUT_KEY_RIGHT : 
            glMatrixMode(GL_MODELVIEW);
            glTranslatef(0, 0, 255);
			glRotatef(-9, 0, 1, 0);
			glTranslatef(0, 0, -255);
        break;
    }
} 

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

int main(int argc, char *argv[]) {
	srand (time(NULL));
	for (int i = 0; i<numBoids; i++) boids.push_back(new Boid());
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("swarm");
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);
	glutDisplayFunc(render);
	glutIdleFunc(update);
    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMove);
    glutSpecialFunc(pressKey);
	glutMainLoop();
	return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
