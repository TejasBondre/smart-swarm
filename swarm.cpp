#include <vector>
#include <GL/glut.h>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <glm/glm.hpp>
//using namespace glm;
using namespace std;
const int numBoids = 100;
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

const unsigned int width = 512;
const unsigned int height = 512;
double timeStep = 1.0;
double speedLimit = 4;
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
		center = dvec3();
		vector<Boid*> neighbours = getNeighbours(boidArray, pos, cohesionDist);
		for (int j = 0; j< neighbours.size(); j++) {
			center = center.vectAdd(neighbours[j]->pos);
		}
		center = center.vectMult(1/(double)neighbours.size());
		dvec3 distanceFromCenter = center.vectAdd(pos.negative());
		dvec3 force = distanceFromCenter.vectMult(cohesionRatio);
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		center = dvec3();
		neighbours = getNeighbours(boidArray, pos, separationDist);
		for (int j = 0; j< neighbours.size(); j++) {
			center = center.vectAdd(neighbours[j]->pos);
		}
		center = center.vectMult(1 / (double)neighbours.size());
		distanceFromCenter = center.vectAdd(pos.negative());
		force = force.vectAdd(distanceFromCenter.vectMult(separationRatio));
		vel = vel.vectAdd(force.vectMult(timeStep));
	}

	void alignVel(vector<Boid*> boidArray) {
		dvec3 avgVel = dvec3();
		vector<Boid*> neighbours = getNeighbours(boidArray, pos, alignmentDist);
		for (int j = 0; j< neighbours.size(); j++) {
			avgVel = avgVel.vectAdd(neighbours[j]->vel);
		}
		avgVel = avgVel.vectMult(1 / (double)neighbours.size());
		double speed = vel.length();
		if(speed>4) speed=4;
		vel = vel.vectAdd(avgVel.vectMult(alignmentRatio));
		vel = vel.vectMult(speed / (vel.length()));
		updatePos();
	}

	void updatePos(void){ 
		pos = pos.vectAdd(vel.vectMult(timeStep)); 
		/* if (pos.x > 512) pos.x = pos.x - 512; if (pos.x < 0) pos.x = pos.x + 512;
		if (pos.y > 512) pos.y = pos.y - 512; if (pos.y < 0) pos.y = pos.y + 512;
		if (pos.z > 512) pos.z = pos.z - 512; if (pos.z < 0) pos.z = pos.z + 512; */
	}
};

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/


vector <Boid*> boids;
void render(void);
void update(void);
GLfloat ctrlpoints[numBoids][3];

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
	glutMainLoop();
	return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	for (int i = 0; i<numBoids; i++) boids.at(i)->updateForces(boids);
	for (int i = 0; i<numBoids; i++) boids.at(i)->alignVel(boids);
	glutPostRedisplay();
}

void render(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	for (int i = 0; i<numBoids; i++) {
		ctrlpoints[i][0] = ((((int)(boids.at(i)->pos.x)) % 512) - 256) / 256.0;
		ctrlpoints[i][1] = ((((int)(boids.at(i)->pos.y)) % 512) - 256) / 256.0;
		ctrlpoints[i][2] = ((((int)(boids.at(i)->pos.z)) % 512) - 256) / 256.0;
		for (int j=0; j<3; j++) if(ctrlpoints[i][j] < -1.0) ctrlpoints[i][j]=ctrlpoints[i][j] +2.0;
		glPointSize(2*((512 - boids.at(i)->pos.z) / 512));
		glColor3f(boids.at(i)->col.x, boids.at(i)->col.y, boids.at(i)->col.z);
		glBegin(GL_POINTS);
		glVertex3fv(&ctrlpoints[i][0]);
		glEnd();
	}
	glFlush();
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/