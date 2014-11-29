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
double timeStep = 0.001;
GLfloat ctrlpoints[numBoids][3];
int boidSize = 5;
double camzz = 1024.0;
double cohesionDist=100, cohesionRatio=0.01;
double alignmentDist=70, alignmentRatio=0.02; 
double separationDist=30, separationRatio=-0.005; 

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

float pos[numBoids][3];
float vel[numBoids][3];
float col[numBoids][3];
float center[3];
float temp[3];

inline void negate(float ans[3], float x[3]) { ans[0]=0-x[0]; ans[1]=0-x[1]; ans[2]=0-x[2]; }
inline void vAdd(float ans[3], float x[3], float y[3]) { ans[0]=x[0]+y[0]; ans[1]=x[1]+y[1]; ans[2]=x[2]+y[2]; }
inline void vSubt(float ans[3], float x[3], float y[3]) { ans[0]=x[0]-y[0]; ans[1]=x[1]-y[1]; ans[2]=x[2]-y[2]; }
inline void vMult(float ans[3], float x[3], float scalar) { ans[0]=x[0]*scalar; ans[1]=x[1]*scalar; ans[2]=x[2]*scalar; }
inline float leng(float x[3]) { return sqrt ( (x[0]*x[0]) + (x[1]*x[1]) + (x[2]*x[2]) ); }

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
	
void getNeighbours(float position[3], int radius, int neighbours[numBoids]) {
	int ii=0;
	for (int i = 0; i < numBoids; i++) {
		float posOther[3] = {pos[i][0], pos[i][1], pos[i][2]};
		vSubt(temp, posOther, position);
		float distanceBetween = leng (temp);
		if (distanceBetween < radius) {
			neighbours[ii] = i;
			ii++;
		}
	}
	for(int i=ii; i<numBoids; i++) neighbours[i]=-1;
}

void updatePos(int k) { 
	vMult(temp, vel[k], timeStep);
	vAdd(pos[k], pos[k], temp);
}

void updateForces(int k) {
	float force[3];
	float distanceFromCenter[3];
	int neighbours[numBoids];
	//++++ COHESION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	vMult(center, center, 0);
	getNeighbours(pos[k], cohesionDist, neighbours);
	double neighbourSize;
	for (int j = 0; j< numBoids; j++) {
		if (neighbours[j] >= 0)
			vAdd(center, center, pos[neighbours[j]]);
		else { neighbourSize = j; break; }
	}
	vMult(center, center, (1.0 / (float)neighbourSize));
	vSubt(distanceFromCenter, center, pos[k]);
	vMult(distanceFromCenter, distanceFromCenter, cohesionRatio);
	vAdd(force, force, distanceFromCenter);

	//++++ SEPARATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	vMult(center, center, 0);
	getNeighbours(pos[k], separationDist, neighbours);
	for (int j = 0; j< numBoids; j++) {
		if (neighbours[j] >= 0)
			vAdd(center, center, pos[neighbours[j]]);
		else { neighbourSize = j; break; }
	}
	vMult(center, center, (1.0/ (float)neighbourSize));
	vSubt(distanceFromCenter, center, pos[k]);
	vMult(distanceFromCenter, distanceFromCenter, separationRatio);
	vAdd(force, force, distanceFromCenter);


	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	vMult(force, force, timeStep);
	vAdd(vel[k], vel[k], force);
}

void alignVel(int k) {
	float avgVel[3];
	float velocityDifference[3];
	int neighbours[numBoids];
	//++++ ALIGNMENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	getNeighbours(pos[k], alignmentDist, neighbours);
	double neighbourSize;
	for (int j = 0; j< numBoids; j++) {
		if (neighbours[j] >= 0)
			vAdd(avgVel, avgVel, pos[neighbours[j]]);
		else { neighbourSize = j; break; }
	}
	vMult(avgVel, avgVel, (1.0 / (float)neighbourSize));
	vSubt(velocityDifference, avgVel, vel[k]);
	vMult(velocityDifference, velocityDifference, alignmentRatio);
	vAdd(vel[k], vel[k], velocityDifference);

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	updatePos(k);
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	for (int i = 0; i<numBoids; i++) updateForces(i);
	for (int i = 0; i<numBoids; i++) alignVel(i);
	glutPostRedisplay();
}

void render(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();  // glFrustum(-1.0, 1.0, -1.0, 1.0, 0.01, 512.0);
	gluPerspective(45.0, 1.0, 0.0, 1.0);  // parameters = (vertical FOV degrees, aspect ratio, near clipping, far clipping)
	gluLookAt(0, 0, camzz,    0, 0, 255.0,   0.0, 1.0, 0.0);  // parameters = (eye x-y-z,  center x-y-z,  up_direction x-y-z)
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i<numBoids; i++) {
		ctrlpoints[i][0] = (((int)(pos[i][0])) % 512) - 256;  if(ctrlpoints[i][0] < -256.0) ctrlpoints[i][0]=ctrlpoints[i][0] +512.0;
		ctrlpoints[i][1] = (((int)(pos[i][1])) % 512) - 256;  if(ctrlpoints[i][1] < -256.0) ctrlpoints[i][1]=ctrlpoints[i][1] +512.0;
		ctrlpoints[i][2] = ((int)(pos[i][2])) % 512;          if(ctrlpoints[i][2] < 0)      ctrlpoints[i][2]=ctrlpoints[i][2] +512.0;

		glColor3f(col[i][0], col[i][1], col[i][2]);
		glBegin(GL_TRIANGLES);
		glVertex3f(ctrlpoints[i][0]-boidSize, ctrlpoints[i][1]-boidSize, ctrlpoints[i][2]);
		glVertex3f(ctrlpoints[i][0]+boidSize, ctrlpoints[i][1]-boidSize, ctrlpoints[i][2]);
		glVertex3f(ctrlpoints[i][0]-boidSize, ctrlpoints[i][1]+boidSize, ctrlpoints[i][2]);
		glEnd();

		glLineWidth(1.0);
		if(1) {		// display bounding box
			glColor3f(1.0, 0.0, 0.0);
			glBegin(GL_LINES); glVertex3f(-256,  256,   0); glVertex3f( 256,  256,   0); glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256,   0); glVertex3f( 256, -256,   0); glEnd();
			glBegin(GL_LINES); glVertex3f(-256,  256,   0); glVertex3f(-256, -256,   0); glEnd();
			glBegin(GL_LINES); glVertex3f( 256,  256,   0); glVertex3f( 256, -256,   0); glEnd();
			glColor3f(0.0, 1.0, 0.0);
			glBegin(GL_LINES); glVertex3f(-256,  256,   0); glVertex3f(-256,  256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256,   0); glVertex3f(-256, -256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f( 256, -256,   0); glVertex3f( 256, -256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f( 256,  256,   0); glVertex3f( 256,  256, 512); glEnd();
			glColor3f(0.0, 0.0, 1.0);
			glBegin(GL_LINES); glVertex3f(-256,  256, 512); glVertex3f( 256,  256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f(-256, -256, 512); glVertex3f( 256, -256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f(-256,  256, 512); glVertex3f(-256, -256, 512); glEnd();
			glBegin(GL_LINES); glVertex3f( 256,  256, 512); glVertex3f( 256, -256, 512); glEnd();
		}
		else {		// display co-ordinate axes
			glColor3f(1.0, 0.0, 0.0); glBegin(GL_LINES); glVertex3f(256, 0, 0); glVertex3f(-256, 0, 0); glEnd();
			glColor3f(0.0, 1.0, 0.0); glBegin(GL_LINES); glVertex3f(0, 256, 0); glVertex3f(0, -256, 0); glEnd();
			glColor3f(0.0, 0.0, 1.0); glBegin(GL_LINES); glVertex3f(0, 0, 512); glVertex3f(0,    0, 0); glEnd();
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
	for (int i = 0; i<numBoids; i++) {
		pos[i][0]=rand()%512;  pos[i][1]=rand()%512;  pos[i][2]=rand()%512;
		vel[i][0]=((rand()%4)-2);  vel[i][1]=((rand()%4)-2);  vel[i][2]=((rand()%4)-2);
		col[i][0]=((rand()%100)/100.0);  col[i][1]=((rand()%100)/100.0);  col[i][2]=((rand()%100)/100.0);
	}
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("smart-swarm");
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
