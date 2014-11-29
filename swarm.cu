#include <vector>
#include <GL/glut.h>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define numBoids 100000
#define numThreads 1000
#define timeStep 1
#define width 512
#define height 512
#define boidSize 1
using namespace std;

GLfloat ctrlpoints[numBoids][3];
float h_pos[numBoids][3];
float h_vel[numBoids][3];
float col[numBoids][3];
const int DATA_SIZE_BYTES = 3* numBoids * sizeof(float);
float * d_pos; 
float * d_vel;
float camzz = 1024.0;

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

__global__ void updateForces(float * d_pos, float * d_vel) {
	int k = ((blockIdx.x)*numThreads) + threadIdx.x;
	float force[3]={0,0,0};
	float centerCohesion[3]={0,0,0}; float centerSeparation[3]={0,0,0}; 
	int neighbourCohesion=0; int neighbourSeparation=0;
	//++++ COHESION and SEPARATION ++++++++++++++++++++++++++++++++++++++++++++++++
	int cohesionDist=100; float cohesionRatio=1.0;
	int separationDist=30; float separationRatio=0.0005;
	for(int i=0; i<numBoids; i++) {
		float posOther[3] = {d_pos[i*3+0], d_pos[i*3+1], d_pos[i*3+2]};
		posOther[0]-=d_pos[k*3+0]; posOther[1]-=d_pos[k*3+1]; posOther[2]-=d_pos[k*3+2];
		float distanceBetween = ( (posOther[0]*posOther[0]) + (posOther[1]*posOther[1]) + (posOther[2]*posOther[2]) );
		if(distanceBetween < cohesionDist*cohesionDist) {
			centerCohesion[0]+=d_pos[i*3+0]; centerCohesion[1]+=d_pos[i*3+1]; centerCohesion[2]+=d_pos[i*3+2]; 
			neighbourCohesion++;
		}
		if (distanceBetween < separationDist*separationDist) {
			centerSeparation[0]+=d_pos[i*3+0]; centerSeparation[1]+=d_pos[i*3+1]; centerSeparation[2]+=d_pos[i*3+2]; 
			neighbourSeparation++;
		}
	}
	centerCohesion[0]/=(float)neighbourCohesion; centerCohesion[1]/=(float)neighbourCohesion; centerCohesion[2]/=(float)neighbourCohesion; 
	centerSeparation[0]/=(float)neighbourSeparation; centerSeparation[1]/=(float)neighbourSeparation; centerSeparation[2]/=(float)neighbourSeparation; 
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	force[0] += ((centerCohesion[0]-d_pos[k*3+0])*cohesionRatio) + ((centerSeparation[0]-d_pos[k*3+0])*separationRatio);
	force[1] += ((centerCohesion[1]-d_pos[k*3+1])*cohesionRatio) + ((centerSeparation[1]-d_pos[k*3+1])*separationRatio);
	force[2] += ((centerCohesion[2]-d_pos[k*3+2])*cohesionRatio) + ((centerSeparation[2]-d_pos[k*3+2])*separationRatio);
	d_vel[k*3+0] += force[0]*timeStep;
	d_vel[k*3+1] += force[1]*timeStep;
	d_vel[k*3+2] += force[2]*timeStep;
}

__global__ void alignVel(float * d_pos, float * d_vel) {
	int k = ((blockIdx.x)*numThreads) + threadIdx.x;
/*	float alignmentDist = 70;
	float alignmentRatio = 0.002;
	float avgVel[3];
	int neighbourSize = 0;
	//++++ ALIGNMENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	for(int i=0; i<numBoids; i++) {
		float posOther[3] = {d_pos[i*3+0], d_pos[i*3+1], d_pos[i*3+2]};
		posOther[0]-=d_pos[k*3+0]; posOther[1]-=d_pos[k*3+1]; posOther[2]-=d_pos[k*3+2];
		float distanceBetween = ( (posOther[0]*posOther[0]) + (posOther[1]*posOther[1]) + (posOther[2]*posOther[2]) );
		if(distanceBetween < alignmentDist*alignmentDist) {
			avgVel[0]+=d_pos[i*3+0]; avgVel[1]+=d_pos[i*3+1]; avgVel[2]+=d_pos[i*3+2]; 
			neighbourSize++;
		}
	}
	avgVel[0]/=(float)neighbourSize; avgVel[1]/=(float)neighbourSize; avgVel[2]/=(float)neighbourSize; 
	d_vel[k*3+0] += (avgVel[0] - d_vel[k*3+0]) * alignmentRatio;
	d_vel[k*3+1] += (avgVel[1] - d_vel[k*3+1]) * alignmentRatio;
	d_vel[k*3+2] += (avgVel[2] - d_vel[k*3+2]) * alignmentRatio;
*/	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	d_pos[k*3+0] += d_vel[k*3+0]*timeStep;
	d_pos[k*3+1] += d_vel[k*3+1]*timeStep;
	d_pos[k*3+2] += d_vel[k*3+2]*timeStep;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	updateForces<<<(numBoids/numThreads), numThreads>>>(d_pos, d_vel);
	alignVel<<<(numBoids/numThreads), numThreads>>>(d_pos, d_vel);
	cudaMemcpy(h_pos, d_pos, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, d_vel, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
	glutPostRedisplay();
}

void render(void) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();  // glFrustum(-1.0, 1.0, -1.0, 1.0, 0.01, 512.0);
	gluPerspective(45.0, 1.0, 0.0, 1.0);  // parameters = (vertical FOV degrees, aspect ratio, near clipping, far clipping)
	gluLookAt(0, 0, camzz,    0, 0, 255.0,   0.0, 1.0, 0.0);  // parameters = (eye x-y-z,  center x-y-z,  up_direction x-y-z)
	glClear(GL_COLOR_BUFFER_BIT);

	for (int i = 0; i<numBoids; i++) {
		ctrlpoints[i][0] = (((int)(h_pos[i][0])) % 512) - 256;  if(ctrlpoints[i][0] < -256.0) ctrlpoints[i][0]=ctrlpoints[i][0] +512.0;
		ctrlpoints[i][1] = (((int)(h_pos[i][1])) % 512) - 256;  if(ctrlpoints[i][1] < -256.0) ctrlpoints[i][1]=ctrlpoints[i][1] +512.0;
		ctrlpoints[i][2] = ((int)(h_pos[i][2])) % 512;          if(ctrlpoints[i][2] < 0)      ctrlpoints[i][2]=ctrlpoints[i][2] +512.0;

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

float xDeltaAngle = 0.0;
float yDeltaAngle = 0.0;
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
	for (int i = 0; i<numBoids; i++) for (int j=0; j<3; j++) {
		h_pos[i][j] = rand()%512; 
		h_vel[i][j] = ((rand()%4)-2); 
		col[i][j] = ((rand()%100)/100.0); 
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

	cudaMalloc((void**) &d_pos, DATA_SIZE_BYTES);
	cudaMalloc((void**) &d_vel, DATA_SIZE_BYTES);
	cudaMemcpy(d_pos, h_pos, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	glutMainLoop();
	cudaFree(d_pos);
	cudaFree(d_vel);
	return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
