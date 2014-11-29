#include <vector>
#include <GL/glut.h>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define numBoids 100
using namespace std;
//const int numBoids = 100;
const unsigned int width = 512;
const unsigned int height = 512;
GLfloat ctrlpoints[numBoids][3];
int boidSize = 5;
float camzz = 1024.0;

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

const int DATA_SIZE_BYTES = 3* numBoids * sizeof(float);
float h_pos[numBoids][3];
float h_vel[numBoids][3];
float col[numBoids][3];
/*
inline void negate(float ans[3], float x[3]) { ans[0]=0-x[0]; ans[1]=0-x[1]; ans[2]=0-x[2]; }
inline void vAdd(float ans[3], float x[3], float y[3]) { ans[0]=x[0]+y[0]; ans[1]=x[1]+y[1]; ans[2]=x[2]+y[2]; }
inline void vSubt(float ans[3], float x[3], float y[3]) { ans[0]=x[0]-y[0]; ans[1]=x[1]-y[1]; ans[2]=x[2]-y[2]; }
inline void vMult(float ans[3], float x[3], float scalar) { ans[0]=x[0]*scalar; ans[1]=x[1]*scalar; ans[2]=x[2]*scalar; }
inline float length(float x[3]) { return sqrt ( (x[0]*x[0]) + (x[1]*x[1]) + (x[2]*x[2]) ); }

||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

float ** d_pos; 
float ** d_vel;

__global__ void updateForces(float ** d_pos, float ** d_vel) {
	int k = threadIdx.x;
	float timeStep = 1.0;
	float cohesionDist=100, cohesionRatio=1.0;
	float separationDist=30, separationRatio=-0.0005; 
	float force[3] = {0,0,0};
	float center[3];
	int neighbourSize;
	//++++ COHESION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	neighbourSize = 0;
	center[0]=0; center[1]=0; center[2]=0;
	for(int i=0; i<numBoids; i++) {
		float posOther[3] = {d_pos[i][0], d_pos[i][1], d_pos[i][2]};
		posOther[0]-=d_pos[k][0]; posOther[1]-=d_pos[k][1]; posOther[2]-=d_pos[k][2];
		float distanceBetween = ( (posOther[0]*posOther[0]) + (posOther[1]*posOther[1]) + (posOther[2]*posOther[2]) );
		if(distanceBetween < cohesionDist*cohesionDist) {
			center[0]+=d_pos[i][0]; center[1]+=d_pos[i][1]; center[2]+=d_pos[i][2]; 
			neighbourSize++;
		}
	}
	center[0]/=(float)neighbourSize; center[1]/=(float)neighbourSize; center[2]/=(float)neighbourSize; 
	force[0] += (center[0]-d_pos[k][0])*cohesionRatio;
	force[1] += (center[1]-d_pos[k][1])*cohesionRatio;
	force[2] += (center[2]-d_pos[k][2])*cohesionRatio;

	//++++ SEPARATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	neighbourSize = 0;
	center[0]=0; center[1]=0; center[2]=0;
	for(int i=0; i<numBoids; i++) {
		float posOther[3] = {d_pos[i][0], d_pos[i][1], d_pos[i][2]};
		posOther[0]-=d_pos[k][0]; posOther[1]-=d_pos[k][1]; posOther[2]-=d_pos[k][2];
		float distanceBetween = ( (posOther[0]*posOther[0]) + (posOther[1]*posOther[1]) + (posOther[2]*posOther[2]) );
		if (distanceBetween < separationDist*separationDist) {
			center[0]+=d_pos[i][0]; center[1]+=d_pos[i][1]; center[2]+=d_pos[i][2]; 
			neighbourSize++;
		}
	}
	center[0]/=(float)neighbourSize; center[1]/=(float)neighbourSize; center[2]/=(float)neighbourSize; 
	force[0] += (center[0]-d_pos[k][0])*separationRatio;
	force[1] += (center[1]-d_pos[k][1])*separationRatio;
	force[2] += (center[2]-d_pos[k][2])*separationRatio;

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	d_vel[k][0] += force[0]*timeStep;
	d_vel[k][1] += force[1]*timeStep;
	d_vel[k][2] += force[2]*timeStep;
}

__global__ void alignVel(float ** d_pos, float ** d_vel) {
	int k = threadIdx.x;
	float timeStep = 1.0;
	float alignmentDist=70, alignmentRatio=0.0002; 
	float avgVel[3];
	int neighbourSize = 0;
	//++++ ALIGNMENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	for(int i=0; i<numBoids; i++) {
		float posOther[3] = {d_pos[i][0], d_pos[i][1], d_pos[i][2]};
		posOther[0]-=d_pos[k][0]; posOther[1]-=d_pos[k][1]; posOther[2]-=d_pos[k][2];
		float distanceBetween = ( (posOther[0]*posOther[0]) + (posOther[1]*posOther[1]) + (posOther[2]*posOther[2]) );
		if(distanceBetween < alignmentDist*alignmentDist) {
			avgVel[0]+=d_pos[i][0]; avgVel[1]+=d_pos[i][1]; avgVel[2]+=d_pos[i][2]; 
			neighbourSize++;
		}
	}
	avgVel[0]/=(float)neighbourSize; avgVel[1]/=(float)neighbourSize; avgVel[2]/=(float)neighbourSize; 
	d_vel[k][0] += (avgVel[0] - d_vel[k][0]) * alignmentRatio;
	d_vel[k][1] += (avgVel[1] - d_vel[k][1]) * alignmentRatio;
	d_vel[k][2] += (avgVel[2] - d_vel[k][2]) * alignmentRatio;

	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	d_pos[k][0] += d_vel[k][0]*timeStep;
	d_pos[k][1] += d_vel[k][1]*timeStep;
	d_pos[k][2] += d_vel[k][2]*timeStep;
}


/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/

void update(void) {
	cudaMemcpy(d_pos, h_pos, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	updateForces<<<1, numBoids>>>(d_pos, d_vel);
	cudaMemcpy(h_pos, d_pos, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, d_vel, DATA_SIZE_BYTES, cudaMemcpyDeviceToHost);

	cudaMemcpy(d_pos, h_pos, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vel, h_vel, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
	alignVel<<<1, numBoids>>>(d_pos, d_vel);
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
	for (int i = 0; i<numBoids; i++) {
		for (int j=0; j<3; j++) {
			h_pos[i][j] = rand()%512; 
			h_vel[i][j] = ((rand()%1)-0.5); 
			col[i][j] = ((rand()%100)/100.0); 
		}
	}
	cudaMalloc((void**) &d_pos, DATA_SIZE_BYTES);
	cudaMalloc((void**) &d_vel, DATA_SIZE_BYTES);

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
	cudaFree(d_pos);
	cudaFree(d_vel);
	return 0;
}

/*||||| insert comment here to describe next logical code block. if no description |||||*/
/*||||| - yet, then retain this comment as a separator to make code reading easier |||||*/
