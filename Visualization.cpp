#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533
#define MAX_NUM_CAM 1000

char *Path;
static bool g_bButton1Down = false;
static GLfloat g_ratio;
static GLfloat g_fViewDistance = 200 * VIEWING_DISTANCE_MIN;
static GLfloat g_nearPlane = 1.0, g_farPlane = 30000;
float g_coordAxisLength = 10;
static int g_Width = 1280, g_Height = 1024;
static int g_xClick = 0, g_yClick = 0;
static int g_mouseYRotate = 0, g_mouseXRotate = 0;
static int g_mouseYPan = 0, g_mouseXPan = 0;

static enum cam_mode { CAM_DEFAULT = 0, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;

static float g_lightPos[4] = { 0, 0, -3000, 0 };  // Position of light
static float g_lightPos2[4] = { 0, 0, 3000, 0 };  // Position of light
static float g_lightBright[4] = { 1, 1, 1, 1 };  // Position of light

const GLfloat Red[3] = { 1, 0, 0 }, Green[3] = { 0, 1, 0 }, Blue[3] = { 0, 0, 1 };
GLfloat PointsCentroid[3], ViewingLoc[3];
float InitLocFromCentroid[3];
GLfloat Scale = 0.25f, CameraSize = 10.0f, pointSize = 0.5f, normalSize = 20.f, arrowThickness = .1f;
vector<int> PickedPoints;
vector<int> PickCams;
VisualizationManager g_vis;

int nviews = 10, timeID = 0, otimeID = 0, maxTime, minTime;
bool drawPose = false, hasColor = false, hasNormal = false;
bool drawNormal = false, ThreeDRecon = false;
static bool ReCenterNeeded = false;

void Draw_Axes(void)
{
	glPushMatrix();

	glLineWidth(2.0);

	glBegin(GL_LINES);
	glColor3f(1, 0, 0); // X axis is red.
	glVertex3f(0, 0, 0);
	glVertex3f(g_coordAxisLength, 0, 0);
	glColor3f(0, 1, 0); // Y axis is green.
	glVertex3f(0, 0, 0);
	glVertex3f(0, g_coordAxisLength, 0);
	glColor3f(0, 0, 1); // z axis is blue.
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, g_coordAxisLength);
	glEnd();

	glPopMatrix();

	//	drawAxes(20);
}
void DrawCamera()
{
	//glPushMatrix();
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glColor3fv(Red);

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize); //
	glEnd();

	// we also have to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glColor3fv(Green);
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
	//glPopMatrix();
}
void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D)
{
	double x = x2 - x1;
	double y = y2 - y1;
	double z = z2 - z1;
	double L = sqrt(x*x + y*y + z*z);

	GLUquadricObj *quadObj;

	glPushMatrix();

	glTranslated(x1, y1, z1);

	if (x != 0.f || y != 0.f)
	{
		glRotated(atan2(y, x) / RADPERDEG, 0., 0., 1.);
		glRotated(atan2(sqrt(x*x + y*y), z) / RADPERDEG, 0., 1., 0.);
	}
	else if (z < 0)
		glRotated(180, 1., 0., 0.);

	glTranslatef(0, 0, L - 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, 2 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	glTranslatef(0, 0, -L + 4 * D);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
	gluDeleteQuadric(quadObj);

	quadObj = gluNewQuadric();
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);
	gluDisk(quadObj, 0.0, D, 32, 1);
	gluDeleteQuadric(quadObj);

	glPopMatrix();

}
void RenderObjects()
{

	if (otimeID != timeID && ThreeDRecon)
	{
		ReadCurrent3DGL(Path, hasColor, hasNormal, timeID);
		otimeID = timeID;
		printf("Loaded frame %d\n", timeID);
	}

	//Draw not picked 3D points
	for (unsigned int i = 0; i < g_vis.PointPosition.size(); ++i)
	{
		bool picked = false;
		for (unsigned int j = 0; j < PickedPoints.size(); j++)
			if (i == PickedPoints[j])
				picked = true;
		if (picked)
			continue;

		glLoadName(i + MAX_NUM_CAM);//for picking purpose
		glPushMatrix();
		glTranslatef(g_vis.PointPosition[i].x - PointsCentroid[0], g_vis.PointPosition[i].y - PointsCentroid[1], g_vis.PointPosition[i].z - PointsCentroid[2]);
		if (hasColor)
			glColor3f(g_vis.PointColor[i].x, g_vis.PointColor[i].y, g_vis.PointColor[i].z);
		else
			glColor3f(1.0, 0.0, 0.0);
		glutSolidSphere(pointSize, 4, 4);
		glPopMatrix();

		if (drawNormal)
		{
			glColor4f(0, 1, 0, 0.5f);
			Point3d newHeadPt = normalSize *g_vis.PointNormal[i] + g_vis.PointPosition[i];
			glPushMatrix();
			Arrow(g_vis.PointPosition[i].x - PointsCentroid[0], g_vis.PointPosition[i].y - PointsCentroid[1], g_vis.PointPosition[i].z - PointsCentroid[2],
				newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
			glPopMatrix();
		}
	}

	//Draw picked 3D points red
	for (unsigned int i = 0; i < PickedPoints.size(); i++)
	{
		int id = PickedPoints[i];
		glLoadName(id + MAX_NUM_CAM);//for picking purpose
		glPushMatrix();
		glTranslatef(g_vis.PointPosition[id].x - PointsCentroid[0], g_vis.PointPosition[id].y - PointsCentroid[1], g_vis.PointPosition[id].z - PointsCentroid[2]);
		//glColor3fv(Red);
		glColor3f(1.0, 0.0, 0.0);
		glutSolidSphere(pointSize, 4, 4);
		glPopMatrix();

		if (drawNormal)
		{
			glColor4f(0, 1, 0, 0.5f);
			Point3d newHeadPt = normalSize *g_vis.PointNormal[id] + g_vis.PointPosition[id];
			glPushMatrix();
			Arrow(g_vis.PointPosition[id].x - PointsCentroid[0], g_vis.PointPosition[id].y - PointsCentroid[1], g_vis.PointPosition[id].z - PointsCentroid[2],
				newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
			glPopMatrix();
		}
	}
	//draw camera 
	if (drawPose)
	{
		for (int j = 0; j < nviews; j++)
		{
			float CameraColor[3] = { 1, 0, 0 };
			if (g_vis.glCameraPoseInfo[j].size() > 0)
			{
				//Draw camere trajectory
				glPushMatrix();
				glBegin(GL_LINE_STRIP);
				for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
				{
					float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
					glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
					glColor3f(CameraColor[0], CameraColor[1], CameraColor[2]);
				}
				glEnd();

				for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
				{
					float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
					GLfloat* R = g_vis.glCameraPoseInfo[j].at(i).Rgl;
					glLoadName(i);//for picking purpose
					glPushMatrix();
					glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
					glMultMatrixf(R);
					DrawCamera();
					glPopMatrix();
				}
			}
		}
	}


	glFlush();
}
void display(void)
{
	// Clear frame buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up viewing transformation, looking down -Z axis
	glLoadIdentity();
	//gluLookAt(0, 0, 0, PointsCentroid[0], PointsCentroid[1], PointsCentroid[2], 0, -1, 0);
	//gluLookAt(ViewingLoc[0] - PointsCentroid[0], ViewingLoc[1] - PointsCentroid[1], ViewingLoc[2] - PointsCentroid[2], 0, 0, 0, 0, -1, 0);
	//gluLookAt(0 - PointsCentroid[0], 0 - PointsCentroid[1], 100 - PointsCentroid[2], 0, 0, 0, 0, -1, 0);
	gluLookAt(0, 0, 100, 0, 0, 0, 0, -1, 0);
	glTranslatef(-g_mouseXPan, -g_mouseYPan, -g_fViewDistance);

	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	// Set up the stationary light
	glLightfv(GL_LIGHT0, GL_POSITION, g_lightPos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_lightBright);
	glLightfv(GL_LIGHT1, GL_POSITION, g_lightPos2);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, g_lightBright);

	RenderObjects();
	Draw_Axes();
	glutSwapBuffers();
}

void InitGraphics(void)
{
	glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	glShadeModel(GL_SMOOTH);
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	//glEnable(GL_LIGHT1);
}
int Pick(int x, int y)
{
	GLuint buff[64];
	GLint hits, view[4];

	//selection data
	glSelectBuffer(64, buff);
	glGetIntegerv(GL_VIEWPORT, view);
	glRenderMode(GL_SELECT);

	//Push stack for picking
	glInitNames();
	glPushName(0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluPickMatrix(x, view[3] - y, 10.0, 10.0, view);
	gluPerspective(65.0, g_ratio, g_nearPlane, g_farPlane);

	glMatrixMode(GL_MODELVIEW);

	glutSwapBuffers();
	RenderObjects();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	hits = glRenderMode(GL_RENDER);
	glMatrixMode(GL_MODELVIEW);

	if (hits > 0)
		return buff[3];
	else
		return -1;
}
void SelectionFunction(int x, int y, bool append_rightClick = false)
{
	int pickedID = Pick(x, y);
	if (pickedID >= MAX_NUM_CAM)// pick points
	{
		pickedID -= MAX_NUM_CAM;

		if (ReCenterNeeded)
		{
			printf("New center picked: %d\n", pickedID);
			PointsCentroid[0] = g_vis.PointPosition[pickedID].x, PointsCentroid[1] = g_vis.PointPosition[pickedID].y, PointsCentroid[2] = g_vis.PointPosition[pickedID].z;
			ReCenterNeeded = false;
		}
		else
		{
			printf("Picked: %d\n", pickedID);
			PickedPoints.push_back(pickedID);
		}
	}
	else //if(pickedID<MAX_NUM_CAM)  //which means camera
		;
}
void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:             // ESCAPE key
		exit(0);
		break;
	case '+':
		pointSize += 0.1;
		printf("plotSize %f\n", pointSize);
		break;
	case 'n':
		break;
	case 'b':
		break;
	}
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		//do something here
		break;
	case GLUT_KEY_DOWN:
		//do something here
		break;
	case GLUT_KEY_LEFT:
		timeID--;
		if (timeID < minTime)
			timeID = minTime;
		break;
	case GLUT_KEY_RIGHT:
		timeID++;
		if (timeID > maxTime)
			timeID = maxTime;
		glutPostRedisplay();
		break;
	}

	glutPostRedisplay();
}
void MouseButton(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		g_bButton1Down = (state == GLUT_DOWN) ? TRUE : false;
		g_xClick = x;
		g_yClick = y;

		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
			g_camMode = CAM_ROTATE;
		else if (glutGetModifiers() == GLUT_ACTIVE_SHIFT)
			g_camMode = CAM_PAN;
		else if (glutGetModifiers() == GLUT_ACTIVE_ALT)
			g_camMode = CAM_ZOOM;
		else
		{
			g_camMode = CAM_DEFAULT;

			if (state == GLUT_DOWN) //picking single point
				SelectionFunction(x, y, false);
		}
	}
	else if (button == GLUT_RIGHT_BUTTON)
	{
		if (glutGetModifiers() == GLUT_ACTIVE_CTRL)
		{
			g_camMode = CAM_DEFAULT;

			ReCenterNeeded = true;
			SelectionFunction(x, y, false);
		}
		if (glutGetModifiers() == GLUT_ACTIVE_ALT)//Deselect
			PickedPoints.clear(), PickCams.clear();
	}
}
void MouseMotion(int x, int y)
{
	if (g_bButton1Down)
	{
		if (g_camMode == CAM_ZOOM)
			g_fViewDistance += (y - g_yClick);
		else if (g_camMode == CAM_ROTATE)
		{
			g_mouseXRotate += (x - g_xClick);
			g_mouseYRotate -= (y - g_yClick);
		}
		else if (g_camMode == CAM_PAN)
		{
			g_mouseXPan -= (x - g_xClick) / 3;
			g_mouseYPan -= (y - g_yClick) / 3;
		}

		g_xClick = x, g_yClick = y;

		glutPostRedisplay();
	}
}
void ReshapeGL(int width, int height)
{
	g_Width = width;
	g_Height = height;
	glViewport(0, 0, g_Width, g_Height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	g_ratio = (float)g_Width / g_Height;
	gluPerspective(65.0, g_ratio, g_nearPlane, g_farPlane);
	glMatrixMode(GL_MODELVIEW);
}
void visualization()
{
	char *myargv[1];
	int myargc = 1;
	myargv[0] = _strdup("SfM");
	glutInit(&myargc, myargv);

	// setup the size, position, and display mode for new windows 
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	// create and set up a window 
	glutCreateWindow("SfM!");
	InitGraphics();
	glutDisplayFunc(display);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);

	glutMainLoop();

}

int visualizationDriver(char *inPath, int nViews, int StartTime, int StopTime, bool Color, bool Normal, bool Pose, bool ThreeD)
{
	nviews = nViews;
	hasColor = Color, hasNormal = Normal,
		drawPose = Pose, ThreeDRecon = ThreeD;
	Path = inPath;
	if (ThreeDRecon)
	{
		Scale = Scale / 5;
		minTime = StartTime, maxTime = StopTime;
	}

	if (StartTime == -1)
		drawPose = false;

	VisualizationManager g_vis;
	if (!ThreeDRecon)
		ReadCurrentSfmGL(Path, hasColor, hasNormal);
	else
		ReadCurrent3DGL(Path, hasColor, hasNormal, timeID);

	if (drawPose)
		ReadCurrentPosesGL(Path, nViews, StartTime, StopTime);
	visualization();

	return 0;
}

void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, Point3i *AllColor, int npts)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		mat_invert(AllViewParas[viewID].R, iR);

		AllViewParas[viewID].Rgl[0] = AllViewParas[viewID].R[0], AllViewParas[viewID].Rgl[1] = AllViewParas[viewID].R[1], AllViewParas[viewID].Rgl[2] = AllViewParas[viewID].R[2], AllViewParas[viewID].Rgl[3] = 0.0;
		AllViewParas[viewID].Rgl[4] = AllViewParas[viewID].R[3], AllViewParas[viewID].Rgl[5] = AllViewParas[viewID].R[4], AllViewParas[viewID].Rgl[6] = AllViewParas[viewID].R[5], AllViewParas[viewID].Rgl[7] = 0.0;
		AllViewParas[viewID].Rgl[8] = AllViewParas[viewID].R[6], AllViewParas[viewID].Rgl[9] = AllViewParas[viewID].R[7], AllViewParas[viewID].Rgl[10] = AllViewParas[viewID].R[8], AllViewParas[viewID].Rgl[11] = 0.0;
		AllViewParas[viewID].Rgl[12] = 0, AllViewParas[viewID].Rgl[13] = 0, AllViewParas[viewID].Rgl[14] = 0, AllViewParas[viewID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[viewID].T, center, 3, 3, 1);
		AllViewParas[viewID].camCenter[0] = -center[0], AllViewParas[viewID].camCenter[1] = -center[1], AllViewParas[viewID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d: ", viewID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	bool color = false;
	if (AllColor != NULL)
		color = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (color)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveCurrentSfmGL2(char *path, CameraData *AllViewParas, vector<int>AvailViews, vector<Point3d>All3D, vector<Point3i>AllColor)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		mat_invert(AllViewParas[viewID].R, iR);

		AllViewParas[viewID].Rgl[0] = AllViewParas[viewID].R[0], AllViewParas[viewID].Rgl[1] = AllViewParas[viewID].R[1], AllViewParas[viewID].Rgl[2] = AllViewParas[viewID].R[2], AllViewParas[viewID].Rgl[3] = 0.0;
		AllViewParas[viewID].Rgl[4] = AllViewParas[viewID].R[3], AllViewParas[viewID].Rgl[5] = AllViewParas[viewID].R[4], AllViewParas[viewID].Rgl[6] = AllViewParas[viewID].R[5], AllViewParas[viewID].Rgl[7] = 0.0;
		AllViewParas[viewID].Rgl[8] = AllViewParas[viewID].R[6], AllViewParas[viewID].Rgl[9] = AllViewParas[viewID].R[7], AllViewParas[viewID].Rgl[10] = AllViewParas[viewID].R[8], AllViewParas[viewID].Rgl[11] = 0.0;
		AllViewParas[viewID].Rgl[12] = 0, AllViewParas[viewID].Rgl[13] = 0, AllViewParas[viewID].Rgl[14] = 0, AllViewParas[viewID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[viewID].T, center, 3, 3, 1);
		AllViewParas[viewID].camCenter[0] = -center[0], AllViewParas[viewID].camCenter[1] = -center[1], AllViewParas[viewID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d: ", viewID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	bool color = false;
	if (AllColor.size() != 0)
		color = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < All3D.size(); ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (color)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void ReadCurrentSfmGL(char *path, bool hasColor, bool hasNormal)
{
	char Fname[200];
	int viewID;

	CamInfo temp;
	sprintf(Fname, "%s/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d: ", &viewID) != EOF)
		{
			for (int jj = 0; jj < 16; jj++)
				fscanf(fp, "%f ", &temp.Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%f ", &temp.camCenter[jj]);

			g_vis.glCameraInfo.push_back(temp);
		}
		fclose(fp);
	}

	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (hasColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/3dGL.xyz", path); fp = fopen(Fname, "r");
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (hasColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}
		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	PointsCentroid[0] /= g_vis.PointPosition.size();
	PointsCentroid[1] /= g_vis.PointPosition.size();
	PointsCentroid[2] /= g_vis.PointPosition.size();

	InitLocFromCentroid[0] = 0.0, InitLocFromCentroid[1] = 0.0, InitLocFromCentroid[2] = 0.0;
	for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
	{
		InitLocFromCentroid[0] = InitLocFromCentroid[0] > abs(g_vis.PointPosition[ii].x - PointsCentroid[0]) ? InitLocFromCentroid[0] : abs(g_vis.PointPosition[ii].x - PointsCentroid[0]);
		InitLocFromCentroid[1] = InitLocFromCentroid[1] > abs(g_vis.PointPosition[ii].y - PointsCentroid[1]) ? InitLocFromCentroid[1] : abs(g_vis.PointPosition[ii].y - PointsCentroid[1]);
		InitLocFromCentroid[2] = InitLocFromCentroid[2] > abs(g_vis.PointPosition[ii].z - PointsCentroid[2]) ? InitLocFromCentroid[2] : abs(g_vis.PointPosition[ii].z - PointsCentroid[2]);
	}
	//Looking from the direction of the smallest variation
	if (InitLocFromCentroid[0] < InitLocFromCentroid[1] && InitLocFromCentroid[0] < InitLocFromCentroid[2])
		ViewingLoc[0] = InitLocFromCentroid[0], ViewingLoc[1] = 0, InitLocFromCentroid[2] = 0;
	else if (InitLocFromCentroid[1] < InitLocFromCentroid[0] && InitLocFromCentroid[1] < InitLocFromCentroid[2])
		ViewingLoc[0] = 0, ViewingLoc[1] = InitLocFromCentroid[1], InitLocFromCentroid[2] = 0;
	else
		ViewingLoc[0] = 0, ViewingLoc[1] = 0, InitLocFromCentroid[2] = InitLocFromCentroid[2];

	return;
}
void ReadCurrent3DGL(char *path, bool hasColor, bool hasNormal, int timeID)
{
	char Fname[200];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (hasColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	sprintf(Fname, "%s/3dGL_%d.xyz", path, timeID); FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (hasNormal)
		{
			fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
			g_vis.PointNormal.push_back(n3d);
		}
		if (hasColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}

		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	PointsCentroid[0] /= g_vis.PointPosition.size();
	PointsCentroid[1] /= g_vis.PointPosition.size();
	PointsCentroid[2] /= g_vis.PointPosition.size();

	InitLocFromCentroid[0] = 0.0, InitLocFromCentroid[1] = 0.0, InitLocFromCentroid[2] = 0.0;
	for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
	{
		InitLocFromCentroid[0] = InitLocFromCentroid[0] > abs(g_vis.PointPosition[ii].x - PointsCentroid[0]) ? InitLocFromCentroid[0] : abs(g_vis.PointPosition[ii].x - PointsCentroid[0]);
		InitLocFromCentroid[1] = InitLocFromCentroid[1] > abs(g_vis.PointPosition[ii].y - PointsCentroid[1]) ? InitLocFromCentroid[1] : abs(g_vis.PointPosition[ii].y - PointsCentroid[1]);
		InitLocFromCentroid[2] = InitLocFromCentroid[2] > abs(g_vis.PointPosition[ii].z - PointsCentroid[2]) ? InitLocFromCentroid[2] : abs(g_vis.PointPosition[ii].z - PointsCentroid[2]);
	}
	//Looking from the direction of the smallest variation
	if (InitLocFromCentroid[0] < InitLocFromCentroid[1] && InitLocFromCentroid[0] < InitLocFromCentroid[2])
		ViewingLoc[0] = InitLocFromCentroid[0], ViewingLoc[1] = 0, InitLocFromCentroid[2] = 0;
	else if (InitLocFromCentroid[1] < InitLocFromCentroid[0] && InitLocFromCentroid[1] < InitLocFromCentroid[2])
		ViewingLoc[0] = 0, ViewingLoc[1] = InitLocFromCentroid[1], InitLocFromCentroid[2] = 0;
	else
		ViewingLoc[0] = 0, ViewingLoc[1] = 0, InitLocFromCentroid[2] = InitLocFromCentroid[2];
	return;
}
void SaveCurrenPosesGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, int timeID)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		if (viewID < 0)
			continue;
		mat_invert(AllViewParas[viewID].R, iR);

		AllViewParas[viewID].Rgl[0] = AllViewParas[viewID].R[0], AllViewParas[viewID].Rgl[1] = AllViewParas[viewID].R[1], AllViewParas[viewID].Rgl[2] = AllViewParas[viewID].R[2], AllViewParas[viewID].Rgl[3] = 0.0;
		AllViewParas[viewID].Rgl[4] = AllViewParas[viewID].R[3], AllViewParas[viewID].Rgl[5] = AllViewParas[viewID].R[4], AllViewParas[viewID].Rgl[6] = AllViewParas[viewID].R[5], AllViewParas[viewID].Rgl[7] = 0.0;
		AllViewParas[viewID].Rgl[8] = AllViewParas[viewID].R[6], AllViewParas[viewID].Rgl[9] = AllViewParas[viewID].R[7], AllViewParas[viewID].Rgl[10] = AllViewParas[viewID].R[8], AllViewParas[viewID].Rgl[11] = 0.0;
		AllViewParas[viewID].Rgl[12] = 0, AllViewParas[viewID].Rgl[13] = 0, AllViewParas[viewID].Rgl[14] = 0, AllViewParas[viewID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[viewID].T, center, 3, 3, 1);
		AllViewParas[viewID].camCenter[0] = -center[0], AllViewParas[viewID].camCenter[1] = -center[1], AllViewParas[viewID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/DinfoGL_%d.txt", path, timeID);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d: ", viewID);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[viewID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveVideoCameraPosesGL(char *path, CameraData *AllViewParas, vector<int>AvailTime, int camID, int StartTime)
{
	char Fname[200];

	//Center = -iR*T 
	double iR[9], center[3];
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime.at(ii);
		if (timeID < 0)
			continue;
		mat_invert(AllViewParas[timeID].R, iR);

		AllViewParas[timeID].Rgl[0] = AllViewParas[timeID].R[0], AllViewParas[timeID].Rgl[1] = AllViewParas[timeID].R[1], AllViewParas[timeID].Rgl[2] = AllViewParas[timeID].R[2], AllViewParas[timeID].Rgl[3] = 0.0;
		AllViewParas[timeID].Rgl[4] = AllViewParas[timeID].R[3], AllViewParas[timeID].Rgl[5] = AllViewParas[timeID].R[4], AllViewParas[timeID].Rgl[6] = AllViewParas[timeID].R[5], AllViewParas[timeID].Rgl[7] = 0.0;
		AllViewParas[timeID].Rgl[8] = AllViewParas[timeID].R[6], AllViewParas[timeID].Rgl[9] = AllViewParas[timeID].R[7], AllViewParas[timeID].Rgl[10] = AllViewParas[timeID].R[8], AllViewParas[timeID].Rgl[11] = 0.0;
		AllViewParas[timeID].Rgl[12] = 0, AllViewParas[timeID].Rgl[13] = 0, AllViewParas[timeID].Rgl[14] = 0, AllViewParas[timeID].Rgl[15] = 1.0;

		mat_mul(iR, AllViewParas[timeID].T, center, 3, 3, 1);
		AllViewParas[timeID].camCenter[0] = -center[0], AllViewParas[timeID].camCenter[1] = -center[1], AllViewParas[timeID].camCenter[2] = -center[2];
	}

	sprintf(Fname, "%s/PinfoGL_%d.txt", path, camID);
	FILE *fp = fopen(Fname, "a+");
	for (int ii = 0; ii < AvailTime.size(); ii++)
	{
		int timeID = AvailTime.at(ii);
		fprintf(fp, "%d ", timeID + StartTime);
		for (int jj = 0; jj < 16; jj++)
			fprintf(fp, "%.16f ", AllViewParas[timeID].Rgl[jj]);
		for (int jj = 0; jj < 3; jj++)
			fprintf(fp, "%.16f ", AllViewParas[timeID].camCenter[jj]);
		fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void ReadCurrentPosesGL(char *path, int nviews, int StartTime, int StopTime)
{
	char Fname[200];
	g_vis.glCameraPoseInfo = new vector<CamInfo>[(StopTime - StartTime + 1)*nviews];

	int timeID;
	CamInfo temp;
	for (int ii = 0; ii < nviews; ii++)
	{
		sprintf(Fname, "%s/PinfoGL_%d.txt", path, ii);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			continue;
		while (fscanf(fp, "%d ", &timeID) != EOF)
		{

			for (int jj = 0; jj < 16; jj++)
				fscanf(fp, "%f ", &temp.Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%f ", &temp.camCenter[jj]);

			g_vis.glCameraPoseInfo[ii].push_back(temp);
		}
		fclose(fp);
	}

	return;
}










