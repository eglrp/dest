#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

static bool g_bButton1Down = false;
static GLfloat g_fViewDistance = 20 * VIEWING_DISTANCE_MIN;
static GLfloat g_nearPlane = .1;
static GLfloat g_farPlane = 30000;
static int g_Width = 1280, g_Height = 1024;
static int g_xClick = 0, g_yClick = 0;
static int g_mouseYRotate = 0, g_mouseXRotate = 0;
static int g_mouseYPan = 0, g_mouseXPan = 0;

static enum cam_mode { CAM_DEFAULT = 0, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;

static float g_lightPos[4] = { 0, 0, -3000, 0 };  // Position of light
static float g_lightPos2[4] = { 0, 0, 3000, 0 };  // Position of light
static float g_lightBright[4] = { 1, 1, 1, 1 };  // Position of light

const GLfloat Red[3] = { 1, 0, 0 };
const GLfloat Green[3] = { 0, 1, 0 };
GLfloat Scale = 0.25f;
VisualizationManager g_vis;

int nviews = 10, timeID = 0, otimeID = 0, maxTime, minTime;
bool drawPose = false, hasColor = false, ThreeDRecon = false;
char Path[] = "D:/Juggling";

void DrawCamera()
{
	//glPushMatrix();
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glColor3fv(Red);
	//glRotatef(angle, X,Y,Z);

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*Scale, 0.5*Scale, 1 * Scale); //
	glVertex3f(0, 0, 0); //
	glVertex3f(0.5*Scale, -0.5*Scale, 1 * Scale); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*Scale, 0.5*Scale, 1 * Scale); //
	glVertex3f(0, 0, 0); //
	glVertex3f(-0.5*Scale, -0.5*Scale, 1 * Scale); //
	glEnd();

	// we also have to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glColor3fv(Green);
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*Scale, 0.5*Scale, 1 * Scale);
	glVertex3f(-0.5*Scale, 0.5*Scale, 1 * Scale);
	glVertex3f(-0.5*Scale, -0.5*Scale, 1 * Scale);
	glVertex3f(0.5*Scale, -0.5*Scale, 1 * Scale);
	glVertex3f(0.5*Scale, 0.5*Scale, 1 * Scale);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
	//glPopMatrix();
}
void RenderObjects()
{
	if (otimeID != timeID && ThreeDRecon)
	{
		ReadCurrent3DGL(Path, hasColor, timeID);
		otimeID = timeID;
		printf("Loaded frame %d\n", timeID);
	}

	for (unsigned int i = 0; i < g_vis.PointPosition.size(); ++i)
	{
		glPushMatrix();
		glTranslatef(g_vis.PointPosition[i].x, g_vis.PointPosition[i].y, g_vis.PointPosition[i].z);
		if (hasColor)
			glColor3f(g_vis.PointColor[i].x, g_vis.PointColor[i].y, g_vis.PointColor[i].z);
		else
			glColor3f(1.0, 0.0, 0.0);
		glutSolidSphere(0.5*Scale, 4, 4);
		glPopMatrix();
	}

	//draw pyramid
	if (drawPose && !ThreeDRecon)
	{
		for (int j = 0; j < nviews; j++)
		{
			float CameraColor[3] = { 1, 0, 0 };// 1.0*rand() / RAND_MAX, 1.0*rand() / RAND_MAX, 1.0*rand() / RAND_MAX
			if (g_vis.glCameraPoseInfo[j].size() > 0)
			{
				/*glPushMatrix();
				glBegin(GL_LINE_STRIP);
				for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
				{
				float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
				glVertex3f(centerPt[0], centerPt[1], centerPt[2]);
				glColor3f(CameraColor[0], CameraColor[1], CameraColor[2]);
				}
				glEnd();*/

				for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
				{
					float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
					GLfloat* R = g_vis.glCameraPoseInfo[j].at(i).Rgl;
					glPushMatrix();
					glTranslatef(centerPt[0], centerPt[1], centerPt[2]);
					glMultMatrixf(R);
					DrawCamera();
					glPopMatrix();
				}
			}
		}
	}
	else if (!ThreeDRecon)
	{
		for (unsigned int i = 0; i < g_vis.glCameraInfo.size(); ++i)
		{
			float* centerPt = g_vis.glCameraInfo[i].camCenter;
			GLfloat* R = g_vis.glCameraInfo[i].Rgl;
			glPushMatrix();
			glTranslatef(centerPt[0], centerPt[1], centerPt[2]);
			glMultMatrixf(R);
			DrawCamera();
			glPopMatrix();
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
	//gluLookAt(0, 0, -g_fViewDistance, 0, 0,-1 , 0, 1, 0);
	gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
	glTranslatef(-g_mouseXPan, -g_mouseYPan, g_fViewDistance);
	//gluLookAt(g_mouseXPan/3, g_mouseYPan/3, -g_fViewDistance,PointPosition[0].x,PointPosition[0].y ,PointPosition[0].z , 0, 1, 0);

	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	glTranslatef(-g_vis.g_trajactoryCenter.x, -g_vis.g_trajactoryCenter.y, -g_vis.g_trajactoryCenter.z);

	// Set up the stationary light
	glLightfv(GL_LIGHT0, GL_POSITION, g_lightPos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_lightBright);
	glLightfv(GL_LIGHT1, GL_POSITION, g_lightPos2);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, g_lightBright);
	// Render the scene
	RenderObjects();
	// Make sure changes appear onscreen
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
void Keyboard(unsigned char key, int x, int y)
{

	switch (key)
	{
	case 27:             // ESCAPE key
		exit(0);
		break;
		//case 'a':
		//case 'A':
		// exit (0);

		//SelectFromMenu(MENU_LIGHTING);
		break;
	case 'p':
		//SelectFromMenu(MENU_POLYMODE);
		break;
	case 't':
		//SelectFromMenu(MENU_TEXTURING);
		break;
	case 'n':
	{
		timeID++;
		if (timeID > maxTime)
			timeID = maxTime;
		glutPostRedisplay();
		break;
	}
	case 'b':
	{
		timeID--;
		if (timeID < minTime)
			timeID = minTime;
		glutPostRedisplay();
		break;
	}
	}
}
void MouseButton(int button, int state, int x, int y)
{
	// Respond to mouse button presses.
	// If button1 pressed, mark this state so we know in motion function.
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
			g_camMode = CAM_DEFAULT;
	}
}
void MouseMotion(int x, int y)
{
	// If button1 pressed, zoom in/out if mouse is moved up/down.
	if (g_bButton1Down)
	{
		if (g_camMode == CAM_ZOOM)
		{
			g_fViewDistance += (y - g_yClick);
			//printf("g_fViewDistance: %f\n",g_fViewDistance);

			//if (g_fViewDistance < VIEWING_DISTANCE_MIN)
			//g_fViewDistance = VIEWING_DISTANCE_MIN;
		}
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

		g_xClick = x;
		g_yClick = y;

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
	gluPerspective(65.0, (float)g_Width / g_Height, g_nearPlane, g_farPlane);
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

	//glutInitDisplayMode(GLUT_RGB);

	// create and set up a window 
	glutCreateWindow("SfM!");
	InitGraphics();
	glutDisplayFunc(display);
	glutKeyboardFunc(Keyboard);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutMotionFunc(MouseMotion);
	//glutIdleFunc (AnimateScene);

	glutMainLoop();

}
int visualizationDriver(char *Path, int nViews, int StartTime, int StopTime, bool Color, bool Pose, bool ThreeD)
{
	nviews = nViews;
	hasColor = Color, drawPose = Pose, ThreeDRecon = ThreeD;

	if (ThreeDRecon)
	{
		Scale = Scale / 5;
		minTime = StartTime, maxTime = StopTime;
	}

	if (StartTime == -1)
		drawPose = false;

	VisualizationManager g_vis;
	if (!ThreeDRecon)
		ReadCurrentSfmGL(Path, hasColor);
	else
		ReadCurrent3DGL(Path, hasColor, timeID);

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
void ReadCurrentSfmGL(char *path, bool isColor)
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
	if (isColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/3dGL.xyz", path); fp = fopen(Fname, "r");
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (isColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}

		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	return;
}
void ReadCurrent3DGL(char *path, bool isColor, int timeID)
{
	char Fname[200];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (isColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/3dGL_%d.xyz", path, timeID); FILE *fp = fopen(Fname, "r");
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (isColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}

		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

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










