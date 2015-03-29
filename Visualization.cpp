#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533
#define MAX_NUM_CAM 1000

char *Path;
static bool g_bButton1Down = false;
static GLfloat g_ratio;
static GLfloat g_fViewDistance = 3000 * VIEWING_DISTANCE_MIN;
static GLfloat g_nearPlane = 1.0, g_farPlane = 30000;
float g_coordAxisLength = 300.f;
static int g_Width = 1280, g_Height = 1024;
static int g_xClick = 0, g_yClick = 0;
static int g_mouseYRotate = 0, g_mouseXRotate = 0;
static int g_mouseYPan = 0, g_mouseXPan = 0;

static enum cam_mode { CAM_DEFAULT = 0, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;

static float g_lightPos[4] = { 0, 0, -3000, 0 };  // Position of light
static float g_lightPos2[4] = { 0, 0, 3000, 0 };  // Position of light
static float g_lightBright[4] = { 1, 1, 1, 1 };  // Position of light

GLfloat Red[3] = { 1, 0, 0 }, Green[3] = { 0, 1, 0 }, Blue[3] = { 0, 0, 1 };
GLfloat PointsCentroid[3], ViewingLoc[3];
float InitLocFromCentroid[3];
GLfloat CameraSize = 100.0f, pointSize = 2.f, normalSize = 20.f, arrowThickness = .1f;
vector<int> PickedPoints;
vector<int> PickedTraject;
vector<int> PickCams;
VisualizationManager g_vis;

int nviews = 10, timeID = 0, otimeID = 0, otimeID2 = 0, maxTime, minTime;
bool drawPointColor = false, drawPatchNormal = false;
bool drawCorpusPoints = true, drawCorpusCameras = true, drawTimeVaryingCorpusPoints = false;
bool drawTimeVaryingCameraPose = false, drawTimeVarying3DPointsTraject = false, drawCameraTraject = false;
bool FullTrajectoryMode = false;
static bool ReCenterNeeded = false, bFullsreen = false, showGroundPlane = false, showAxis = false, SaveScreen = false;
static bool showInit3D = true, showFinal3D = false;


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
	if (pickedID >= MAX_NUM_CAM && pickedID < g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size()+MAX_NUM_CAM)// pick points
	{
		pickedID -= MAX_NUM_CAM;

		if (ReCenterNeeded)
		{
			printf("New center picked: %d\n", pickedID);
			PointsCentroid[0] = g_vis.CorpusPointPosition[pickedID].x, PointsCentroid[1] = g_vis.CorpusPointPosition[pickedID].y, PointsCentroid[2] = g_vis.CorpusPointPosition[pickedID].z;
			ReCenterNeeded = false;
			g_mouseXPan = 0.0f, g_mouseYPan = 0.0f;
		}
		else
		{
			printf("Picked %d of (%.3f %.3f %.3f) \n", pickedID, g_vis.CorpusPointPosition[pickedID].x, g_vis.CorpusPointPosition[pickedID].y, g_vis.CorpusPointPosition[pickedID].z);

			bool already = false;
			for (int ii = 0; ii < PickedPoints.size(); ii++)
				if (pickedID == PickedPoints[ii])
				{
					already = true; break;
				}

			if (!already)
				PickedPoints.push_back(pickedID);
		}
	}
	else if (pickedID >= g_vis.CorpusPointPosition.size() + MAX_NUM_CAM)//if(pickedID<MAX_NUM_CAM)  //which means camera
	{
		pickedID -= g_vis.CorpusPointPosition.size() + MAX_NUM_CAM;
		if (pickedID<0 || pickedID>g_vis.Track3DLength.size())
			return;

		printf("Pick trajectory # %d of length %d\n", pickedID, g_vis.Track3DLength[pickedID]);

		bool already = false;
		for (int ii = 0; ii < PickedTraject.size(); ii++)
			if (pickedID == PickedTraject[ii])
			{
				already = true; break;
			}

		if (!already)
			PickedTraject.push_back(pickedID);
	}
}
void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:             // ESCAPE key
		exit(0);
		break;
	case 'f':
		bFullsreen = !bFullsreen;
		if (bFullsreen)
			glutFullScreen();
		break;
	case 'c':
		printf("Current cameraSize: %f. Please enter the new size: ", CameraSize);
		cin >> CameraSize;
		printf("New cameraSize: %f\n", CameraSize);
		break;
	case 'p':
		printf("Current pointSize: %f. Please enter the new size: ", pointSize);
		cin >> pointSize;
		printf("New pointSize: %f\n", pointSize);
		break;
	case 'g':
		printf("Toggle ground plane display\n ");
		showGroundPlane = !showGroundPlane;
		break;
	case 't':
		printf("Toggle 3D point trajectory display\n ");
		drawTimeVarying3DPointsTraject = !drawTimeVarying3DPointsTraject;
		break;
	case 'T':
		printf("Toggle Camera trajectory display\n ");
		drawCameraTraject = !drawCameraTraject;
		break;
	case '1':
		printf("Toggle corpus points display\n ");
		drawCorpusPoints = !drawCorpusPoints;
		break;
	case '2':
		printf("Toggle corpus cameras display\n ");
		drawCorpusCameras = !drawCorpusCameras;
		break;
	case '3':
		printf("Toggle corpus trajectory display\n ");
		drawTimeVaryingCorpusPoints = !drawTimeVaryingCorpusPoints;
		break;
	case 'A':
		printf("Toggle axis display\n ");
		showAxis = !showAxis;
		break;
	case 's':
		printf("Toggle screen saving\n ");
		SaveScreen = !SaveScreen;
		break;
	case 'a':
		g_mouseXRotate += 5;
		break;
	case 'd':
		g_mouseXRotate -= 5;
		break;
	case 'i':
		showInit3D = !showInit3D;
		break;
	case 'o':
		showFinal3D = !showFinal3D;
		break;
	}
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		timeID = maxTime;
		printf("Current time: %d\n", timeID);
		break;
	case GLUT_KEY_DOWN:
		timeID = 0;
		printf("Current time: %d\n", timeID);
		break;
	case GLUT_KEY_LEFT:
		timeID--;
		if (timeID < minTime)
			timeID = minTime;
		printf("Current time: %d\n", timeID);
		break;
	case GLUT_KEY_RIGHT:
		timeID++;
		if (timeID > maxTime)
			timeID = maxTime;
		printf("Current time: %d\n", timeID);
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
			PickedPoints.clear(), PickCams.clear(), PickedTraject.clear();
	}
	else if (button == GLUT_MIDDLE_BUTTON)
	{
		g_xClick = x;
		g_yClick = y;
		g_bButton1Down = true;
		g_camMode = CAM_ZOOM;
	}
}
void MouseMotion(int x, int y)
{
	if (g_bButton1Down)
	{
		if (g_camMode == CAM_ZOOM)
			g_fViewDistance += 5.0f*(y - g_yClick);
		else if (g_camMode == CAM_ROTATE)
		{
			g_mouseXRotate += (x - g_xClick);
			g_mouseYRotate -= (y - g_yClick);
		}
		else if (g_camMode == CAM_PAN)
		{
			g_mouseXPan -= (x - g_xClick);
			g_mouseYPan -= (y - g_yClick);
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
	gluPerspective(60.0, g_ratio, g_nearPlane, g_farPlane);
	glMatrixMode(GL_MODELVIEW);
}

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

	// we also has to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
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
void RenderGroundPlane()
{
	int gridNum = 10;
	double width = 1000;
	double halfWidth = width / 2;
	Point3f origin(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]);
	Point3f axis1 = Point3f(1.0f, 0.0f, 0.0f)* width;
	Point3f axis2 = Point3f(0.0f, 0.0f, 1.0f) * width;
	glBegin(GL_QUADS);
	for (int y = -gridNum; y <= gridNum; ++y)
		for (int x = -gridNum; x <= gridNum; ++x)
		{
			if ((x + y) % 2 == 0)
				continue;
			else
				glColor4f(0.7, 0.7, 0.7, 0.9);

			Point3f p1 = origin + axis1*x + axis2*y;
			Point3f p2 = p1 + axis1;
			Point3f p3 = p1 + axis2;
			Point3f p4 = p1 + axis1 + axis2;

			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p1.x, p1.y, p1.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p2.x, p2.y, p2.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p4.x, p4.y, p4.z);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(p3.x, p3.y, p3.z);
		}
	glEnd();

}
void RenderSkeleton(vector<Point3d> pt3D, GLfloat *color)
{
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	int i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 2; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 4; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 5; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 6; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 3; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 2; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 7; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 8; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 9; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 1; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 12; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 10; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	i = 11; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();
}
void RenderSkeleton2(vector<Point3d> pt3D, GLfloat *color)
{
	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	int i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 1; i < 6; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 6; i < 11; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 0; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 11; i < 17; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 14; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 24; i < 31; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();

	glPushMatrix();
	glBegin(GL_LINE_STRIP);
	i = 14; glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	for (i = 17; i < 24; i++)
		glVertex3f(pt3D[i].x - PointsCentroid[0], pt3D[i].y - PointsCentroid[1], pt3D[i].z - PointsCentroid[2]);			glColor3fv(color);
	glEnd();
	glPopMatrix();
}
void RenderObjects()
{
	int CountFilesLoaded = 0;
	
	if (otimeID != timeID && drawTimeVarying3DPointsTraject)
	{
		ReadCurrentTrajectory("C:/temp", timeID);

		//ReCenterNeeded = true;
		//if (ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, ReCenterNeeded))
		//CountFilesLoaded++;

		otimeID = timeID;
	}

	//Draw not picked corpus points
	if (drawCorpusPoints)
	{
		for (unsigned int i = 0; i < g_vis.CorpusPointPosition.size(); ++i)
		{
			bool picked = false;
			for (unsigned int j = 0; j < PickedPoints.size(); j++)
				if (i == PickedPoints[j])
					picked = true;
			if (picked)
				continue;

			glLoadName(i + MAX_NUM_CAM);//for picking purpose
			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition[i].x - PointsCentroid[0], g_vis.CorpusPointPosition[i].y - PointsCentroid[1], g_vis.CorpusPointPosition[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.CorpusPointColor[i].x, g_vis.CorpusPointColor[i].y, g_vis.CorpusPointColor[i].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();

			if (drawPatchNormal)
			{
				glColor4f(0, 1, 0, 0.5f);
				Point3d newHeadPt = normalSize *g_vis.PointNormal[i] + g_vis.CorpusPointPosition[i];
				glPushMatrix();
				Arrow(g_vis.CorpusPointPosition[i].x - PointsCentroid[0], g_vis.CorpusPointPosition[i].y - PointsCentroid[1], g_vis.CorpusPointPosition[i].z - PointsCentroid[2],
					newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
				glPopMatrix();
			}
		}

		for (unsigned int i = 0; i < g_vis.CorpusPointPosition2.size(); ++i)
		{
			bool picked = false;
			for (unsigned int j = 0; j < PickedPoints.size(); j++)
				if (i == PickedPoints[j])
					picked = true;
			if (picked)
				continue;

			glLoadName(i + g_vis.CorpusPointPosition.size()+ MAX_NUM_CAM);//for picking purpose
			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition2[i].x - PointsCentroid[0], g_vis.CorpusPointPosition2[i].y - PointsCentroid[1], g_vis.CorpusPointPosition2[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.CorpusPointColor2[i].x, g_vis.CorpusPointColor2[i].y, g_vis.CorpusPointColor2[i].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();

			if (drawPatchNormal)
			{
				glColor4f(0, 1, 0, 0.5f);
				Point3d newHeadPt = normalSize *g_vis.PointNormal[i] + g_vis.CorpusPointPosition[i];
				glPushMatrix();
				Arrow(g_vis.CorpusPointPosition2[i].x - PointsCentroid[0], g_vis.CorpusPointPosition2[i].y - PointsCentroid[1], g_vis.CorpusPointPosition2[i].z - PointsCentroid[2],
					newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
				glPopMatrix();
			}
		}
		//RenderSkeleton2(g_vis.CorpusPointPosition, Red);

		//Draw picked 3D points red
		for (unsigned int i = 0; i < PickedPoints.size(); i++)
		{
			int id = PickedPoints[i];
			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition[id].x - PointsCentroid[0], g_vis.CorpusPointPosition[id].y - PointsCentroid[1], g_vis.CorpusPointPosition[id].z - PointsCentroid[2]);
			glColor3fv(Red);
			glutSolidSphere(pointSize*10, 4, 4);
			glPopMatrix();

			if (drawPatchNormal)
			{
				glColor4f(0, 1, 0, 0.5f);
				Point3d newHeadPt = normalSize *g_vis.PointNormal[id] + g_vis.CorpusPointPosition[id];
				glPushMatrix();
				Arrow(g_vis.CorpusPointPosition[id].x - PointsCentroid[0], g_vis.CorpusPointPosition[id].y - PointsCentroid[1], g_vis.CorpusPointPosition[id].z - PointsCentroid[2],
					newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
				glPopMatrix();
			}
		}
	}

	if (drawTimeVaryingCorpusPoints)//if the corpus 3D is some how kind of time varying
	{
		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < min(timeID, (int)(g_vis.CorpusPointPosition.size())); ++i)
		{
			glVertex3f(g_vis.CorpusPointPosition[i].x - PointsCentroid[0], g_vis.CorpusPointPosition[i].y - PointsCentroid[1], g_vis.CorpusPointPosition[i].z - PointsCentroid[2]);
			glColor3f(g_vis.CorpusPointColor[i].x, g_vis.CorpusPointColor[i].y, g_vis.CorpusPointColor[i].z);
		}
		glEnd();
		glPopMatrix();

		for (int i = 0; i < min(timeID, (int)(g_vis.CorpusPointPosition.size())); ++i)
		{
			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition[i].x - PointsCentroid[0], g_vis.CorpusPointPosition[i].y - PointsCentroid[1], g_vis.CorpusPointPosition[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.CorpusPointColor[i].x, g_vis.CorpusPointColor[i].y, g_vis.CorpusPointColor[i].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}


		glPushMatrix();
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < min(timeID, (int)(g_vis.CorpusPointPosition2.size())); ++i)
		{
			glVertex3f(g_vis.CorpusPointPosition2[i].x - PointsCentroid[0], g_vis.CorpusPointPosition2[i].y - PointsCentroid[1], g_vis.CorpusPointPosition2[i].z - PointsCentroid[2]);
			glColor3f(g_vis.CorpusPointColor2[i].x, g_vis.CorpusPointColor2[i].y, g_vis.CorpusPointColor2[i].z);
		}
		glEnd();
		glPopMatrix();

		for (int i = 0; i < min(timeID, (int)(g_vis.CorpusPointPosition2.size())); ++i)
		{
			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition2[i].x - PointsCentroid[0], g_vis.CorpusPointPosition2[i].y - PointsCentroid[1], g_vis.CorpusPointPosition2[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.CorpusPointColor2[i].x, g_vis.CorpusPointColor2[i].y, g_vis.CorpusPointColor2[i].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}
	}

	//draw 3d trajectories
	if (drawTimeVarying3DPointsTraject)
	{
		//Concat 3D points from 3D points at individual time instance
		for (int pid = 0; pid < g_vis.PointPosition.size(); pid++)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			for (int fid = 0; fid <= min(maxTime, timeID - minTime); fid++)
			{
				glVertex3f(g_vis.catPointPosition[pid][fid].x - PointsCentroid[0], g_vis.catPointPosition[pid][fid].y - PointsCentroid[1], g_vis.catPointPosition[pid][fid].z - PointsCentroid[2]);
				glColor3fv(Red);
			}
			glEnd();
			glPopMatrix();
		}

		//somehow similar to the full trajectory mode
		for (int ii = 0; ii < g_vis.Traject3D.size(); ii++)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			for (int fid = 0; fid < g_vis.Track3DLength[ii]; fid++)
			{
				glVertex3f(g_vis.Traject3D[ii][fid].WC.x - PointsCentroid[0], g_vis.Traject3D[ii][fid].WC.y - PointsCentroid[1], g_vis.Traject3D[ii][fid].WC.z - PointsCentroid[2]);
				glColor3fv(Red);
			}
			glEnd();
			glPopMatrix();
		}
	}

	//draw Corpus camera 
	if (drawCorpusCameras)
	{
		for (int j = 0; j < g_vis.glCorpusCameraInfo.size(); j++)
		{
			float CameraColor[3] = { 1, 0, 0 };
			float* centerPt = g_vis.glCorpusCameraInfo[j].camCenter;
			GLfloat* R = g_vis.glCorpusCameraInfo[j].Rgl;

			glLoadName(j);//for picking purpose
			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);
			DrawCamera();
			glPopMatrix();
		}
	}

	//draw time varying camera pose
	if (drawTimeVaryingCameraPose)
	{
		for (int j = 0; j < nviews; j++)
		{
			float CameraColor[3] = { 1, 0, 0 };
			if (g_vis.glCameraPoseInfo[j].size() > 0)
			{
				if (drawCameraTraject)
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					int maxtime = min(timeID, (int)g_vis.glCameraPoseInfo[j].size());
					for (unsigned int i = 0; i <= maxtime; ++i)
					{
						float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
						glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
						glColor3f(CameraColor[0], CameraColor[1], CameraColor[2]);
					}
					glEnd();
				}

				for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
				{
					if (i != timeID)
						continue;

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

	//gluLookAt(-100.92273509 - PointsCentroid[0], -165.08951125 - PointsCentroid[1], 1485.49577164 - PointsCentroid[2], 0, 0, 0, 0, 1, 0);
	glTranslatef(-g_mouseXPan, -g_mouseYPan, -g_fViewDistance);
	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	// Set up the stationary light
	glLightfv(GL_LIGHT0, GL_POSITION, g_lightPos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, g_lightBright);
	glLightfv(GL_LIGHT1, GL_POSITION, g_lightPos2);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, g_lightBright);

	RenderObjects();
	if (showAxis)
		Draw_Axes();
	if (showGroundPlane)
		RenderGroundPlane();
	glutSwapBuffers();

	if (SaveScreen && otimeID2 != timeID)
	{
		char Fname[200];
		if (showInit3D)
			sprintf(Fname, "%s/B/%d.png", Path, timeID);
		else if (showFinal3D)
			sprintf(Fname, "%s/A/%d.png", Path, timeID);
		screenShot(Fname, 1024, 768, true);
		otimeID2 = timeID;
	}
}
void visualization()
{
	char *myargv[1];
	int myargc = 1;
	myargv[0] = _strdup("SfM");
	glutInit(&myargc, myargv);

	// setup the size, position, and display mode for new windows 
	glutInitWindowSize(1200, 900);
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
int visualizationDriver(char *inPath, int nViews, int StartTime, int StopTime, bool hasColor, bool hasPatchNormal, bool hasTimeVaryingCameraPose, bool hasTimeVarying3DPoints, bool hasFullTrajectory, int CurrentTime)
{
	Path = inPath;
	nviews = nViews;
	drawPointColor = hasColor, drawPatchNormal = hasPatchNormal;
	drawTimeVaryingCameraPose = hasTimeVaryingCameraPose, drawTimeVarying3DPointsTraject = hasTimeVarying3DPoints;
	FullTrajectoryMode = hasFullTrajectory;

	minTime = StartTime, maxTime = StopTime, timeID = CurrentTime;

	VisualizationManager g_vis;

	ReadCurrentSfmGL(Path, drawPointColor, drawPatchNormal);
	ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, false);

	//Abitary trajectory input
	ReadCurrentTrajectory(Path, 0);

	if (drawTimeVaryingCameraPose)
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

	bool hasColor = false;
	if (AllColor != NULL)
		hasColor = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasColor)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, vector<Point3d>All3D, vector<Point3i>AllColor)
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

	bool hasColor = false;
	if (AllColor.size() != 0)
		hasColor = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < All3D.size(); ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasColor)
			fprintf(fp, "%d %d %d\n", AllColor[ii].x, AllColor[ii].y, AllColor[ii].z);
		else
			fprintf(fp, "\n");
	}
	fclose(fp);

	return;
}
void ReadCurrentSfmGL(char *path, bool drawPointColor, bool drawPatchNormal)
{
	char Fname[200];
	int viewID;

	CamInfo temp;
	sprintf(Fname, "%s/Corpus/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		while (fscanf(fp, "%d: ", &viewID) != EOF)
		{
			for (int jj = 0; jj < 16; jj++)
				fscanf(fp, "%f ", &temp.Rgl[jj]);
			for (int jj = 0; jj < 3; jj++)
				fscanf(fp, "%f ", &temp.camCenter[jj]);

			g_vis.glCorpusCameraInfo.push_back(temp);
		}
		fclose(fp);
	}
	else
	{
		printf("Cannot load %s. Try with %s/BA_Camera_AllParams_after.txt ...", Fname, path);

		Corpus CorpusData;
		sprintf(Fname, "%s/BA_Camera_AllParams_after.txt", path);
		if (loadBundleAdjustedNVMResults(Fname, CorpusData))
		{
			for (int ii = 0; ii < CorpusData.nCamera; ii++)
			{
				for (int jj = 0; jj < 16; jj++)
					temp.Rgl[jj] = CorpusData.camera[ii].Rgl[jj];
				for (int jj = 0; jj < 3; jj++)
					temp.camCenter[jj] = CorpusData.camera[ii].camCenter[jj];
				g_vis.glCorpusCameraInfo.push_back(temp);
			}
			printf("succeeded.\n");
		}
		else
		{
			printf("Cannot load %s\n", Fname);
			abort();
		}
	}

	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.CorpusPointColor.clear(), g_vis.CorpusPointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	/*sprintf(Fname, "%s/Corpus/3dGL.xyz", path); fp = fopen(Fname, "r");
	if (fp == NULL)
	{
	printf("Cannot load %s\n", Fname);
	abort();
	}
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
	if (drawPointColor)
	{
	fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
	fColor.x = 1.0*iColor.x / 255;
	fColor.y = 1.0*iColor.y / 255;
	fColor.z = 1.0*iColor.z / 255;
	g_vis.CorpusPointColor.push_back(fColor);
	}
	PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
	g_vis.CorpusPointPosition.push_back(t3d);
	}
	fclose(fp);*/


	Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
	for (unsigned int i = 0; i <= 255; i++)
		colorMapSource.at<uchar>(i, 0) = i;
	Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);
	Point3f tempColor;

	sprintf(Fname, "%s/C0_0.txt", path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int ii;
		while (fscanf(fp, "%d %f %f %f ", &ii, &t3d.x, &t3d.y, &t3d.z) != EOF)
		{
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
			g_vis.CorpusPointPosition.push_back(t3d);
		}
		fclose(fp);

		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			double colorIdx = 1.0*ii / g_vis.CorpusPointPosition.size() * 255.0; colorIdx = min(255.0, colorIdx);
			tempColor.z = colorMap.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
			tempColor.y = colorMap.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
			tempColor.x = colorMap.at<Vec3b>(colorIdx, 0)[2] / 255.0;	//red
			g_vis.CorpusPointColor.push_back(tempColor);
		}
	}
	else
		printf("Cannot load %s\n", Fname);

	Mat colorMap2; applyColorMap(colorMapSource, colorMap2, COLORMAP_AUTUMN);
	sprintf(Fname, "%s/C1_0.txt", path); fp = fopen(Fname, "r");
	if (fp != NULL)
	{
		int ii;
		while (fscanf(fp, "%d %f %f %f ", &ii, &t3d.x, &t3d.y, &t3d.z) != EOF)
			g_vis.CorpusPointPosition2.push_back(t3d);
		fclose(fp);

		for (int ii = 0; ii < g_vis.CorpusPointPosition2.size(); ii++)
		{
			double colorIdx = 1.0*ii / g_vis.CorpusPointPosition2.size() * 255.0; colorIdx = min(255.0, colorIdx);
			tempColor.z = colorMap2.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
			tempColor.y = colorMap2.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
			tempColor.x = colorMap2.at<Vec3b>(colorIdx, 0)[2] / 255.0;	//red
			g_vis.CorpusPointColor2.push_back(tempColor);
		}
	}
	else
		printf("Cannot load %s\n", Fname);

	PointsCentroid[0] /= g_vis.CorpusPointPosition.size();
	PointsCentroid[1] /= g_vis.CorpusPointPosition.size();
	PointsCentroid[2] /= g_vis.CorpusPointPosition.size();


	return;
}
bool ReadCurrent3DGL(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[200];
	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	if (setCoordinate)
		PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	//sprintf(Fname, "%s/GT_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	sprintf(Fname, "%s/3DPoints/%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return false;
	}
	while (fscanf(fp, "%f %f %f ", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPatchNormal)
		{
			fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
			g_vis.PointNormal.push_back(n3d);
		}
		if (drawPointColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor.push_back(fColor);
		}

		if (setCoordinate)
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition.size(), PointsCentroid[1] /= g_vis.PointPosition.size(), PointsCentroid[2] /= g_vis.PointPosition.size();

	//Concatenate points in case the trajectory mode is used
	if (!FullTrajectoryMode) //in efficient of there are many points
	{
		if (g_vis.catPointPosition == NULL)
			g_vis.catPointPosition = new vector<Point3d>[g_vis.PointPosition.size()];
		if (timeID - minTime == g_vis.catPointPosition[0].size())
			for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
				g_vis.catPointPosition[ii].push_back(g_vis.PointPosition[ii]);
	}

	return true;
}
void ReadCurrentTrajectory(char *path, int timeID)
{
	char Fname[200];
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	g_vis.Track3DLength.clear(), g_vis.Traject3D.clear();

	//sprintf(Fname, "%s/3DTracks/0.txt", path); FILE *fp = fopen(Fname, "r");
	sprintf(Fname, "C:/temp/Off_%d.txt", timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	int ntracks = 1, currentTime = 1, nframes = 2316;
	//fscanf(fp, "%s %d", Fname, &ntracks);
	for (int ii = 0; ii < ntracks; ii++)
	{
		//fscanf(fp, "%d", &nframes);
		Trajectory3D *track3D = new Trajectory3D[nframes];
		for (int jj = 0; jj < nframes; jj++)
		{
			track3D[jj].timeID = currentTime + jj;
			fscanf(fp, "%lf %lf %lf ", &track3D[jj].WC.x, &track3D[jj].WC.y, &track3D[jj].WC.z);
		}

		g_vis.Track3DLength.push_back(nframes);
		g_vis.Traject3D.push_back(track3D);
	}
	fclose(fp);

	return;
}
void SaveCurrentPosesGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, int timeID)
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
		sprintf(Fname, "%s/Calib/PinfoGL_%d.txt", path, ii);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			sprintf("Cannot load %s\n", Fname);
			continue;
		}
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
int screenShot(char *Fname, int width, int height, bool color)
{
	int ii, jj, kk;

	unsigned char *data = new unsigned char[width*height * 4];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	for (kk = 0; kk < 3; kk++)
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				cvImg->imageData[3 * ii + 3 * jj*width + kk] = data[3 * ii + 3 * (height - 1 - jj)*width + kk];

	if (color)
		cvSaveImage(Fname, cvImg);
	else
	{
		IplImage *cvImgGray = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		cvCvtColor(cvImg, cvImgGray, CV_BGR2GRAY);
		cvSaveImage(Fname, cvImgGray);
		cvReleaseImage(&cvImgGray);
	}

	cvReleaseImage(&cvImg);

	delete[]data;
	return 0;
}










