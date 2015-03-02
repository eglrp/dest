#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533
#define MAX_NUM_CAM 1000

char *Path;
static bool g_bButton1Down = false;
static GLfloat g_ratio;
static GLfloat g_fViewDistance = 500 * VIEWING_DISTANCE_MIN;
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
GLfloat Scale = 0.025f, CameraSize = 20.0f, pointSize = 2.f, normalSize = 20.f, arrowThickness = .1f;
vector<int> PickedPoints;
vector<int> PickedTraject;
vector<int> PickCams;
VisualizationManager g_vis;

int nviews = 10, timeID = 0, otimeID = 0, otimeID2 = 0, maxTime, minTime;
bool drawCameraPose = false, drawPointColor = false, drawPatchNormal = false;
bool drawPt3DTraject = false, drawCameraTraject = false, drawPt3D = false;
static bool ReCenterNeeded = false, bFullsreen = false, showGroundPlane = false, SaveScreen = false;
static bool showInit3D = true, showFinal3D = true;

void Draw_Axes(void)
{
	glPushMatrix();

	glLineWidth(2.0);

	glBegin(GL_LINES);
	glColor3f(1, 0, 0); // X axis is red.
	glVertex3f(0, 0, 0);
	glVertex3f(g_coordAxisLength*Scale, 0, 0);
	glColor3f(0, 1, 0); // Y axis is green.
	glVertex3f(0, 0, 0);
	glVertex3f(0, g_coordAxisLength*Scale, 0);
	glColor3f(0, 0, 1); // z axis is blue.
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, g_coordAxisLength*Scale);
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
	double width = 50;
	double halfWidth = width / 2;
	//Point3f origin(0.0f, 0.0f, 0.0f);
	Point3f origin(-PointsCentroid[0], -PointsCentroid[1], -PointsCentroid[2]);
	Point3f axis1 = Point3f(1.0f, 0.0f, 0.0f)* width;
	Point3f axis2 = Point3f(0.0f, 0.0f, 1.0f) * width;
	glBegin(GL_QUADS);
	for (int y = -gridNum; y <= 5 * gridNum; ++y)
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
void RenderSkeleton(vector<Point3d> pt3D,GLfloat *color)
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
void RenderObjects()
{
	int CountFilesLoaded = 0;
	ReCenterNeeded = true;
	if (otimeID != timeID)
	{
		if (ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, ReCenterNeeded))
			CountFilesLoaded++;

		if (ReadCurrent3DGL2(Path, drawPointColor, drawPatchNormal, timeID, false))
			CountFilesLoaded++;

		if (ReadCurrent3DGL3(Path, drawPointColor, drawPatchNormal, timeID, false))
			CountFilesLoaded++;

		otimeID = timeID;
		if (CountFilesLoaded > 0)
			printf("Loaded frame %d\n", timeID);
		else
			return;
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
		if (drawPointColor)
			glColor3f(g_vis.PointColor[i].x, g_vis.PointColor[i].y, g_vis.PointColor[i].z);
		else
			glColor3fv(Red);
		glutSolidSphere(pointSize, 4, 4);
		glPopMatrix();

		if (drawPatchNormal)
		{
			glColor4f(0, 1, 0, 0.5f);
			Point3d newHeadPt = normalSize *g_vis.PointNormal[i] + g_vis.PointPosition[i];
			glPushMatrix();
			Arrow(g_vis.PointPosition[i].x - PointsCentroid[0], g_vis.PointPosition[i].y - PointsCentroid[1], g_vis.PointPosition[i].z - PointsCentroid[2],
				newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
			glPopMatrix();
		}
	}
	RenderSkeleton(g_vis.PointPosition, Red);

	if (showInit3D)
	{
		for (unsigned int i = 0; i < g_vis.PointPosition2.size(); ++i)
		{
			glPushMatrix();
			glTranslatef(g_vis.PointPosition2[i].x - PointsCentroid[0], g_vis.PointPosition2[i].y - PointsCentroid[1], g_vis.PointPosition2[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.PointColor2[i].x, g_vis.PointColor2[i].y, g_vis.PointColor2[i].z);
			else
				glColor3fv(Green);

			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}

		//RenderSkeleton(g_vis.PointPosition2, Green);
	}
	else
		g_vis.PointPosition2.clear();

	if (showFinal3D)
	{
		for (unsigned int i = 0; i < g_vis.PointPosition3.size(); ++i)
		{
			glPushMatrix();
			glTranslatef(g_vis.PointPosition3[i].x - PointsCentroid[0], g_vis.PointPosition3[i].y - PointsCentroid[1], g_vis.PointPosition3[i].z - PointsCentroid[2]);
			if (drawPointColor)
				glColor3f(g_vis.PointPosition3[i].x, g_vis.PointPosition3[i].y, g_vis.PointPosition3[i].z);
			else
				glColor3fv(Blue);

			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}
		//RenderSkeleton(g_vis.PointPosition3, Blue);
	}
	else
		g_vis.PointPosition3.clear();

	//Draw picked 3D points red
	for (unsigned int i = 0; i < PickedPoints.size(); i++)
	{
		int id = PickedPoints[i];
		glLoadName(id + MAX_NUM_CAM);//for picking purpose
		glPushMatrix();
		glTranslatef(g_vis.PointPosition[id].x - PointsCentroid[0], g_vis.PointPosition[id].y - PointsCentroid[1], g_vis.PointPosition[id].z - PointsCentroid[2]);
		glColor3fv(Red);
		glutSolidSphere(pointSize, 4, 4);
		glPopMatrix();

		if (drawPatchNormal)
		{
			glColor4f(0, 1, 0, 0.5f);
			Point3d newHeadPt = normalSize *g_vis.PointNormal[id] + g_vis.PointPosition[id];
			glPushMatrix();
			Arrow(g_vis.PointPosition[id].x - PointsCentroid[0], g_vis.PointPosition[id].y - PointsCentroid[1], g_vis.PointPosition[id].z - PointsCentroid[2],
				newHeadPt.x - PointsCentroid[0], newHeadPt.y - PointsCentroid[1], newHeadPt.z - PointsCentroid[2], arrowThickness);
			glPopMatrix();
		}
	}

	//draw 3d trajectories
	if (drawPt3DTraject)
	{
		for (int pid = 0; pid < g_vis.PointPosition3.size(); pid++)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			for (int fid = 0; fid <= timeID - minTime; fid++)
			{
				glVertex3f(g_vis.catPointPosition[pid][fid].x - PointsCentroid[0], g_vis.catPointPosition[pid][fid].y - PointsCentroid[1], g_vis.catPointPosition[pid][fid].z - PointsCentroid[2]);
				glColor3fv(Red);
			}
			glEnd();
			glPopMatrix();
		}

		for (int pid = 0; pid < g_vis.PointPosition3.size(); pid++)
		{
			glPushMatrix();
			glBegin(GL_LINE_STRIP);
			for (int fid = 0; fid <= timeID - minTime; fid++)
			{
				glVertex3f(g_vis.catPointPosition3[pid][fid].x - PointsCentroid[0], g_vis.catPointPosition3[pid][fid].y - PointsCentroid[1], g_vis.catPointPosition3[pid][fid].z - PointsCentroid[2]);
				glColor3fv(Blue);
			}
			glEnd();
			glPopMatrix();
		}
	}

	//draw camera 
	if (timeID == -1)
	{
		for (int j = 0; j < g_vis.glCameraInfo.size(); j++)
		{
			float CameraColor[3] = { 1, 0, 0 };
			float* centerPt = g_vis.glCameraInfo[j].camCenter;
			GLfloat* R = g_vis.glCameraInfo[j].Rgl;

			glLoadName(j);//for picking purpose
			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);
			DrawCamera();
			glPopMatrix();
		}
	}

	if (drawCameraPose)
	{
		for (int j = 0; j < nviews; j++)
		{
			float CameraColor[3] = { 1, 0, 0 };
			if (g_vis.glCameraPoseInfo[j].size() > 0)
			{
				//Draw camere trajectory
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
	//Draw_Axes();
	if (showGroundPlane)
		RenderGroundPlane();
	glutSwapBuffers();

	if (SaveScreen && otimeID2 != timeID)
	{
		char Fname[200]; sprintf(Fname, "C:/temp/Img_%d.png", timeID);
		screenShot(Fname, 1024, 768, true);
		otimeID2 = timeID;
	}
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
	if (pickedID >= MAX_NUM_CAM && pickedID < g_vis.PointPosition.size() + MAX_NUM_CAM)// pick points
	{
		pickedID -= MAX_NUM_CAM;

		if (ReCenterNeeded)
		{
			printf("New center picked: %d\n", pickedID);
			PointsCentroid[0] = g_vis.PointPosition[pickedID].x, PointsCentroid[1] = g_vis.PointPosition[pickedID].y, PointsCentroid[2] = g_vis.PointPosition[pickedID].z;
			ReCenterNeeded = false;
			g_mouseXPan = 0.0f, g_mouseYPan = 0.0f;
		}
		else
		{
			printf("Picked %d of (%.3f %.3f %.3f) \n", pickedID, g_vis.PointPosition[pickedID].x, g_vis.PointPosition[pickedID].y, g_vis.PointPosition[pickedID].z);

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
	else if (pickedID >= g_vis.PointPosition.size() + MAX_NUM_CAM)//if(pickedID<MAX_NUM_CAM)  //which means camera
	{
		pickedID -= g_vis.PointPosition.size() + MAX_NUM_CAM;
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
		printf("Toggle trajectory display\n ");
		drawCameraTraject = !drawCameraTraject;
		drawPt3DTraject = !drawPt3DTraject;
		break;
	case 's':
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
		//do something here
		break;
	case GLUT_KEY_DOWN:
		//do something here
		break;
	case GLUT_KEY_LEFT:
		timeID--;
		if (timeID < minTime)
			timeID = minTime;

		//if (timeID < 0)
		//	timeID = 0;
		//ReadCurrentTrajectory2(Path, timeID);
		break;
	case GLUT_KEY_RIGHT:
		timeID++;
		if (timeID > maxTime)
			timeID = maxTime;

		//if (timeID > 500)
		//	timeID = 500;
		//ReadCurrentTrajectory2(Path, timeID);
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

int visualizationDriver(char *inPath, int nViews, int StartTime, int StopTime, bool hasPointColor, bool hasPatchNormal, bool hasCameraPose, bool has3DPoints, bool Traject3DfromOneTime, int CurrentTime)
{
	nviews = nViews;
	drawPointColor = hasPointColor, drawPatchNormal = hasPatchNormal, drawCameraPose = hasCameraPose, drawPt3D = has3DPoints;
	Path = inPath;
	if (drawPt3D)
		minTime = StartTime, maxTime = StopTime;

	timeID = CurrentTime;
	if (StartTime == -1)
		drawCameraPose = false, timeID = -1;

	VisualizationManager g_vis;
	if (!drawPt3D)
		ReadCurrentSfmGL(Path, drawPointColor, drawPatchNormal);
	else
	{
		ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, true);
		ReadCurrent3DGL2(Path, drawPointColor, drawPatchNormal, CurrentTime, false);
		ReadCurrent3DGL3(Path, drawPointColor, drawPatchNormal, CurrentTime, false);
	}

	if (Traject3DfromOneTime)
	{
		drawPt3DTraject = true;
		ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, CurrentTime, true);
	}

	if (drawCameraPose)
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

	bool hasPointColor = false;
	if (AllColor != NULL)
		hasPointColor = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < npts; ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasPointColor)
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

	bool hasPointColor = false;
	if (AllColor.size() != 0)
		hasPointColor = true;

	sprintf(Fname, "%s/3dGL.xyz", path);
	fp = fopen(Fname, "w+");
	for (int ii = 0; ii < All3D.size(); ii++)
	{
		if (abs(All3D[ii].x) + abs(All3D[ii].y) + abs(All3D[ii].z) < 0.001)
			continue;
		fprintf(fp, "%.16f %.16f %.16f ", All3D[ii].x, All3D[ii].y, All3D[ii].z);
		if (hasPointColor)
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
	else
	{
		printf("Cannot load %s\n", Fname);
		abort();
	}

	g_vis.PointPosition.clear(); g_vis.PointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor.clear(), g_vis.PointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/3dGL.xyz", path); fp = fopen(Fname, "r");
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
			g_vis.PointColor.push_back(fColor);
		}
		PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	PointsCentroid[0] /= g_vis.PointPosition.size();
	PointsCentroid[1] /= g_vis.PointPosition.size();
	PointsCentroid[2] /= g_vis.PointPosition.size();

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
	sprintf(Fname, "%s/GT_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
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

	if (g_vis.catPointPosition == NULL)
		g_vis.catPointPosition = new vector<Point3d>[g_vis.PointPosition.size()];
	if (timeID - minTime == g_vis.catPointPosition[0].size())
		for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
			g_vis.catPointPosition[ii].push_back(g_vis.PointPosition[ii]);

	return true;
}
bool ReadCurrent3DGL2(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[200];
	g_vis.PointPosition2.clear(); g_vis.PointPosition2.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor2.clear(), g_vis.PointColor2.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	sprintf(Fname, "%s/B_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return false;
	}
	while (fscanf(fp, "%f %f %f", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPatchNormal)
		{
			fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
			g_vis.PointNormal2.push_back(n3d);
		}
		if (drawPointColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor2.push_back(fColor);
		}

		g_vis.PointPosition2.push_back(t3d);
	}
	fclose(fp);

	return true;
}
bool ReadCurrent3DGL3(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[200];
	g_vis.PointPosition3.clear(); g_vis.PointPosition3.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor3.clear(), g_vis.PointColor3.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	sprintf(Fname, "%s/B2_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return false;
	}
	while (fscanf(fp, "%f %f %f", &t3d.x, &t3d.y, &t3d.z) != EOF)
	{
		if (drawPatchNormal)
		{
			fscanf(fp, "%f %f %f ", &n3d.x, &n3d.y, &n3d.z);
			g_vis.PointNormal3.push_back(n3d);
		}
		if (drawPointColor)
		{
			fscanf(fp, "%d %d %d ", &iColor.x, &iColor.y, &iColor.z);
			fColor.x = 1.0*iColor.x / 255;
			fColor.y = 1.0*iColor.y / 255;
			fColor.z = 1.0*iColor.z / 255;
			g_vis.PointColor3.push_back(fColor);
		}

		g_vis.PointPosition3.push_back(t3d);
	}
	fclose(fp);


	if (g_vis.catPointPosition3 == NULL)
		g_vis.catPointPosition3 = new vector<Point3d>[g_vis.PointPosition3.size()];
	if (timeID - minTime == g_vis.catPointPosition3[0].size())
		for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
			g_vis.catPointPosition3[ii].push_back(g_vis.PointPosition3[ii]);

	return true;
}
void ReadCurrentTrajectory(char *path, int timeID)
{
	char Fname[200];
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	//sprintf(Fname, "%s/Traject3D/Track_1.txt", path); FILE *fp = fopen(Fname, "r");
	sprintf(Fname, "C:/temp/i1.txt"); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		printf("Cannot load %s\n", Fname);
		return;
	}
	int npts = 1, currentTime = 1, ntracks = 1640;
	//fscanf(fp, "%s %d", Fname, &npts);
	for (int ii = 0; ii < npts; ii++)
	{
		//fscanf(fp, "%d %d", &currentTime, &ntracks);
		Trajectory3D *track3D = new Trajectory3D[ntracks];
		double mX = 0.0, mY = 0.0, mZ = 0.0;
		for (int jj = 0; jj < ntracks; jj++)
		{
			track3D[jj].timeID = currentTime + jj;
			fscanf(fp, "%lf %lf %lf ", &track3D[jj].WC.x, &track3D[jj].WC.y, &track3D[jj].WC.z);
			mX += track3D[jj].WC.x, mY += track3D[jj].WC.y, mZ += track3D[jj].WC.z;
		}
		mX = mX / ntracks, mY = mY / ntracks, mZ = mZ / ntracks;
		for (int jj = 0; jj < ntracks; jj++)
		{
			track3D[jj].WC.x = 1.0*(track3D[jj].WC.x - mX) + mX;
			track3D[jj].WC.y = 1.0*(track3D[jj].WC.y - mY) + mY;
			track3D[jj].WC.z = 1.0*(track3D[jj].WC.z - mZ) + mZ;
		}
		g_vis.Track3DLength.push_back(ntracks);
		g_vis.Traject3D.push_back(track3D);
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
int screenShot(char *Fname, int width, int height, bool color)
{
	int ii, jj, kk;

	unsigned char *data = new unsigned char[width*height * 4];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	for (kk = 0; kk<3; kk++)
		for (jj = 0; jj<height; jj++)
			for (ii = 0; ii<width; ii++)
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










