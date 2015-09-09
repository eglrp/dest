#include"Visualization.h"

#pragma comment(lib, "glut32.lib")

using namespace std;
using namespace cv;

#define RADPERDEG 0.0174533

char *Path;

GLfloat UnitScale = 1.0f; //1 unit corresponds to 1 mm
GLfloat g_ratio, g_coordAxisLength, g_fViewDistance, g_nearPlane, g_farPlane;
int g_Width = 1400, g_Height = 900, org_Width = g_Width, org_Height = g_Height, g_xClick = 0, g_yClick = 0, g_mouseYRotate = 0, g_mouseXRotate = 0;

GLfloat CameraSize, pointSize, normalSize, arrowThickness;
double Discontinuity3DThresh, DiscontinuityTimeThresh = 1000.0; //unit: ms
int nviews = MaxnCams, timeID = 0, TrajecID = 0, otimeID = 0, oTrajecID = 0, TrialID = 0, oTrialID = 0, maxTime = 0, maxTrial = 10000, nTraject = 1;

enum cam_mode { CAM_DEFAULT, CAM_ROTATE, CAM_ZOOM, CAM_PAN };
static cam_mode g_camMode = CAM_DEFAULT;
GLfloat Red[3] = { 1, 0, 0 }, Green[3] = { 0, 1, 0 }, Blue[3] = { 0, 0, 1 }, White[3] = { 1, 1, 1 }, Yellow[3] = { 1.0f, 1.0f, 0 }, Magneta[3] = { 1.f, 0.f, 1.f }, Cycan[3] = { 0.f, 1.f, 1.f };

bool drawPointColor = false, drawPatchNormal = false;
bool g_bButton1Down = false, ReCenterNeeded = false, PickingMode = false, bFullsreen = false, showGroundPlane = false, showAxis = false;
bool SaveScreen = false, SaveStaticViewingParameters = false, SetStaticViewingParameters = false, SaveDynamicViewingParameters = false, SetDynamicViewingParameters = false;

bool drawCorpusPoints = false, drawCorpusCameras = true, drawTimeVaryingCorpusPoints = false, AutomaticUpdate = false;
bool drawTimeVaryingCameraPose = false, drawCameraTraject = false;
bool colorVisibility = false, OneTimeInstanceOnly = false, IndiviualTrajectory = false, Trajectory_Time = true, EndTime = false, showSkeleton = false;
bool drawTimeVarying3DPoints = false, drawCatTimeVarying3DPoint = false, drawNative3DTrajectory = true;
bool TimeVaryingPointsOne = true, TimeVaryingPointsTwo = true, Native3DTrajectoryOne = true, FirstPose = true, SecondPose = false;
double DisplayStartTime = 0.0, DisplayTimeStep = 0.016; //60fps

GLfloat PointsCentroid[3], PointVar[3];
bool *PlottedTimeVaryingCameras = 0;
int *maxFramesToDrawPerCamera = 0;
vector<int> PickedStationaryPoints, PickedDynamicPoints, PickedTraject, PickCams;
vector<double>TimeInstancesStack, TimeInstancesStack2;
vector<Point3d> PickPoint3D, SkeletonPoints;

typedef struct { GLfloat  viewDistance, CentroidX, CentroidY, CentroidZ; int timeID, mouseYRotate, mouseXRotate; } ViewingParas;
vector <ViewingParas> DynamicViewingParas;
VisualizationManager g_vis;

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
	RenderObjects();
	glutSwapBuffers();

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
	if (PickingMode)
	{
		int pickedID = Pick(x, y);
		if (pickedID < 0)
			return;
		if (pickedID < MaxnCams)  //camera
		{
			printf("Pick camera #%d\n", pickedID);

			bool already = false;
			for (int ii = 0; ii < PickCams.size(); ii++)
			{
				if (pickedID == PickCams[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickCams.push_back(pickedID);
		}
		else if (pickedID >= MaxnCams && pickedID < g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size() + MaxnCams)// pick points
		{
			pickedID -= MaxnCams;

			if (ReCenterNeeded)
			{
				printf("New center picked: %d\n", pickedID);
				PointsCentroid[0] = g_vis.CorpusPointPosition[pickedID].x, PointsCentroid[1] = g_vis.CorpusPointPosition[pickedID].y, PointsCentroid[2] = g_vis.CorpusPointPosition[pickedID].z;
				ReCenterNeeded = false;
			}
			else
			{
				printf("Picked %d of (%.3f %.3f %.3f) \n", pickedID, g_vis.CorpusPointPosition[pickedID].x, g_vis.CorpusPointPosition[pickedID].y, g_vis.CorpusPointPosition[pickedID].z);

				bool already = false;
				for (int ii = 0; ii < PickedStationaryPoints.size(); ii++)
				{
					if (pickedID == PickedStationaryPoints[ii])
					{
						already = true; break;
					}
				}
				if (!already)
					PickedStationaryPoints.push_back(pickedID);
			}
		}
		else if (pickedID >= g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size() + MaxnCams && pickedID < g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size() + MaxnCams + MaxnTrajectories) //entire trajectories
		{
			pickedID -= g_vis.CorpusPointPosition.size() + MaxnCams;
			if (pickedID<0 || pickedID>g_vis.Track3DLength.size())
				return;

			printf("Pick trajectory # %d of length %d\n", pickedID, g_vis.Track3DLength[pickedID]);

			bool already = false;
			for (int ii = 0; ii < PickedTraject.size(); ii++)
			{
				if (pickedID == PickedTraject[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedTraject.push_back(pickedID);
		}
		else if (pickedID >= g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size() + MaxnCams + MaxnTrajectories) //end point in the trajectory
		{
			int tid = (pickedID - g_vis.CorpusPointPosition.size() - g_vis.CorpusPointPosition2.size() - MaxnCams - MaxnTrajectories) / MaxnFrames;
			int fid = (pickedID - g_vis.CorpusPointPosition.size() - g_vis.CorpusPointPosition2.size() - MaxnCams - MaxnTrajectories) % MaxnFrames;
			printf("Pick (tid, time): (%d %.2f). 3D: %.4f %.4f %.4f\n", tid, g_vis.Traject3D[tid][fid].timeID, g_vis.Traject3D[tid][fid].WC.x, g_vis.Traject3D[tid][fid].WC.y, g_vis.Traject3D[tid][fid].WC.z);

			bool already = false;
			for (int ii = 0; ii < PickedDynamicPoints.size(); ii++)
			{
				if (pickedID == PickedDynamicPoints[ii])
				{
					already = true; break;
				}
			}
			if (!already)
				PickedDynamicPoints.push_back(pickedID);
		}
	}

	/*double thresh = 20;
	int count = 0;
	for (int ii = 0; ii < g_vis.Traject3D.size(); ii++)
	{
	double otime = g_vis.Traject3D[ii][0].timeID, ntime = g_vis.Traject3D[ii][0].timeID;
	Point3d n3d = g_vis.Traject3D[ii][0].WC, o3d = g_vis.Traject3D[ii][0].WC;

	for (int fid = 0; fid < g_vis.Track3DLength[ii]; fid++)
	{
	ntime = g_vis.Traject3D[ii][fid].timeID;
	n3d = g_vis.Traject3D[ii][fid].WC;
	double dist = sqrt(pow(n3d.x - o3d.x, 2) + pow(n3d.y - o3d.y, 2) + pow(n3d.z - o3d.z, 2));
	if (dist > thresh)
	{
	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	continue;
	}
	if (ntime - otime > 33.3 * 4)
	{
	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	continue;
	}

	if (count == pickedID)
	{
	bool already = false;
	for (int ii = 0; ii < PickedStationaryPoints.size(); ii++)
	{
	if (pickedID == PickedStationaryPoints[ii])
	{
	already = true;
	break;
	}
	}

	if (!already)
	{
	PickedStationaryPoints.push_back(pickedID);
	PickPoint3D.push_back(g_vis.Traject3D[ii][fid].WC);
	printf("Pick point # %d of trajectory %d\n", fid, ii);
	}
	}
	count++;

	o3d = g_vis.Traject3D[ii][fid].WC;
	otime = g_vis.Traject3D[ii][fid].timeID;
	}
	}*/

	return;
}
void Keyboard(unsigned char key, int x, int y)
{
	char Fname[200];
	switch (key)
	{
	case 27:             // ESCAPE key
		if (SaveDynamicViewingParameters)
		{
			sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
				fprintf(fp, "%d %.8f %d %d %.8f %.8f %.8f \n", DynamicViewingParas[ii].timeID, DynamicViewingParas[ii].viewDistance, DynamicViewingParas[ii].mouseXRotate, DynamicViewingParas[ii].mouseYRotate,
				DynamicViewingParas[ii].CentroidX, DynamicViewingParas[ii].CentroidY, DynamicViewingParas[ii].CentroidZ);
			fclose(fp);
		}
		exit(0);
		break;
	case 'i':
		printf("Please enter commands: ");
		cin >> Fname;
		if (strcmp(Fname, "TrajectoryTime") == 0 || strcmp(Fname, "TrajTime") == 0)
		{
			Trajectory_Time = !Trajectory_Time;
			if (Trajectory_Time)
				printf("Trajectory Time: ON\n");
			else
				printf("Trajectory Time: OFF\n");
		}
		if (strcmp(Fname, "TrajectoryTimeEnd") == 0 || strcmp(Fname, "TrajTimeEnd") == 0)
		{
			EndTime = !EndTime;
			if (EndTime)
				printf("Trajectory END Time : ON\n");
			else
				printf("Trajectory END Time :OFF\n");
		}
		if (strcmp(Fname, "IndiviualTrajectory") == 0 || strcmp(Fname, "IndiTraj") == 0)
		{
			IndiviualTrajectory = !IndiviualTrajectory;
			if (IndiviualTrajectory)
				printf("Indiviual Trajectory: ON\n");
			else
				printf("Indiviual Trajectory: OFF\n");
		}
		if (strcmp(Fname, "OneTime") == 0)
		{
			OneTimeInstanceOnly = !OneTimeInstanceOnly;
			if (OneTimeInstanceOnly)
				printf("Show only 1 time instance: ON\n");
			else
				printf("Show only 1 time instance:  OFF\n");
		}
		if (strcmp(Fname, "SwitchPose") == 0)
		{
			if (FirstPose)
				printf("Pose 2: ON\n");
			else
				printf("Pose : ON\n");
			FirstPose = !FirstPose; SecondPose = !SecondPose;
		}
		break;
	case 'o':
		printf("Current time: %d. Please enter the new time: ", timeID);
		cin >> timeID;
		if (timeID < 0)
			timeID = 0;
		if (timeID > TimeInstancesStack.size() - 1)
			timeID = TimeInstancesStack.size() - 1;
	case 'F':
		bFullsreen = !bFullsreen;
		if (bFullsreen)
			glutFullScreen();
		else
		{
			glutReshapeWindow(org_Width, org_Height);
			glutInitWindowPosition(0, 0);
		}
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
	case 'P':
		PickingMode = !PickingMode;
		if (PickingMode)
			printf("Picking Mode: ON\n ");
		else
			printf("Picking Mode: OFF\n ");
		break;
	case 'g':
		printf("Toggle ground plane display\n ");
		showGroundPlane = !showGroundPlane;
		break;
	case 't':
		drawTimeVarying3DPoints = !drawTimeVarying3DPoints;
		if (drawCameraTraject)
			printf("3D point trajectory display: ON\n ");
		else
			printf("3D point trajectory display: OFF\n ");
		break;
	case 'T':
		drawCameraTraject = !drawCameraTraject;
		if (drawCameraTraject)
			printf("Camera trajectory display: ON\n ");
		else
			printf("Camera trajectory display: OFF\n ");
		break;
	case 'r':
		AutomaticUpdate = !AutomaticUpdate;
		if (AutomaticUpdate)
		{
			DisplayStartTime = omp_get_wtime();
			printf("Automatic diplay: ON\n ");
		}
		else
			printf("Automatic diplay: OFF\n ");

		break;
	case '1':
		printf("Toggle corpus points display\n ");
		drawCorpusPoints = !drawCorpusPoints;
		break;
	case '2':
		
		drawCorpusCameras = !drawCorpusCameras;
		if (drawCorpusCameras)
			printf("Corpus cameras display: ON\n ");
		else
			printf("Corpus cameras display: OFF\n ");
		break;
	case '3':
		printf("Toggle corpus trajectory display\n ");
		drawTimeVaryingCorpusPoints = !drawTimeVaryingCorpusPoints;
		break;
	case '4':
		printf("Save OpenGL viewing parameters\n");
		SaveStaticViewingParameters = true;
		break;
	case '5':
		printf("Read OpenGL viewing parameters\n");
		SetStaticViewingParameters = true;
		break;
	case '6':
		SaveDynamicViewingParameters = !SaveDynamicViewingParameters;
		if (SaveDynamicViewingParameters)
		{
			printf("Save OpenGL dynamic viewing parameters: ON\nStart pushing into stack");
			timeID = 0;
		}
		else
		{
			printf("Save OpenGL dynamic viewing parameters: OFF\n. Flush the stack out\n");
			sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "w+");
			for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
				fprintf(fp, "%d %.8f %d %d %.8f %.8f %.8f \n", DynamicViewingParas[ii].timeID, DynamicViewingParas[ii].viewDistance, DynamicViewingParas[ii].mouseXRotate, DynamicViewingParas[ii].mouseYRotate,
				DynamicViewingParas[ii].CentroidX, DynamicViewingParas[ii].CentroidY, DynamicViewingParas[ii].CentroidZ);
			fclose(fp);
		}
		DynamicViewingParas.clear();

		break;
	case '7':
		printf("Read OpenGL dynamic viewing parameters\n");
		SetDynamicViewingParameters = true;
		DynamicViewingParas.clear();
		break;
	case '8':
		Native3DTrajectoryOne = !Native3DTrajectoryOne;
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
	case 'f':
		TimeVaryingPointsTwo = !TimeVaryingPointsTwo;
		break;
	}
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_PAGE_UP:
		TrialID++;
		if (TrialID > maxTrial)
			TrialID = maxTrial;
		printf("Current data trial: %d ...", TrialID);
		break;
	case GLUT_KEY_PAGE_DOWN:
		TrialID--;
		if (TrialID < 0)
			TrialID = 0;
		printf("Current data trial: %d ...", TrialID);
		break;
	case GLUT_KEY_UP:
		TrajecID++;
		if (TrajecID > nTraject)
			TrajecID = TrajecID;
		printf("Current TrajecID: %d\n", TrajecID);
		break;
	case GLUT_KEY_DOWN:
		TrajecID--;
		if (TrajecID < 0)
			TrajecID = 0;
		printf("Current TrajecID: %d\n", TrajecID);
		break;
	case GLUT_KEY_LEFT:
		timeID--;
		if (timeID < 0)
			timeID = 0;
		printf("Current time: %.2f (id: %d)\n", TimeInstancesStack[timeID], timeID);
		PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_RIGHT:
		timeID++;
		if (TimeInstancesStack.size() >0 && timeID >TimeInstancesStack.size() - 1 )
			timeID = TimeInstancesStack.size()-1;
		printf("Current time: %.2f (id: %d)\n", TimeInstancesStack[timeID], timeID);
		PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_HOME:
		timeID = 0;
		printf("Current time: %.2f (id: %d)\n", TimeInstancesStack[timeID], timeID);
		PickedDynamicPoints.clear();
		break;
	case GLUT_KEY_END:
		timeID = TimeInstancesStack.size() - 1;
		printf("Current time: %.2f (id: %d)\n", TimeInstancesStack[timeID], timeID);
		PickedDynamicPoints.clear();
		break;
	}

	glutPostRedisplay();
}
void MouseButton(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		g_bButton1Down = (state == GLUT_DOWN) ? true : false;
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
		{
			printf("Deselect all points\n");
			PickedStationaryPoints.clear(), PickedDynamicPoints.clear(), PickCams.clear(), PickedTraject.clear(), PickPoint3D.clear();
		}
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
			g_fViewDistance += 5.0f*(y - g_yClick) *UnitScale;
		else if (g_camMode == CAM_ROTATE)
		{
			showAxis = true;
			g_mouseXRotate += (x - g_xClick);
			g_mouseYRotate -= (y - g_yClick);
			g_mouseXRotate = g_mouseXRotate % 360;
			g_mouseYRotate = g_mouseYRotate % 360;
		}
		else if (g_camMode == CAM_PAN)
		{
			showAxis = true;
			float dX = -(x - g_xClick)*UnitScale, dY = (y - g_yClick)*UnitScale;

			float cphi = cos(-Pi*g_mouseYRotate / 180), sphi = sin(-Pi*g_mouseYRotate / 180);
			float Rx[9] = { 1, 0, 0, 0, cphi, -sphi, 0, sphi, cphi };

			cphi = cos(-Pi*g_mouseXRotate / 180), sphi = sin(-Pi*g_mouseXRotate / 180);
			float Ry[9] = { cphi, 0, sphi, 0, 1, 0, -sphi, 0, cphi };

			float R[9];  mat_mul(Rx, Ry, R, 3, 3, 3);
			float incre[3], orgD[3] = { dX, dY, 0 }; mat_mul(R, orgD, incre, 3, 3, 1);

			PointsCentroid[0] += incre[0], PointsCentroid[1] += incre[1], PointsCentroid[2] += incre[2];
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

void Draw_Axes(void)
{
	glPushMatrix();

	glBegin(GL_LINES);
	glColor3f(1, 0, 0); // X axis is red.
	glVertex3f(0, 0, 0);
	glVertex3f(g_coordAxisLength, 0, 0);
	glColor3f(0, 1, 0); // Y axis is green.
	glVertex3f(0, 0, 0);
	glVertex3f(0, g_coordAxisLength, 0);
	glColor3f(0, 0, 1); // Z axis is blue.
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, g_coordAxisLength);
	glEnd();

	glPopMatrix();
}
void DrawCamera(bool highlight)
{
	glColorMaterial(GL_FRONT, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	if (!highlight)
		glColor3fv(Red);
	else
		glColor3fv(Blue);

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

	if (!highlight)
		glColor3fv(Green);
	else
		glColor3fv(Blue);

	// we also has to draw a square for the bottom of the pyramid so that as it rotates we wont be able see inside of it but all a square is is two triangle put together
	glBegin(GL_LINE_STRIP);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glVertex3f(-0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, -0.5*CameraSize, 1 * CameraSize);
	glVertex3f(0.5*CameraSize, 0.5*CameraSize, 1 * CameraSize);
	glEnd();
	glDisable(GL_COLOR_MATERIAL);
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
void DetermineDiscontinuityInTrajectory(vector<Point2i> &segNode, Trajectory3D* Trajec3D, int nframes)
{
	double otime = Trajec3D[0].timeID, ntime = Trajec3D[0].timeID;
	Point3d n3d = Trajec3D[0].WC, o3d = Trajec3D[0].WC;
	Point2i Node; Node.x = 0;
	for (int fid = 0; fid < nframes; fid++)
	{
		ntime = Trajec3D[fid].timeID;
		n3d = Trajec3D[fid].WC;
		double dist = sqrt(pow(n3d.x - o3d.x, 2) + pow(n3d.y - o3d.y, 2) + pow(n3d.z - o3d.z, 2));
		if (dist > Discontinuity3DThresh)
		{
			o3d = Trajec3D[fid].WC;
			otime = Trajec3D[fid].timeID;
			Node.y = fid;
			segNode.push_back(Node);
			if (fid + 1 < nframes)
				Node.x = fid + 1;
			continue;
		}
		if (ntime - otime > DiscontinuityTimeThresh)
		{
			o3d = Trajec3D[fid].WC;
			otime = Trajec3D[fid].timeID;
			Node.y = fid;
			segNode.push_back(Node);
			if (fid + 1 < nframes)
				Node.x = fid + 1;
			continue;
		}

		o3d = Trajec3D[fid].WC;
		otime = Trajec3D[fid].timeID;
	}
	Node.y = nframes - 1;
	segNode.push_back(Node);

	return;
}
void RenderObjects()
{
	if (otimeID != timeID && drawTimeVarying3DPoints)
	{
		ReCenterNeeded = true;
		ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, ReCenterNeeded);
		ReadCurrent3DGL2(Path, drawPointColor, drawPatchNormal, timeID, ReCenterNeeded);
		otimeID = timeID;
	}

	if ((oTrialID != TrialID))
	{
		Read3DTrajectory(Path, TrialID, colorVisibility);
		oTrialID = TrialID;
	}
	if (TimeInstancesStack.size() > 0 && TimeInstancesStack.back() - 1 < timeID)
		timeID = TimeInstancesStack.back() - 1;
	if (EndTime)
		timeID = TimeInstancesStack.back() - 1;

	//Draw not picked corpus points
	if (drawCorpusPoints)
	{
		for (unsigned int i = 0; i < g_vis.CorpusPointPosition.size(); ++i)
		{
			glLoadName(i + MaxnCams);//for picking purpose

			bool picked = false;
			for (unsigned int j = 0; j < PickedStationaryPoints.size(); j++)
				if (i == PickedStationaryPoints[j])
					picked = true;

			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition[i].x - PointsCentroid[0], g_vis.CorpusPointPosition[i].y - PointsCentroid[1], g_vis.CorpusPointPosition[i].z - PointsCentroid[2]);
			if (!picked&&drawPointColor)
				glColor3f(g_vis.CorpusPointColor[i].x, g_vis.CorpusPointColor[i].y, g_vis.CorpusPointColor[i].z);
			else
				glColor3fv(Red);
			glutSolidSphere(pointSize / 1.25, 4, 4);
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
			glLoadName(i + g_vis.CorpusPointPosition.size() + MaxnCams);//for picking purpose

			bool picked = false;
			for (unsigned int j = 0; j < PickedStationaryPoints.size(); j++)
				if (i == PickedStationaryPoints[j])
					picked = true;

			glPushMatrix();
			glTranslatef(g_vis.CorpusPointPosition2[i].x - PointsCentroid[0], g_vis.CorpusPointPosition2[i].y - PointsCentroid[1], g_vis.CorpusPointPosition2[i].z - PointsCentroid[2]);
			if (!picked&&drawPointColor)
				glColor3f(g_vis.CorpusPointColor2[i].x, g_vis.CorpusPointColor2[i].y, g_vis.CorpusPointColor2[i].z);
			else
				glColor3fv(Green);
			glutSolidSphere(pointSize / 1.25, 4, 4);
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
				glColor3fv(Green);
			glutSolidSphere(pointSize, 4, 4);
			glPopMatrix();
		}
	}

	//draw Corpus camera 
	if (drawCorpusCameras)
	{
		for (int ii = 0; ii < g_vis.glCorpusCameraInfo.size(); ii++)
		{
			float* centerPt = g_vis.glCorpusCameraInfo[ii].camCenter;
			GLfloat* R = g_vis.glCorpusCameraInfo[ii].Rgl;
			glLoadName(ii);//for picking purpose

			bool picked = false;
			for (int jj = 0; jj < PickCams.size(); jj++)
				if (ii == PickCams[jj])
				{
					picked = true;
					break;
				}

			glPushMatrix();
			glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
			glMultMatrixf(R);
			if (picked)
				DrawCamera(true);
			else
				DrawCamera();
			glPopMatrix();
		}
	}

	if (drawTimeVarying3DPoints)
	{
		if (TimeVaryingPointsOne)
		{
			for (int i = 0; i < g_vis.PointPosition.size(); i++)
			{
				glPushMatrix();
				glTranslatef(g_vis.PointPosition[i].x - PointsCentroid[0], g_vis.PointPosition[i].y - PointsCentroid[1], g_vis.PointPosition[i].z - PointsCentroid[2]);
				if (drawPointColor)
					glColor3f(g_vis.PointColor[i].x, g_vis.PointColor[i].y, g_vis.PointColor[i].z);
				else
					glColor3fv(Blue);
				glutSolidSphere(pointSize, 4, 4);
				glPopMatrix();
			}
			if (g_vis.PointPosition.size() > 0)
				RenderSkeleton2(g_vis.PointPosition, Blue);
		}

		if (TimeVaryingPointsTwo)
		{
			for (int i = 0; i < g_vis.PointPosition2.size(); i++)
			{
				glPushMatrix();
				glTranslatef(g_vis.PointPosition2[i].x - PointsCentroid[0], g_vis.PointPosition2[i].y - PointsCentroid[1], g_vis.PointPosition2[i].z - PointsCentroid[2]);
				if (drawPointColor)
					glColor3f(g_vis.PointColor2[i].x, g_vis.PointColor2[i].y, g_vis.PointColor2[i].z);
				else
					glColor3fv(Yellow);
				glutSolidSphere(pointSize, 4, 4);
				glPopMatrix();
			}
			if (g_vis.PointPosition2.size() > 0)
				RenderSkeleton2(g_vis.PointPosition2, Yellow);
		}

		//Concat 3D points from 3D points at individual time instance
		if (drawCatTimeVarying3DPoint)
		{
			if (TimeVaryingPointsOne)
			{
				for (int pid = 0; pid < g_vis.PointPosition.size(); pid++)
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					for (int fid = 0; fid <= min(maxTime, timeID); fid++)
					{
						glVertex3f(g_vis.catPointPosition[pid][fid].x - PointsCentroid[0], g_vis.catPointPosition[pid][fid].y - PointsCentroid[1], g_vis.catPointPosition[pid][fid].z - PointsCentroid[2]);
						glColor3fv(Blue);
					}
					glEnd();
					glPopMatrix();
				}
			}

			if (TimeVaryingPointsTwo)
			{
				for (int pid = 0; pid < g_vis.PointPosition2.size(); pid++)
				{
					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					for (int fid = 0; fid <= min(maxTime, timeID); fid++)
					{
						glVertex3f(g_vis.catPointPosition2[pid][fid].x - PointsCentroid[0], g_vis.catPointPosition2[pid][fid].y - PointsCentroid[1], g_vis.catPointPosition2[pid][fid].z - PointsCentroid[2]);
						glColor3fv(Yellow);
					}
					glEnd();
					glPopMatrix();
				}
			}
		}
	}

	//draw 3d trajectories
	if (drawNative3DTrajectory)
	{
		if (OneTimeInstanceOnly &&showSkeleton)
			SkeletonPoints.clear();

		if (drawCameraTraject)
		{
			for (int vid = 0; vid < nviews; vid++)
			{
				int maxFramesToDraw = 0;
				for (int tid = 0; tid < g_vis.Traject3D.size(); tid++)
				{
					for (int fid = 0; fid < g_vis.Track3DLength[tid]; fid++)
					{
						if (g_vis.Traject3D[tid][fid].timeID > TimeInstancesStack[timeID])
							break;
						if (vid == g_vis.Traject3D[tid][fid].viewID)
							maxFramesToDraw = max(maxFramesToDraw, g_vis.Traject3D[tid][fid].frameID);
					}
				}
				maxFramesToDrawPerCamera[vid] = maxFramesToDraw;
			}
		}

		vector<Point2i> segNode;
		GLfloat TrajPointColorI[4]; 	float alpha = 0.2;
		for (int tid = 0; tid < (int)g_vis.Traject3D.size(); tid++)
		{
			if (IndiviualTrajectory)
				if (tid != TrajecID)
					continue;

			segNode.clear();
			DetermineDiscontinuityInTrajectory(segNode, g_vis.Traject3D[tid], g_vis.Track3DLength[tid]);

			//draw lines
			glLineWidth(2.0);
			if (!OneTimeInstanceOnly)
			{
				for (int segID = 0; segID < segNode.size(); segID++)
				{
					if (segNode[segID].y - segNode[segID].x < 3)
						continue;

					glPushMatrix();
					glBegin(GL_LINE_STRIP);
					for (int fid = segNode[segID].x; fid < segNode[segID].y; fid++)
					{
						TrajPointColorI[0] = g_vis.Traject3D[tid][fid].rgb.x, TrajPointColorI[1] = g_vis.Traject3D[tid][fid].rgb.y, TrajPointColorI[2] = g_vis.Traject3D[tid][fid].rgb.z, TrajPointColorI[3] = 1.0f;
						if (Trajectory_Time)
							TrajPointColorI[3] = 1.0f;// 1.0f* pow(g_vis.Traject3D[tid][fid].timeID / TimeInstancesStack[timeID], 3);

						if (Trajectory_Time && g_vis.Traject3D[tid][fid].timeID > TimeInstancesStack[timeID])
							continue;
						if (TrajPointColorI[3] < 7.5e-2)
							continue;

						glVertex3f(g_vis.Traject3D[tid][fid].WC.x - PointsCentroid[0], g_vis.Traject3D[tid][fid].WC.y - PointsCentroid[1], g_vis.Traject3D[tid][fid].WC.z - PointsCentroid[2]);
						glColor4fv(TrajPointColorI);
					}
					glEnd();
					glPopMatrix();
				}
			}

			//draw points on the lines
			for (int segID = 0; segID < segNode.size(); segID++)
			{
				if (segNode[segID].y - segNode[segID].x < 3)
					continue;

				for (int fid = segNode[segID].x; fid < segNode[segID].y; fid++)
				{
					if (!OneTimeInstanceOnly &&Trajectory_Time && g_vis.Traject3D[tid][fid].timeID > TimeInstancesStack[timeID])
						continue;
					else if (Trajectory_Time && g_vis.Traject3D[tid][fid].timeID != TimeInstancesStack[timeID])
						continue;

					if (OneTimeInstanceOnly &&showSkeleton)
						SkeletonPoints.push_back(Point3d(g_vis.Traject3D[tid][fid].WC.x, g_vis.Traject3D[tid][fid].WC.y, g_vis.Traject3D[tid][fid].WC.z));

					TrajPointColorI[0] = g_vis.Traject3D[tid][fid].rgb.x, TrajPointColorI[1] = g_vis.Traject3D[tid][fid].rgb.y, TrajPointColorI[2] = g_vis.Traject3D[tid][fid].rgb.z, TrajPointColorI[3] = 1.0f;
					if (Trajectory_Time)
						TrajPointColorI[3] = 1.0f* pow(g_vis.Traject3D[tid][fid].timeID / TimeInstancesStack[timeID], 3);

					if (TrajPointColorI[3] < 7.5e-2)
						continue;

					int pickID = g_vis.CorpusPointPosition.size() + g_vis.CorpusPointPosition2.size() + MaxnCams + MaxnTrajectories + tid*MaxnFrames + fid;
					glLoadName(pickID);//for picking purpose
					bool picked = false;
					for (unsigned int ii = 0; ii < PickedDynamicPoints.size(); ii++)
					{
						if (pickID == PickedDynamicPoints[ii])
						{
							picked = true;
							break;
						}
					}
					if (picked)
						TrajPointColorI[0] = 0.f, TrajPointColorI[1] = 0.f, TrajPointColorI[2] = 1.f, TrajPointColorI[3] = 1.0f;

					glPushMatrix();
					Point3d xyz = g_vis.Traject3D[tid][fid].WC;
					glTranslatef(g_vis.Traject3D[tid][fid].WC.x - PointsCentroid[0], g_vis.Traject3D[tid][fid].WC.y - PointsCentroid[1], g_vis.Traject3D[tid][fid].WC.z - PointsCentroid[2]);
					glColor4fv(TrajPointColorI);
					glutSolidSphere(.01, 10, 10);
					glPopMatrix();
				}
			}


			if (Trajectory_Time && drawTimeVaryingCameraPose)
			{
				for (int segID = 0; segID < segNode.size(); segID++)
				{
					if (segNode[segID].y - segNode[segID].x < 3)
						continue;

					for (int fid = segNode[segID].x; fid < segNode[segID].y; fid++)
					{
						if (timeID>0 && g_vis.Traject3D[tid][fid].timeID <= TimeInstancesStack[timeID - 1])
							continue;
						if (g_vis.Traject3D[tid][fid].timeID > TimeInstancesStack[timeID])
							continue;

						int visibleCam = g_vis.Traject3D[tid][fid].viewID, visibleCam_FrameID = g_vis.Traject3D[tid][fid].frameID;
						if (visibleCam_FrameID< 0 || visibleCam_FrameID>g_vis.glCameraPoseInfo[visibleCam].size() || PlottedTimeVaryingCameras[visibleCam])
							continue;
						PlottedTimeVaryingCameras[visibleCam] = true;

						float* centerPt = g_vis.glCameraPoseInfo[visibleCam][visibleCam_FrameID].camCenter;
						if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
							continue;
						GLfloat* R = g_vis.glCameraPoseInfo[visibleCam][visibleCam_FrameID].Rgl;

						glLoadName(visibleCam);//for picking purpose
						glPushMatrix();
						glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
						glMultMatrixf(R);
						DrawCamera();
						glPopMatrix();

						if (drawCameraTraject)
						{
							glBegin(GL_LINE_STRIP);
							for (int fid = 0; fid <= maxFramesToDrawPerCamera[visibleCam]; fid++)
							{
								float* centerPt = g_vis.glCameraPoseInfo[visibleCam][fid].camCenter;
								if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
									continue;
								glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
								glColor3fv(White);
							}
							glEnd();
						}
					}
				}
			}
		}

		if (OneTimeInstanceOnly &&showSkeleton)
		{
			if ((int)SkeletonPoints.size() == 31)
				RenderSkeleton2(SkeletonPoints, Red);
		}

		if (drawTimeVaryingCameraPose)
			for (int ii = 0; ii < nviews; ii++)
				PlottedTimeVaryingCameras[ii] = false;
	}

	//draw time varying camera pose
	if (FirstPose)
	{
		if ((drawTimeVaryingCameraPose && !Trajectory_Time) || drawTimeVaryingCameraPose && timeID == TimeInstancesStack.size() - 1)
		{
			for (int j = 0; j < nviews; j++)
			{
				if (g_vis.glCameraPoseInfo[j].size() > 0)
				{
					if (drawCameraTraject)
					{
						glPushMatrix();
						glBegin(GL_LINE_STRIP);
						int maxtime = min(timeID, (int)g_vis.glCameraPoseInfo[j].size() - 1);
						for (unsigned int i = 0; i <= maxtime; ++i)
						{
							if (g_vis.glCameraPoseInfo[j][i].frameID < 0 || g_vis.glCameraPoseInfo[j][i].frameID >timeID)
								continue;
							float* centerPt = g_vis.glCameraPoseInfo[j][i].camCenter;
							if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
								continue;
							glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							if (j == 0)
								glColor3fv(Red);
							else if (j == 1)
								glColor3fv(Green);
							else if (j == 2)
								glColor3fv(Blue);
							else if (j == 3)
								glColor3fv(Yellow);
							else if (j == 4)
								glColor3fv(Magneta);
							else if (j == 5)
								glColor3fv(Cycan);
							else
								glColor3fv(White);
						}
						glEnd();

						//drawPoints on the trajectory
						for (unsigned int i = 0; i <= maxtime; ++i)
						{
							if (g_vis.glCameraPoseInfo[j][i].frameID < 0 || g_vis.glCameraPoseInfo[j][i].frameID >timeID)
								continue;
							float* centerPt = g_vis.glCameraPoseInfo[j][i].camCenter;
							if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
								continue;
							glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);

							glPushMatrix();
							glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							if (j == 0)
								glColor3fv(Red);
							else if (j == 1)
								glColor3fv(Green);
							else if (j == 2)
								glColor3fv(Blue);
							else if (j == 3)
								glColor3fv(Yellow);
							else if (j == 4)
								glColor3fv(Magneta);
							else if (j == 5)
								glColor3fv(Cycan);
							else
								glColor3fv(White);

							glutSolidSphere(pointSize / 10.0, 10, 10);
							glPopMatrix();
						}

						//Draw camera at the end of the trajectory
						int dist, closestID = -1, closestFrame = maxTime + 1;
						for (int i = 0; i < maxtime; i++)
						{
							dist = abs(g_vis.glCameraPoseInfo[j][i].frameID - timeID);
							if (closestFrame > dist)
								closestFrame = dist, closestID = i;
						}

						if (closestFrame == 0)
						{
							float* centerPt = g_vis.glCameraPoseInfo[j][closestID].camCenter;
							GLfloat* R = g_vis.glCameraPoseInfo[j][closestID].Rgl;

							glPushMatrix();
							glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							glMultMatrixf(R);
							DrawCamera();
							glPopMatrix();
						}
					}

					for (unsigned int i = 0; i < g_vis.glCameraPoseInfo[j].size(); ++i)
					{
						if (g_vis.glCameraPoseInfo[j][i].frameID < 0 || g_vis.glCameraPoseInfo[j][i].frameID >timeID)
							continue;

						if (drawCameraTraject)
						{
							if (!OneTimeInstanceOnly && g_vis.glCameraPoseInfo[j][i].frameID > timeID)
								continue;
							if (OneTimeInstanceOnly && g_vis.glCameraPoseInfo[j][i].frameID != timeID)
								continue;
						}
						else
							if (g_vis.glCameraPoseInfo[j][i].frameID != timeID)
								continue;

						float* centerPt = g_vis.glCameraPoseInfo[j].at(i).camCenter;
						if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
							continue;
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
	}
	if (SecondPose)
	{
		if (TimeInstancesStack2.size() > 0 && (drawTimeVaryingCameraPose && !Trajectory_Time) || drawTimeVaryingCameraPose && timeID == TimeInstancesStack2.size() - 1)
		{
			for (int j = 0; j < nviews; j++)
			{
				if (g_vis.glCameraPoseInfo2[j].size() > 0)
				{
					if (drawCameraTraject)
					{
						glPushMatrix();
						glBegin(GL_LINE_STRIP);
						int maxtime = min(timeID, (int)g_vis.glCameraPoseInfo2[j].size() - 1);
						for (unsigned int i = 0; i <= maxtime; ++i)
						{
							if (g_vis.glCameraPoseInfo2[j][i].frameID < 0 || g_vis.glCameraPoseInfo2[j][i].frameID >timeID)
								continue;
							float* centerPt = g_vis.glCameraPoseInfo2[j][i].camCenter;
							if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
								continue;
							glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							if (j == 0)
								glColor3fv(Red);
							else if (j == 1)
								glColor3fv(Green);
							else if (j == 2)
								glColor3fv(Blue);
							else if (j == 3)
								glColor3fv(Yellow);
							else if (j == 4)
								glColor3fv(Magneta);
							else if (j == 5)
								glColor3fv(Cycan);
							else
								glColor3fv(White);
						}
						glEnd();

						//drawPoints on the trajectory
						for (unsigned int i = 0; i <= maxtime; ++i)
						{
							if (g_vis.glCameraPoseInfo2[j][i].frameID < 0 || g_vis.glCameraPoseInfo2[j][i].frameID >timeID)
								continue;
							float* centerPt = g_vis.glCameraPoseInfo2[j][i].camCenter;
							if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
								continue;
							glVertex3f(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);

							glPushMatrix();
							glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							if (j == 0)
								glColor3fv(Red);
							else if (j == 1)
								glColor3fv(Green);
							else if (j == 2)
								glColor3fv(Blue);
							else if (j == 3)
								glColor3fv(Yellow);
							else if (j == 4)
								glColor3fv(Magneta);
							else if (j == 5)
								glColor3fv(Cycan);
							else
								glColor3fv(White);

							glutSolidSphere(pointSize / 10.0, 10, 10);
							glPopMatrix();
						}

						//Draw camera at the end of the trajectory
						int dist, closestID = -1, closestFrame = maxTime + 1;
						for (int i = 0; i < maxtime; i++)
						{
							dist = abs(g_vis.glCameraPoseInfo2[j][i].frameID - timeID);
							if (closestFrame > dist)
								closestFrame = dist, closestID = i;
						}

						if (closestFrame == 0)
						{
							float* centerPt = g_vis.glCameraPoseInfo2[j][closestID].camCenter;
							GLfloat* R = g_vis.glCameraPoseInfo2[j][closestID].Rgl;

							glPushMatrix();
							glTranslatef(centerPt[0] - PointsCentroid[0], centerPt[1] - PointsCentroid[1], centerPt[2] - PointsCentroid[2]);
							glMultMatrixf(R);
							DrawCamera();
							glPopMatrix();
						}
					}

					for (unsigned int i = 0; i < g_vis.glCameraPoseInfo2[j].size(); ++i)
					{
						if (g_vis.glCameraPoseInfo2[j][i].frameID < 0 || g_vis.glCameraPoseInfo2[j][i].frameID >timeID)
							continue;

						if (drawCameraTraject)
						{
							if (!OneTimeInstanceOnly && g_vis.glCameraPoseInfo2[j][i].frameID > timeID)
								continue;
							if (OneTimeInstanceOnly && g_vis.glCameraPoseInfo2[j][i].frameID != timeID)
								continue;
						}
						else
							if (g_vis.glCameraPoseInfo2[j][i].frameID != timeID)
								continue;

						float* centerPt = g_vis.glCameraPoseInfo2[j].at(i).camCenter;
						if (abs(centerPt[0]) + abs(centerPt[1]) + abs(centerPt[2]) < 0.01)
							continue;
						GLfloat* R = g_vis.glCameraPoseInfo2[j].at(i).Rgl;

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
	}

	glFlush();
}

void display(void)
{
	drawTimeVarying3DPoints;
	if (SaveStaticViewingParameters)
	{
		char Fname[200]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path);	FILE *fp = fopen(Fname, "w+");
		fprintf(fp, "%.8f %d %d %.8f %.8f %.8f ", g_fViewDistance, g_mouseYRotate, g_mouseXRotate, PointsCentroid[0], PointsCentroid[1], PointsCentroid[2]);
		fclose(fp);
		SaveStaticViewingParameters = false;
	}
	if (SetStaticViewingParameters)
	{
		char Fname[200]; sprintf(Fname, "%s/OpenGLViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			fscanf(fp, "%f %d %d %f %f %f", &g_fViewDistance, &g_mouseYRotate, &g_mouseXRotate, &PointsCentroid[0], &PointsCentroid[1], &PointsCentroid[2]);
			fclose(fp);
		}
		SetStaticViewingParameters = false;
	}

	if (SaveDynamicViewingParameters)
	{
		ViewingParas vparas;
		vparas.timeID = timeID, vparas.viewDistance = g_fViewDistance, vparas.mouseXRotate = g_mouseXRotate, vparas.mouseYRotate = g_mouseYRotate,
			vparas.CentroidX = PointsCentroid[0], vparas.CentroidY = PointsCentroid[1], vparas.CentroidZ = PointsCentroid[2];
		DynamicViewingParas.push_back(vparas);
	}
	if (SetDynamicViewingParameters)
	{
		char Fname[200]; sprintf(Fname, "%s/OpenGLDynamicViewingPara.txt", Path); FILE *fp = fopen(Fname, "r");
		if (fp != NULL)
		{
			ViewingParas vparas;
			while (fscanf(fp, "%d %f %d %d %f %f %f", &vparas.timeID, &vparas.viewDistance, &vparas.mouseYRotate, &vparas.mouseXRotate, &vparas.CentroidX, &vparas.CentroidY, &vparas.CentroidZ) != EOF)
				DynamicViewingParas.push_back(vparas);
			fclose(fp);
		}
		SetDynamicViewingParameters = false;

		//Reset time and start rendering;
		timeID = 0;
	}
	for (int ii = 0; ii < (int)DynamicViewingParas.size(); ii++)
	{
		if (DynamicViewingParas[ii].timeID == timeID)
		{
			g_fViewDistance = DynamicViewingParas[ii].viewDistance, g_mouseXRotate = DynamicViewingParas[ii].mouseXRotate, g_mouseYRotate = DynamicViewingParas[ii].mouseYRotate,
				PointsCentroid[0] = DynamicViewingParas[ii].CentroidX, PointsCentroid[1] = DynamicViewingParas[ii].CentroidY, PointsCentroid[2] = DynamicViewingParas[ii].CentroidZ;
			break;
		}
	}
	// Clear frame buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();
	glTranslatef(0, 0, -g_fViewDistance);
	//glRotated(180,  0, 0, 1), glRotated(180, 0, 1, 0);
	glRotated(-g_mouseYRotate, 1, 0, 0);
	glRotated(-g_mouseXRotate, 0, 1, 0);

	RenderObjects();
	if (showAxis)
		Draw_Axes(), showAxis = false;

	if (showGroundPlane)
		RenderGroundPlane();

	glutSwapBuffers();

	if (SaveScreen && oTrajecID != timeID)
	{
		char Fname[200];	sprintf(Fname, "%s/ScreenShot", Path, timeID); makeDir(Fname);
		sprintf(Fname, "%s/ScreenShot/%d.png", Path, timeID);
		screenShot(Fname, g_Width, g_Height, true);
		oTrajecID = timeID;
	}
}
void IdleFunction(void)
{
	if (AutomaticUpdate)
	{
		double DisplayCurrentTime = omp_get_wtime();
		if (DisplayCurrentTime - DisplayStartTime > DisplayTimeStep)
		{
			DisplayStartTime = DisplayCurrentTime;
			timeID++;
			if (timeID > TimeInstancesStack.size() - 1)
				timeID = TimeInstancesStack.size() - 1;
			printf("Current time: %d\n", timeID);
		}
		display();
	}
	return;
}
void Visualization()
{
	char *myargv[1];
	int myargc = 1;
	myargv[0] = "SfM";
	glutInit(&myargc, myargv);

	glutInitWindowSize(g_Width, g_Height);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("SfM!");


	glShadeModel(GL_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	//select clearing (background) color
	//glClearColor(1.0, 1.0, 1.0, 0.0);
	glClearColor(0.0, 0.0, 0.0, 0.0);

	glutDisplayFunc(display);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	glutReshapeFunc(ReshapeGL);
	glutMouseFunc(MouseButton);
	glutIdleFunc(IdleFunction);
	glutMotionFunc(MouseMotion);

	glutMainLoop();

}
int visualizationDriver(char *inPath, int nViews, int StartTime, int StopTime, bool hasColor, bool hasPatchNormal, bool hasTimeVaryingCameraPose, bool hasTimeVarying3DPoints, bool hasCatTimeVarying3DPoints, bool docolorVisibility, int CurrentTime)
{
	Path = inPath;
	nviews = nViews;
	drawPointColor = hasColor, drawPatchNormal = hasPatchNormal;
	drawTimeVaryingCameraPose = hasTimeVaryingCameraPose, drawTimeVarying3DPoints = hasTimeVarying3DPoints;
	drawCatTimeVarying3DPoint = hasCatTimeVarying3DPoints;
	colorVisibility = docolorVisibility;
	showSkeleton = true;

	maxTime = StopTime, timeID = CurrentTime;
	VisualizationManager g_vis;
	ReadCurrentSfmGL(Path, drawPointColor, drawPatchNormal);
	//ReadCurrent3DGL(Path, drawPointColor, drawPatchNormal, timeID, true);
	//ReadCurrent3DGL2(Path, drawPointColor, drawPatchNormal, timeID, false);
	//Read3DTrajectory(Path, 0, colorVisibility);
	//Read3DTrajectory2(Path, 140, 0, colorVisibility);
	Read3DTrajectory2(Path, 30, 0);
	//Read3DTrajectoryWithCovariance(Path);

	//PointsCentroid[0] = 919, PointsCentroid[1] = 154, PointsCentroid[2] = 894;
	//PointsCentroid[0] = -108, PointsCentroid[1] = -1356, PointsCentroid[2] = 5228;

	UnitScale = sqrt(pow(PointVar[0], 2) + pow(PointVar[1], 2) + pow(PointVar[2], 2)) / 750.0;
	g_coordAxisLength = 50.f*UnitScale, g_fViewDistance = 1000 * UnitScale* VIEWING_DISTANCE_MIN;
	g_nearPlane = 1.0*UnitScale, g_farPlane = 30000.f * UnitScale;
	CameraSize = 20.0f*UnitScale, pointSize = 1.0f*UnitScale, normalSize = 5.f*UnitScale, arrowThickness = .1f*UnitScale;
	Discontinuity3DThresh = 500.0*UnitScale;

	if (drawTimeVaryingCameraPose)
	{
		ReadCurrentPosesGL(Path, nViews, StartTime, StopTime);
		//ReadCurrentPosesGL2(Path, nViews, StartTime, StopTime);

		PlottedTimeVaryingCameras = new bool[nViews];
		maxFramesToDrawPerCamera = new int[nViews];
		for (int ii = 0; ii < nViews; ii++)
			PlottedTimeVaryingCameras[ii] = false;
	}
	Visualization();

	return 0;
}

void GetRCGL(CameraData &camInfo)
{
	double iR[9], center[3];
	mat_invert(camInfo.R, iR);

	camInfo.Rgl[0] = camInfo.R[0], camInfo.Rgl[1] = camInfo.R[1], camInfo.Rgl[2] = camInfo.R[2], camInfo.Rgl[3] = 0.0;
	camInfo.Rgl[4] = camInfo.R[3], camInfo.Rgl[5] = camInfo.R[4], camInfo.Rgl[6] = camInfo.R[5], camInfo.Rgl[7] = 0.0;
	camInfo.Rgl[8] = camInfo.R[6], camInfo.Rgl[9] = camInfo.R[7], camInfo.Rgl[10] = camInfo.R[8], camInfo.Rgl[11] = 0.0;
	camInfo.Rgl[12] = 0, camInfo.Rgl[13] = 0, camInfo.Rgl[14] = 0, camInfo.Rgl[15] = 1.0;

	mat_mul(iR, camInfo.T, center, 3, 3, 1);
	camInfo.camCenter[0] = -center[0], camInfo.camCenter[1] = -center[1], camInfo.camCenter[2] = -center[2];
	return;
}
void GetRCGL(double *R, double *T, double *Rgl, double *C)
{
	double iR[9], center[3];
	mat_invert(R, iR);

	Rgl[0] = R[0], Rgl[1] = R[1], Rgl[2] = R[2], Rgl[3] = 0.0;
	Rgl[4] = R[3], Rgl[5] = R[4], Rgl[6] = R[5], Rgl[7] = 0.0;
	Rgl[8] = R[6], Rgl[9] = R[7], Rgl[10] = R[8], Rgl[11] = 0.0;
	Rgl[12] = 0, Rgl[13] = 0, Rgl[14] = 0, Rgl[15] = 1.0;

	mat_mul(iR, T, center, 3, 3, 1);
	C[0] = -center[0], C[1] = -center[1], C[2] = -center[2];
	return;
}
void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, Point3i *AllColor, int npts)
{
	char Fname[200];
	for (int ii = 0; ii < AvailViews.size(); ii++)
		GetRCGL(AllViewParas[AvailViews.at(ii)]);

	sprintf(Fname, "%s/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d ", viewID);
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
	for (int ii = 0; ii < AvailViews.size(); ii++)
		GetRCGL(AllViewParas[AvailViews.at(ii)]);

	sprintf(Fname, "%s/DinfoGL.txt", path);
	FILE *fp = fopen(Fname, "w+");
	for (int ii = 0; ii < AvailViews.size(); ii++)
	{
		int viewID = AvailViews.at(ii);
		fprintf(fp, "%d ", viewID);
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
		while (fscanf(fp, "%d ", &viewID) != EOF)
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
			for (int ii = 0; ii < CorpusData.nCameras; ii++)
			{
				GetRCGL(CorpusData.camera[ii].R, CorpusData.camera[ii].T, CorpusData.camera[ii].Rgl, CorpusData.camera[ii].camCenter);
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
			//abort();
		}
	}

	g_vis.CorpusPointPosition.clear(); g_vis.CorpusPointPosition.reserve(10e5);
	if (drawPointColor)
		g_vis.CorpusPointColor.clear(), g_vis.CorpusPointColor.reserve(10e5);

	Point3i iColor; Point3f fColor; Point3f t3d;
	sprintf(Fname, "%s/Corpus/3dGL.xyz", path); fp = fopen(Fname, "r");
	if (fp == NULL)
		printf("Cannot load %s\n", Fname);
	else
	{
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
		fclose(fp);
	}

	if (g_vis.CorpusPointPosition.size() > 0)
	{
		PointsCentroid[0] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[1] /= g_vis.CorpusPointPosition.size();
		PointsCentroid[2] /= g_vis.CorpusPointPosition.size();

		PointVar[0] = 0.0, PointVar[1] = 0.0, PointVar[2] = 0.0;
		for (int ii = 0; ii < g_vis.CorpusPointPosition.size(); ii++)
		{
			PointVar[0] += pow(g_vis.CorpusPointPosition[ii].x - PointsCentroid[0], 2);
			PointVar[1] += pow(g_vis.CorpusPointPosition[ii].y - PointsCentroid[1], 2);
			PointVar[2] += pow(g_vis.CorpusPointPosition[ii].z - PointsCentroid[2], 2);
		}
		PointVar[0] = sqrt(PointVar[0] / g_vis.CorpusPointPosition.size());
		PointVar[1] = sqrt(PointVar[2] / g_vis.CorpusPointPosition.size());
		PointVar[2] = sqrt(PointVar[2] / g_vis.CorpusPointPosition.size());
	}
	else
		PointsCentroid[0] = PointsCentroid[1] = PointsCentroid[2] = 0;

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
	sprintf(Fname, "%s/Dynamic/GT_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	//sprintf(Fname, "%s/Dynamic/%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	//sprintf(Fname, "%s/Dynamic/3dGL_%d.xyz", path, timeID); FILE *fp = fopen(Fname, "r");
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
		else
			g_vis.PointColor.push_back(Point3f(255, 0, 0));

		if (setCoordinate)
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition.push_back(t3d);
	}
	fclose(fp);

	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition.size(), PointsCentroid[1] /= g_vis.PointPosition.size(), PointsCentroid[2] /= g_vis.PointPosition.size();

	//Concatenate points in case the trajectory mode is used
	if (drawCatTimeVarying3DPoint) //in efficient of there are many points
	{
		if (g_vis.catPointPosition == NULL)
			g_vis.catPointPosition = new vector<Point3d>[g_vis.PointPosition.size()];
		if (timeID == g_vis.catPointPosition[0].size())
			for (int ii = 0; ii < g_vis.PointPosition.size(); ii++)
				g_vis.catPointPosition[ii].push_back(g_vis.PointPosition[ii]);
	}

	return true;
}
bool ReadCurrent3DGL2(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate)
{
	char Fname[200];
	g_vis.PointPosition2.clear(); g_vis.PointPosition2.reserve(10e5);
	if (drawPointColor)
		g_vis.PointColor2.clear(), g_vis.PointColor2.reserve(10e5);

	if (setCoordinate)
		PointsCentroid[0] = 0.0f, PointsCentroid[1] = 0.0f, PointsCentroid[2] = 0.f;
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	sprintf(Fname, "%s/Dynamic/A_%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
	//sprintf(Fname, "%s/3DPoints/%d.txt", path, timeID); FILE *fp = fopen(Fname, "r");
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

		if (setCoordinate)
			PointsCentroid[0] += t3d.x, PointsCentroid[1] += t3d.y, PointsCentroid[2] += t3d.z;
		g_vis.PointPosition2.push_back(t3d);
	}
	fclose(fp);

	if (setCoordinate)
		PointsCentroid[0] /= g_vis.PointPosition2.size(), PointsCentroid[1] /= g_vis.PointPosition2.size(), PointsCentroid[2] /= g_vis.PointPosition2.size();

	//Concatenate points in case the trajectory mode is used
	if (drawCatTimeVarying3DPoint) //in efficient of there are many points
	{
		if (g_vis.catPointPosition2 == NULL)
			g_vis.catPointPosition2 = new vector<Point3d>[g_vis.PointPosition.size()];
		if (timeID == g_vis.catPointPosition2[0].size())
			for (int ii = 0; ii < g_vis.PointPosition2.size(); ii++)
				g_vis.catPointPosition2[ii].push_back(g_vis.PointPosition2[ii]);
	}

	return true;
}
int Read3DTrajectory(char *path, int trialID, bool colorVisibility)
{
	char Fname[200];
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	g_vis.Track3DLength.clear(), g_vis.Traject3D.clear();
	TimeInstancesStack.clear();

	const int nframes = 5000;
	Point3d P3dTemp[nframes];
	double x, y, z, t, timeStamp[nframes];
	int camID[nframes], sortedCamID[nframes], frameID[nframes], dummy[nframes];
	vector<int>AvailCamID, nVisibles, AddedCamID;

	Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
	for (unsigned int i = 0; i <= 255; i++)
		colorMapSource.at<uchar>(i, 0) = i;
	Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);

	maxTime = 0;
	int npts = 0;
	while (true)
	{
		if (trialID == 0)
			sprintf(Fname, "%s/Track3D/OptimizedRaw_Track_%d.txt", path, npts);//sprintf(Fname, "%s/Track3D/%d.txt", path, npts); //
		else if (trialID == 1)
			sprintf(Fname, "%s/Track3D/DCTResampled_Track_%d.txt", path, npts);
		else if (trialID == 2)
			sprintf(Fname, "%s/Track3D/SplineResampled_Track_%d.txt", path, npts);
		else if (trialID == 3)
			sprintf(Fname, "%s/Track3D/frameSynced_Track_%d.txt", path, npts);
		//sprintf(Fname, "%s/OptimizedRaw_Track_%d_%d.txt", path, npts, trialID);

		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			break;
		}
		AvailCamID.clear();
		int count = 0, alreadyAddedCount, cID, fID;
		while (fscanf(fp, "%lf %lf %lf %lf %d %d", &x, &y, &z, &t, &cID, &fID) != EOF)
		{
			timeStamp[count] = t, camID[count] = cID, frameID[count] = fID;
			P3dTemp[count].x = x, P3dTemp[count].y = y, P3dTemp[count].z = z;
			count++;

			alreadyAddedCount = 0;
			for (int ii = 0; ii < (int)AvailCamID.size(); ii++)
				if (AvailCamID[ii] == cID)
					alreadyAddedCount++;

			if (alreadyAddedCount == 0)
				AvailCamID.push_back(cID);

			//Read all the time possible
			bool found = false;
			for (int ii = 0; ii < TimeInstancesStack.size(); ii++)
			{
				if (abs(TimeInstancesStack[ii] - t) < 0.01)
				{
					found = true;
					break;
				}
			}
			if (!found)
				TimeInstancesStack.push_back(t);
		}
		fclose(fp);

		for (int ii = 0; ii < count; ii++)
			dummy[ii] = ii;
		Quick_Sort_Double(timeStamp, dummy, 0, count - 1);

		Trajectory3D *track3D = new Trajectory3D[count];
		for (int ii = 0; ii < count; ii++)
		{
			int id = dummy[ii];
			sortedCamID[ii] = camID[id];
			track3D[ii].timeID = timeStamp[ii];
			track3D[ii].WC = P3dTemp[id], track3D[ii].viewID = camID[id], track3D[ii].frameID = frameID[id];
		}

		//look in a window to determine #camera sees the point at that time instance in the window
		if (colorVisibility)
		{
			int nCams = (int)AvailCamID.size(), minVis = nCams;
			nVisibles.clear();
			for (int jj = 0; jj < nCams / 2; jj++)
			{
				AddedCamID.clear();
				for (int ii = 0; ii < nCams; ii++)
				{
					for (int kk = 0; kk < nCams; kk++)
					{
						if (sortedCamID[jj + ii] == AvailCamID[kk])
						{
							alreadyAddedCount = 0;
							for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
								if (AvailCamID[kk] == AddedCamID[ll])
									alreadyAddedCount++;
							if (alreadyAddedCount == 0)
								AddedCamID.push_back(AvailCamID[kk]);
							break;
						}
					}
				}
				if ((int)AddedCamID.size() < minVis)
					minVis = (int)AddedCamID.size();
				nVisibles.push_back((int)AddedCamID.size());
			}
			for (int jj = nCams / 2; jj < count - nCams / 2; jj++)
			{
				AddedCamID.clear();
				for (int ii = 0; ii < nCams; ii++)
				{
					for (int kk = 0; kk < nCams; kk++)
					{
						if (sortedCamID[jj - nCams / 2 + ii] == AvailCamID[kk])
						{
							alreadyAddedCount = 0;
							for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
								if (AvailCamID[kk] == AddedCamID[ll])
									alreadyAddedCount++;
							if (alreadyAddedCount == 0)
								AddedCamID.push_back(AvailCamID[kk]);
							break;
						}
					}
				}
				if ((int)AddedCamID.size() < minVis)
					minVis = (int)AddedCamID.size();
				nVisibles.push_back((int)AddedCamID.size());
			}
			for (int jj = count - nCams / 2; jj < count; jj++)
			{
				AddedCamID.clear();
				for (int ii = 0; ii < nCams; ii++)
				{
					for (int kk = 0; kk < nCams; kk++)
					{
						if (sortedCamID[jj - nCams + ii] == AvailCamID[kk])
						{
							alreadyAddedCount = 0;
							for (int ll = 0; ll < (int)AddedCamID.size(); ll++)
								if (AvailCamID[kk] == AddedCamID[ll])
									alreadyAddedCount++;
							if (alreadyAddedCount == 0)
								AddedCamID.push_back(AvailCamID[kk]);
							break;
						}
					}
				}
				if ((int)AddedCamID.size() < minVis)
					minVis = (int)AddedCamID.size();
				nVisibles.push_back((int)AddedCamID.size());
			}

			int range = nCams - minVis, nvis;
			for (int ii = 0; ii < count; ii++)
			{
				nvis = nVisibles[ii];
				int colorIdx = (int)(1.0*(nvis - minVis) / (0.01 + range)* 255.0 + 0.5);

				Point3f PointColor;
				PointColor.z = colorMap.at<Vec3b>(colorIdx, 0)[0] / 255.0; //blue
				PointColor.y = colorMap.at<Vec3b>(colorIdx, 0)[1] / 255.0; //green
				PointColor.x = colorMap.at<Vec3b>(colorIdx, 0)[2] / 255.0;	//red
				//RandTrajColor.push_back(PointColor);
				track3D[ii].rgb = PointColor;
			}
		}
		else
		{
			Point3f PointColor(1.0, 0.0, 0.0); //Offset 0.2 so that non of the trajectories can be too dark to be seen
			for (int ii = 0; ii < count; ii++)
				track3D[ii].rgb = PointColor;
			//RandTrajColor.push_back(PointColor);
		}

		g_vis.Track3DLength.push_back(count);
		g_vis.Traject3D.push_back(track3D);

		maxTime = max(maxTime, count);
		npts++;
	}

	//Arrange all time instances in chronological order
	std::sort(TimeInstancesStack.begin(), TimeInstancesStack.end());
	printf("Max time instances: %d\n", TimeInstancesStack.size());

	nTraject = max(nTraject, npts);

	return 0;
}
int Read3DTrajectory2(char *path, int seedID, int trialID)
{
	char Fname[200];
	Point3i iColor; Point3f fColor; Point3f t3d, n3d;
	g_vis.Track3DLength.clear(), g_vis.Traject3D.clear();
	TimeInstancesStack.clear();

	const int nframes = 5000;
	Point3d P3dTemp[nframes];
	double x, y, z;
	vector<int>AvailCamID, nVisibles, AddedCamID;

	Mat colorMapSource = Mat::zeros(256, 1, CV_8U);
	for (unsigned int i = 0; i <= 255; i++)
		colorMapSource.at<uchar>(i, 0) = i;
	Mat colorMap; applyColorMap(colorMapSource, colorMap, COLORMAP_COOL);

	maxTime = 0;
	int pid, fid, nf, npts = 0;
	for (int seedID = 30; seedID <= 60; seedID+=15)
	{
		sprintf(Fname, "%s/Track3D/DynamicFeatures_%d.txt", path, seedID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
			printf("Cannot load %s\n", Fname);
		else
		{
			while (fscanf(fp, "%d %d", &pid, &nf) != EOF)
			{
				Point3f PointColor(1.0, 0.0, 0.0);
				Trajectory3D *track3D = new Trajectory3D[nf];
				for (int jj = 0; jj < nf; jj++)
				{
					fscanf(fp, "%d %lf %lf %lf ", &fid, &x, &y, &z);
					track3D[jj].timeID = fid;
					track3D[jj].WC = Point3d(x, y, z), track3D[jj].frameID = fid;
					track3D[jj].rgb = PointColor;

					//Read all the time possible
					bool found = false;
					for (int ii = 0; ii < TimeInstancesStack.size(); ii++)
					{
						if (abs(TimeInstancesStack[ii] - fid) < 0.01)
						{
							found = true;
							break;
						}
					}
					if (!found)
						TimeInstancesStack.push_back(fid);
				}
				g_vis.Track3DLength.push_back(nf);
				g_vis.Traject3D.push_back(track3D);

				maxTime = max(maxTime, nf + seedID);
				npts++;
			}
		}
	}

	//Arrange all time instances in chronological order
	std::sort(TimeInstancesStack.begin(), TimeInstancesStack.end());
	printf("Max time instances: %d\n", TimeInstancesStack.size());

	nTraject = max(nTraject, npts);

	return 0;
}
int Read3DTrajectoryWithCovariance(char *path, int trialID)
{
	char Fname[200];
	g_vis.Track3DLength3.clear(), g_vis.Traject3D3.clear();

	const int nframes = 5000;
	Point3d P3dTemp[nframes];
	double x, y, z, t, timeStamp[nframes];
	int camID[nframes], frameID[nframes], dummy[nframes];

	maxTime = 0;
	int npts = 0;
	while (true)
	{
		sprintf(Fname, "%s/ATrackMSTD_%d_%d.txt", path, npts, trialID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			break;
		}
		int count = 0, cID, fID;
		while (fscanf(fp, "%lf %lf %lf %lf %d %d", &x, &y, &z, &t, &cID, &fID) != EOF)
		{
			timeStamp[count] = t, camID[count] = cID, frameID[count] = fID;
			P3dTemp[count].x = x, P3dTemp[count].y = y, P3dTemp[count].z = z;
			count++;

			//Read all the time possible
			bool found = false;
			for (int ii = 0; ii < TimeInstancesStack.size(); ii++)
			{
				if (abs(TimeInstancesStack[ii] - t) < 0.01)
				{
					found = true;
					break;
				}
			}
			if (!found)
				TimeInstancesStack.push_back(t);
		}
		fclose(fp);

		for (int ii = 0; ii < count; ii++)
			dummy[ii] = ii;
		Quick_Sort_Double(timeStamp, dummy, 0, count - 1);

		Trajectory3D *track3D = new Trajectory3D[count];
		for (int ii = 0; ii < count; ii++)
		{
			int id = dummy[ii];
			track3D[ii].timeID = timeStamp[ii];
			track3D[ii].WC = P3dTemp[id], track3D[ii].viewID = camID[id], track3D[ii].frameID = frameID[id];
		}

		g_vis.Track3DLength3.push_back(count);
		g_vis.Traject3D3.push_back(track3D);
		maxTime = max(maxTime, count);
		npts++;
	}

	//Arrange all time instances in chronological order
	std::sort(TimeInstancesStack.begin(), TimeInstancesStack.end());

	nTraject = max(nTraject, npts);

	return 0;
}
void ReadCurrentPosesGL(char *path, int nviews, int StartTime, int StopTime)
{
	char Fname[200];
	g_vis.glCameraPoseInfo = new vector<CamInfo>[(StopTime - StartTime + 1)*nviews];

	bool createTimeStack = TimeInstancesStack.size() == 0 ? true : false;

	int timeID;
	CamInfo temp;
	double rt[3], R[9], T[3], Rgl[16], Cgl[3];
	for (int ii = 0; ii < nviews; ii++)
	{
		sprintf(Fname, "%s/CamPose_%d.txt", path, ii);	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		bool firsttime = true;
		int lasAvailFrame = 0;
		while (fscanf(fp, "%d ", &timeID) != EOF)
		{
			if (timeID != 1 && firsttime) //filling the empty space
			{
				for (int jj = lasAvailFrame; jj < timeID - 1; jj++)
				{
					temp.frameID = -1;
					g_vis.glCameraPoseInfo[ii].push_back(temp);
				}
			}

			temp.frameID = timeID;

			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &rt[jj]);

			GetRTFromrt(rt, R, T);
			GetRCGL(R, T, Rgl, Cgl);

			for (int jj = 0; jj < 16; jj++)
				temp.Rgl[jj] = Rgl[jj];
			for (int jj = 0; jj < 3; jj++)
				temp.camCenter[jj] = Cgl[jj];

			if (ii == 0 && createTimeStack)
				TimeInstancesStack.push_back(timeID);

			g_vis.glCameraPoseInfo[ii].push_back(temp);
			lasAvailFrame = timeID;
		}
		fclose(fp);
	}

	return;
}
void ReadCurrentPosesGL2(char *path, int nviews, int StartTime, int StopTime)
{
	char Fname[200];
	g_vis.glCameraPoseInfo2 = new vector<CamInfo>[(StopTime - StartTime + 1)*nviews];

	bool createTimeStack = TimeInstancesStack2.size() == 0 ? true : false;

	int timeID;
	CamInfo temp;
	double rt[3], R[9], T[3], Rgl[16], Cgl[3];
	for (int ii = 0; ii < nviews; ii++)
	{
		sprintf(Fname, "%s/CamPose_%d.txt", path, ii);	FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			printf("Cannot load %s\n", Fname);
			continue;
		}
		bool firsttime = true;
		int lasAvailFrame = 0;
		while (fscanf(fp, "%d ", &timeID) != EOF)
		{
			if (timeID != 1 && firsttime) //filling the empty space
			{
				for (int jj = lasAvailFrame; jj < timeID - 1; jj++)
				{
					temp.frameID = -1;
					g_vis.glCameraPoseInfo2[ii].push_back(temp);
				}
			}

			temp.frameID = timeID;

			for (int jj = 0; jj < 6; jj++)
				fscanf(fp, "%lf ", &rt[jj]);

			GetRTFromrt(rt, R, T);
			GetRCGL(R, T, Rgl, Cgl);

			for (int jj = 0; jj < 16; jj++)
				temp.Rgl[jj] = Rgl[jj];
			for (int jj = 0; jj < 3; jj++)
				temp.camCenter[jj] = Cgl[jj];

			if (ii == 0 && createTimeStack)
				TimeInstancesStack2.push_back(timeID);

			g_vis.glCameraPoseInfo2[ii].push_back(temp);
			lasAvailFrame = timeID;
		}
		fclose(fp);
	}

	return;
}
int screenShot(char *Fname, int width, int height, bool color)
{
	int ii, jj;

	unsigned char *data = new unsigned char[width*height * 4];
	IplImage *cvImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);

	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			cvImg->imageData[3 * ii + 3 * jj*width] = data[3 * ii + 3 * (height - 1 - jj)*width + 2],
			cvImg->imageData[3 * ii + 3 * jj*width + 1] = data[3 * ii + 3 * (height - 1 - jj)*width + 1],
			cvImg->imageData[3 * ii + 3 * jj*width + 2] = data[3 * ii + 3 * (height - 1 - jj)*width];

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