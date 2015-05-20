#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>
#include "GL\glut.h"
#include "DataStructure.h"
#include "Ultility.h"

using namespace cv;
using namespace std;

#if !defined( VISUALIZATION_H )
#define VISUALIZATION_H

#define VIEWING_DISTANCE_MIN  1.0

void Draw_Axes();
void DrawCamera();
void RenderObjects();
void display(void);
void InitGraphics(void);
void Keyboard(unsigned char key, int x, int y);
void MouseButton(int button, int state, int x, int y);
void MouseMotion(int x, int y);
void ReshapeGL(int width, int height);

void visualization();
int visualizationDriver(char *inPath, int nViews, int StartTime, int StopTime, bool hasColor, bool hasPatchNormal, bool hasTimeVaryingCameraPose, bool hasTimeVarying3DPoints, bool hasFullTrajectory, int CurrentTime);

void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, Point3d *All3D, Point3i *AllColor, int npts);
void SaveCurrentSfmGL(char *path, CameraData *AllViewParas, vector<int>AvailViews, vector<Point3d>All3D, vector<Point3i>AllColor);
void ReadCurrentSfmGL(char *path, bool hasColor, bool hasNormal);
bool ReadCurrent3DGL(char *path, bool hasColor, bool hasNormal, int timeID, bool setCoordinate);
bool ReadCurrent3DGL2(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate);
bool ReadCurrent3DGL3(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate);
int ReadCurrentTrajectory(char *path, int timeID);
int ReadCurrentTrajectory2(char *path, int timeID);
int ReadCurrentTrajectoryWithCovariance(char *path, int timeID);

void ReadCurrentPosesGL(char *path, int nviews, int StartTime, int StopTime);
int screenShot(char *Fname, int width, int height, bool color);


bool ReadDomePointCloud(char *path, bool drawPointColor, bool drawPatchNormal, int timeID, bool setCoordinate);
#endif


