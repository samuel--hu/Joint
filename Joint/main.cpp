#include "joint.h"

#pragma comment (lib, "opencv_core2410d.lib")
#pragma comment (lib, "opencv_highgui2410d.lib")
#pragma comment (lib, "opencv_imgproc2410d.lib")
#pragma comment (lib, "opencv_objdetect2410d.lib")

int GlobalParams::n_landmark = 68;
int GlobalParams::n_initial = 5;
int GlobalParams::depth = 5;
double GlobalParams::radius[5] = { 0.4, 0.3, 0.2, 0.15, 0.12 };
int GlobalParams::numFeats[5] = { 500, 500, 500, 300, 300 };

int main() {
	Joint joint;
	joint.loadSample("E://database//lfpw//train.txt");
	return 0;
}