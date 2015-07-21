#include "joint.h"

#pragma comment (lib, "opencv_core2410d.lib")
#pragma comment (lib, "opencv_highgui2410d.lib")
#pragma comment (lib, "opencv_imgproc2410d.lib")
#pragma comment (lib, "opencv_objdetect2410d.lib")

int GlobalParames::n_landmark = 68;

int main() {
	Joint joint;
	joint.loadSample("E://database//lfpw//train.txt");
	return 0;
}