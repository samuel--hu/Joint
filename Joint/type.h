#ifndef TYPE_H
#define TYPE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

struct GlobalParams
{
	static int landmarks;
};

struct BoundingBox
{
	BoundingBox():
		x(0), y(0), w(0), h(0), cx(0), cy(0) {

	}

	double x;
	double y;
	double w;
	double h;
	double cx;
	double cy;
};

struct Sample
{
	Mat_<double> ground_truth_shape;
	Mat_<double> current_shape;
	Mat_<uchar> image;
	bool isFace;
};

#endif