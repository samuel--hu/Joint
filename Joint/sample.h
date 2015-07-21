#ifndef SAMPLE_H
#define SAMPLE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

typedef cv::Mat_<double> Shape;
typedef cv::Mat_<uchar> Image;

struct BoundingBox : public cv::Rect_<double>
{
	BoundingBox(double x, double y, double w, double h) :
		cv::Rect_<double>(x, y, w, h) {
		center.x = x + w * 0.5;
		center.y = y + h * 0.5;
	}

	BoundingBox():
		cv::Rect_<double>(0, 0, 0, 0) {
		center.x = 0;
		center.y = 0;
	}

	cv::Point2d center;
};

struct Sample
{
	Sample(Image &image, Shape &ground_truth, BoundingBox &bb, int label) :
		image(image), ground_truth(ground_truth), bb(bb), label(label) {

	}

	Sample() {

	}

	int label;

	Shape ground_truth;
	Shape current;
	Image image;

	BoundingBox bb;

	double score;
	double weight;
};

#endif