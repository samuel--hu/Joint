#ifndef JOINT_H
#define JOINT_H

#include <vector>
#include <string>
#include "sample.h"

using namespace std;
using namespace cv;

struct GlobalParams
{
	static int n_landmark;
	static int n_initial;

	static double overlap;
	// recall ratio
	static double recall;
	// total stages 
	static int stages;

	static double radius[5];
	static int numFeats[5];
	static int depth;
	static int numTrees;

};

class Joint
{
public:

	void loadSample(const string database);
	void augment();

public:

	static Shape loadShape(const string file);
	static bool belong(Shape &shape, BoundingBox &bb);
	static void adjust(Image &image, Shape &shape, BoundingBox &bb);

	static Shape Project(const Shape &shape, const BoundingBox &bb);
	static Shape ReProject(const Shape &shape, const BoundingBox &bb);

	static Shape GetMeanShape(const vector<Sample> &samples);
	static Shape GetShapeResidual(const Sample &sample, const Shape &meanShape);
	static void SimilarityTransform(const Mat_<double> &shape1, const Mat_<double> &shape2, Mat_<double>& rotation, double& scale);

	static double CalculateCovar(const vector<double>& v1, const vector<double>& v2);
	static double CalculateVar(const vector<double>& v1);
	static double CalculateVar(const Mat_<double> & v1);

public:

	vector<Sample> samples;
	vector<Sample> augmented_samples;
};

#endif