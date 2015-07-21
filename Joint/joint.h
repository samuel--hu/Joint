#ifndef JOINT_H
#define JOINT_H

#include <vector>
#include <string>
#include "tree.h"

using namespace std;
using namespace cv;

struct GlobalParams
{
	static int n_landmark;
	static int n_initial;
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

	static Shape project(const Shape &shape, const BoundingBox &bb);
	static Shape re_project(const Shape &shape, const BoundingBox &bb);

public:

	vector<Sample> samples;
	vector<Sample> augmented_samples;
};

#endif