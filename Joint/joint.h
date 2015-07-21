#ifndef JOINT_H
#define JOINT_H

#include <vector>
#include <string>
#include "tree.h"

using namespace std;
using namespace cv;

struct GlobalParames
{
	static int n_landmark;
};

class Joint
{
public:

	void loadSample(const string database);

public:

	static Shape loadShape(const string file);
	static bool belong(Shape &shape, BoundingBox &bb);
	static void adjust(Image &image, Shape &shape, BoundingBox &bb);

public:

	vector<Sample> samples;
};

#endif