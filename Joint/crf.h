#ifndef CRF_H
#define CRF_H

#include "type.h"

class Node
{
public:
	Node() :
		left(NULL), right(NULL), threshold(0) {

	}

	virtual void split() = 0;

	double threshold;
	Point2d feat[2];
	Node *right;
	Node *left;
};

class ClassificationNode : public Node
{
public:
	ClassificationNode() :
		Node() {

	}

	void split();
};

class RegressionNode : public Node
{
public:
	RegressionNode() :
		Node() {

	}

	void split();
};

class CRF
{
public:

	void loadSamples(string filePath);
	void train();
	void save(string filePath);
	void read(string filePath);

	vector<Sample> samples;
};

#endif
