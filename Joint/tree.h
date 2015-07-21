#ifndef TREE_H
#define TREE_H

#include "sample.h"
#include "joint.h"

class Node
{
public:
	Node()
	{
		sample_idx.clear();
		isSplit = false;
		pNodeID = 0;
		depth = 0;
		cNodesID[0] = 0;
		cNodesID[1] = 0;
		threshold = 0;
		isLeaf = true;
	}

	bool isSplit;
	int pNodeID;
	int depth;
	int cNodesID[2];
	bool isLeaf;
	//threshold;
	double threshold;
	cv::Point2d feat[2];

	//samples' index
	std::vector<int> sample_idx;
};

enum spliteType { CLASSIFICATION, REGRESSION };

class Tree
{
public:
	// ID of landmarks
	int landmarkID;
	// depth of the tree
	int maxDepth;
	// number of maximum nodes
	int maxNumNodes;
	// number of leaf nodes
	int numLeafNodes;
	// number of nodes
	int numNodes;

	// number of pixel features
	int numFeats;
	// max radius of local coordinates
	double radioRadius;
	// index of leaves
	std::vector<int> leafID;
	// tree nodes
	std::vector<Node> nodes;

	Tree() {
		maxDepth = GlobalParams::depth;
		maxNumNodes = pow(2, maxDepth) - 1;
		nodes.reserve(maxNumNodes);
	}

	void SplitNode(
		const spliteType &sType,
		const std::vector<Sample> &samples,
		const cv::Mat_<double> &meanShape,
		const std::vector<int> &sample_idx,
		// output
		double &threshold,
		cv::Point2d* feat,
		std::vector<int> &lcID,
		std::vector<int> &rcID
		);

	void Train(std::vector<Sample> &samples,
		const cv::Mat_<double> &meanShape,
		int stages,
		int landmarkID);
};


#endif