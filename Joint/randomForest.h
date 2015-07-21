// randomForest.h
//
// Hough Forest
//
// created by Toxic
//

#ifndef RANDOMFOREST_H
#define RANDOMFOREST_H

#include "tree.h"

class RandomForest {
public:
	std::vector<std::vector<Tree>> rfs;
	// number of trees
	int numTrees;
	// number of landmarks
	int numLandmarks;
	// max depth of tree
	int maxDepth;
	// training stage
	int stages;

	RandomForest() {
		numTrees = GlobalParams::numTrees;
		numLandmarks = GlobalParams::n_landmark;
		maxDepth = GlobalParams::depth;

		// resize the random forest
		rfs.resize(numTrees);
		for (int i = 0; i < numTrees; i++) {
			rfs.resize(numLandmarks);
		}
	}
	
	void Train(vector<Sample>& samples, 
		const Shape& meanShape,
		int stages);
};
#endif