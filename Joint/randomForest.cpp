//
//  randomForest.cpp
//	
//	implementation of hough forest
//
//	Toixc
//

#include "randomforest.h"

void RandomForest::Train(
	vector<Sample> &samples,
	const Shape &meanShape,
	int stages_
	) {
	stages = stages_;
	RNG randomGenerator(getTickCount());
	vector<int> index;
	bool* key;
	for (int i = 0; i < numTrees; i++) {
		for (int j = 0; j < numLandmarks; j++) {
			
			for (int l = 0; l < samples.size(); l++) {
				// update weight of each sample
				samples[i].weight = exp(-samples[i].label * samples[i].weight);
			}

			// randomly select 2000 different samples
			index.clear();
			key = new bool[samples.size()];
			memset(key, false, sizeof(bool)* samples.size());
			int k = randomGenerator.uniform(0, (int)samples.size());
			key[k] = true;
			index.push_back(k);
			while (index.size() < 2000) {
				k = randomGenerator.next();
				if (key[k] == false) {
					index.push_back(k);
					key[k] = true;
				}
			}
			delete [] key;
			key = nullptr;
			vector<Sample> tempSamples;
			for (int n = 0; n < index.size(); n++) {
				tempSamples.push_back(samples[index[n]]);
			}
			rfs[i][j].Train(tempSamples, meanShape, stages, j);

			
		}
	}
}