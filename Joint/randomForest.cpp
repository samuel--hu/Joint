//
//  randomForest.cpp
//	
//	implementation for hough forest
//
//	Toxic
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

			// when training of rfs[i][j] is finished 
			// all of the training samples go through the random tree for 
			// classification and regression.
			for (int l = 0; l < samples.size(); l++) {
				rfs[i][j].nodes[0].sample_idx.push_back(i);
			}

			for (int l = 0; l < rfs[i][j].numNodes; l++) {

				// only internal node has children
				if (!rfs[i][j].nodes[l].isLeaf) {
					Node &node = rfs[i][j].nodes[l];
					for (int m = 0; m < node.sample_idx.size(); m++) {
						Mat_<double> rotation;
						double scale;
						Shape temp = Joint::Project(samples[node.sample_idx[m]].current,
							samples[node.sample_idx[m]].bb);
						Joint::SimilarityTransform(temp, meanShape, rotation, scale);
						double x1 = node.feat[0].x * rfs[i][j].radioRadius;
						double y1 = node.feat[0].y * rfs[i][j].radioRadius;
						double x2 = node.feat[1].x * rfs[i][j].radioRadius; 
						double y2 = node.feat[1].y * rfs[i][j].radioRadius;
						double project_x1 = rotation(0, 0) * x1 + rotation(0, 1) * y1;
						double project_y1 = rotation(1, 0) * x1 + rotation(1, 1) * y1;
						project_x1 = scale * project_x1 * samples[node.sample_idx[m]].bb.width / 2.0;
						project_y1 = scale * project_y1 * samples[node.sample_idx[m]].bb.height / 2.0;
						int real_x1 = project_x1 + samples[node.sample_idx[m]].current(j, 0);
						int real_y1 = project_y1 + samples[node.sample_idx[m]].current(j, 1);
						real_x1 = max(0.0,
							min((double)real_x1, samples[node.sample_idx[m]].image.cols - 1.0));
						real_y1 = max(0.0,
							min((double)real_y1, samples[node.sample_idx[m]].image.rows - 1.0));

						double project_x2 = rotation(0, 0) * x2 + rotation(0, 1) * y2;
						double project_y2 = rotation(1, 0) * x2 + rotation(1, 1) * y2;
						project_x2 = scale * project_x2 * samples[node.sample_idx[m]].bb.width / 2.0;
						project_y2 = scale * project_y2 * samples[node.sample_idx[m]].bb.height / 2.0;
						int real_x2 = project_x2 + samples[node.sample_idx[m]].current(j, 0);
						int real_y2 = project_y2 + samples[node.sample_idx[m]].current(j, 1);
						real_x2 = max(0.0,
							min((double)real_x2, samples[node.sample_idx[m]].image.cols - 1.0));
						real_y2 = max(0.0,
							min((double)real_y2, samples[node.sample_idx[m]].image.rows - 1.0));

						// calculate feature value
						int difference = ((int)(samples[node.sample_idx[m]].image(real_y1, real_x1))
							- (int)(samples[node.sample_idx[m]].image(real_y2, real_x2)));
						if (difference < node.threshold) {
							// put this sample into its left child
							rfs[i][j].nodes[node.cNodesID[0]].sample_idx.push_back(m);
						}
						else {
							// put this ample into its fight child
							rfs[i][j].nodes[node.cNodesID[1]].sample_idx.push_back(m);
						}
					}
					// all of the samples in this node have gone to its children node
					node.sample_idx.clear();
				}
			}

			// calculate the classification score of each leaf node
			// add classification score to each sample which fall into
			// corresponding leaf node
			vector<int> &nodeID = rfs[i][j].leafID;
			for (int l = 0; l < nodeID.size(); l++) {
				Node& temp = rfs[i][j].nodes[nodeID[l]];
				double positive = 0;
				double negative = 0;
				for (int m = 0; m < temp.sample_idx.size(); m++) {
					if (samples[temp.sample_idx[m]].label == 1) {
						positive += samples[temp.sample_idx[m]].weight;
					} 
					else {
						negative += samples[temp.sample_idx[m]].weight;
					}
				}
				// calculate score
				double score = 1 / 2 * log(positive / negative);
				for (int m = 0; m < temp.sample_idx.size(); m++) {
					samples[temp.sample_idx[m]].score += score;
				}
				temp.sample_idx.clear();
			}

			// delete samples acrroding to classificationscore 
			// and reall ratio 
			vector<Sample> samples_sorted = samples;
			std::sort(samples_sorted.begin(), samples_sorted.end(), Joint::ScoreAscending);
			int posCount;
			for (int l = 0; l < samples.size(); l++) {
				if (samples[l].label == 1) {
					posCount++;
				}
			}
			int cutCount = posCount * GlobalParams::recall;
			int tempPos = 0;
			double scoreThresh;
			for (int l = samples_sorted.size() - 1; l >= 0; l--) {
				if (tempPos == cutCount) {
					scoreThresh = samples_sorted[l].score;
					break;
				}
				else if (samples_sorted[l].label == 1) {
					tempPos++;
				}
			}
			vector<Sample> newSamples;
			for (int l = 0; l < samples.size(); l++) {
				if (samples[l].score >= scoreThresh) {
					newSamples.push_back(samples[i]);
				}
			}
			samples = newSamples;
		}
	}
}