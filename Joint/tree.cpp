#include "tree.h"

void Tree::Train(vector<Sample> &samples,
	const Mat_<double> &meanShape,
	int stages_,
	int landmarkID_
	) {
	// set parameters 
	landmarkID = landmarkID_;
	numFeats = GlobalParams::numFeats[stages_];
	radioRadius = GlobalParams::radius[stages_];
	numNodes = 1;
	numLeafNodes = 1;

	// index: indicates the training samples id in training data set
	int num_nodes_iter;
	int num_split;
	for (int i = 0; i < samples.size(); i++) {
		
		// push the indies of training samples into root node
		nodes[0].sample_idx.push_back(i);
	}

	// initialize the root
	nodes[0].isSplit = false;
	nodes[0].pNodeID = 0;
	nodes[0].depth = 1;
	nodes[0].cNodesID[0] = 0;
	nodes[0].cNodesID[1] = 0;
	nodes[0].isLeaf = true;
	nodes[0].threshold = 0;
	nodes[0].feat[0].x = 1;
	nodes[0].feat[0].y = 1;
	nodes[0].feat[1].x = 1;
	nodes[0].feat[1].y = 1;

	bool stop = false;
	int num_nodes = 1;
	int num_leafnodes = 1;
	double thresh;
	Point2d feat[2];

	vector<int> lcID, rcID;
	lcID.reserve(nodes[0].sample_idx.size());
	rcID.reserve(nodes[0].sample_idx.size());
	while (!stop) {
		num_nodes_iter = num_nodes;
		num_split = 0;
		for (int n = 0; n < num_nodes_iter; n++) {
			if (!nodes[n].isSplit) {
				if (nodes[n].depth == maxDepth) {
					nodes[n].isSplit = true;
				}
			}
			else {
				// separate the training samples into left and right path
				// splite the tree
				// In each internal node, we randomly choose to either minimize the 
				// binary entropy for classification (with probablity p)
				// or the variance of ficial point increments for regression
				// (with probability 1-p)
				RNG randonGenerator(getTickCount());
				double p = 1 - 0.1 * stages_;
				double val = randonGenerator.uniform(0.0, 1.0);
				if (val <= p) {
					SplitNode(CLASSIFICATION, samples, meanShape, nodes[n].sample_idx,
						thresh, feat, lcID, rcID);
				}
				else {
					SplitNode(REGRESSION, samples, meanShape, nodes[n].sample_idx,
						thresh, feat, lcID, rcID);
				}

				// set the threshold and feature for current node
				nodes[n].feat[0] = feat[0];
				nodes[n].feat[1] = feat[1];
				nodes[n].threshold = thresh;
				nodes[n].isSplit = true;
				nodes[n].isLeaf = false;
				nodes[n].cNodesID[0] = num_nodes;
				nodes[n].cNodesID[1] = num_nodes + 1;

				// add left and right child into the random tree
				nodes[num_nodes].sample_idx = lcID;
				nodes[num_nodes].isSplit = false;
				nodes[num_nodes].pNodeID = n;
				nodes[num_nodes].depth = nodes[n].depth + 1;
				nodes[num_nodes].cNodesID[0] = 0;
				nodes[num_nodes].cNodesID[1] = 0;
				nodes[num_nodes].isLeaf = true;

				nodes[num_nodes + 1].sample_idx = rcID;
				nodes[num_nodes + 1].isSplit = false;
				nodes[num_nodes + 1].pNodeID = n;
				nodes[num_nodes + 1].depth = nodes[n].depth + 1;
				nodes[num_nodes + 1].cNodesID[0] = 0;
				nodes[num_nodes + 1].cNodesID[1] = 0;
				nodes[num_nodes + 1].isLeaf = true;

				num_split++;
				num_leafnodes++;
				num_nodes += 2;
			}
		}
		if (num_split == 0) {
			stop = 1;
		}
		else {
			numNodes = num_nodes;
			numLeafNodes = num_leafnodes;
		}
	}

	// mark leaf nodes.
	// clear sample indices in each node
	leafID.clear();
	for (int i = 0; i < numNodes; i++) {
		nodes[i].sample_idx.clear();
		if (nodes[i].isLeaf) {
			leafID.push_back(i);
		}
	}
}
void Tree::SplitNode(
	const spliteType& sType,
	const vector<Sample>& samples,
	const Mat_<double>& meanShape,
	const vector<int>& sample_idx,
	// output
	double& threshold,
	Point2d* feat,
	vector<int> &lcID,
	vector<int> &rcID
	) {
	if (sample_idx.size() == 0) {
		threshold = 0;
		feat = new Point2d[2];
		feat[0].x = 0;
		feat[0].y = 0;
		feat[1].x = 0;
		feat[1].y = 0;
		lcID.clear();
		rcID.clear();
		return;
	}

	RNG randomGenerator(getTickCount());
	Mat_<double> candidatePixelLocations(numFeats, 4);
	for (int i = 0; i < candidatePixelLocations.rows; i++) {
		double x1 = randomGenerator.uniform(-1.0, 1.0);
		double x2 = randomGenerator.uniform(-1.0, 1.0);
		double y1 = randomGenerator.uniform(-1.0, 1.0);
		double y2 = randomGenerator.uniform(-1.0, 1.0);
		if ((x1 * x1 + y1 * y1 > 1.0) || (x2 * x2 + y2 * y2 > 1.0)) {
			i--;
			continue;
		}
		candidatePixelLocations(i, 0) = x1 * radioRadius;
		candidatePixelLocations(i, 1) = y1 * radioRadius;
		candidatePixelLocations(i, 2) = x2 * radioRadius;
		candidatePixelLocations(i, 3) = y2 * radioRadius;
	}

	// get pixel difference features
	Mat_<int> densities(numFeats, (int)sample_idx.size());
	for (int i = 0; i < sample_idx.size(); i++) {
		Mat_<double> rotation;
		double scale;
		Mat_<double> temp = Joint::Project(samples[sample_idx[i]].current,
			samples[sample_idx[i]].bb);
		Joint::SimilarityTransform(temp, meanShape, rotation, scale);

		for (int j = 0; i < numFeats; i++) {
			double project_x1 = rotation(0, 0) * candidatePixelLocations(j, 0)
				+ rotation(0, 1) * candidatePixelLocations(j, 1);
			double project_y1 = rotation(1, 0) * candidatePixelLocations(j, 0)
				+ rotation(1, 1) * candidatePixelLocations(j, 1);
			project_x1 = scale * project_x1 * samples[sample_idx[i]].bb.width / 2.0;
			project_y1 = scale * project_y1 * samples[sample_idx[i]].bb.height / 2.0;
			int real_x1 = scale * project_x1 + samples[sample_idx[i]].current(landmarkID, 0);
			int real_y1 = scale * project_y1 + samples[sample_idx[i]].current(landmarkID, 1);
			real_x1 = max(0.0, min((double)real_x1, samples[sample_idx[i]].image.cols - 1.0));
			real_y1 = max(0.0, min((double)real_y1, samples[sample_idx[i]].image.rows - 1.0));

			double project_x2 = rotation(0, 0) * candidatePixelLocations(j, 2)
				+ rotation(0, 1) * candidatePixelLocations(j, 3);
			double project_y2 = rotation(1, 0) * candidatePixelLocations(j, 2)
				+ rotation(1, 1) * candidatePixelLocations(j, 3);
			project_x2 = scale * project_x2 * samples[sample_idx[i]].bb.width / 2.0;
			project_y2 = scale * project_y2 * samples[sample_idx[i]].bb.height / 2.0;
			int real_x2 = scale * project_x2 + samples[sample_idx[i]].current(landmarkID, 0);
			int real_y2 = scale * project_y2 + samples[sample_idx[i]].current(landmarkID, 1);
			real_x2 = max(0.0, min((double)real_x2, samples[sample_idx[i]].image.cols - 1.0));
			real_y2 = max(0.0, min((double)real_y2, samples[sample_idx[i]].image.rows - 1.0));

			densities(j, i) = ((int)(samples[sample_idx[i]].image(real_y1, real_x1))
				- (int)(samples[sample_idx[i]].image(real_y2, real_x2)));
		}
	}

	// pick the feature
	Mat_<int> densities_sorted = densities.clone();
	cv::sort(densities, densities_sorted, CV_SORT_ASCENDING);

	if (sType == CLASSIFICATION) {
		// classification node
		double min_entropy = INT_MAX;
		double tempThresh = 0;
		double pos_pos_count = 0;
		double neg_pos_count = 0;
		double pos_neg_count = 0;
		double neg_neg_count = 0;
		double entropy = 0;
		double min_id;
		for (int i = 0; i < numFeats; i++) {
			int ind = (int)(sample_idx.size() * randomGenerator.uniform(0.05, 0.95));
			tempThresh = densities_sorted(i, ind);
			for (int j = 0; j < sample_idx.size(); j++) {
				if (densities(i, j) < tempThresh) {
					if (samples[sample_idx[j]].label == -1) {
						neg_neg_count++;
					}
					else {
						neg_pos_count++;
					}
				}
				else {
					if (samples[sample_idx[j]].label == 1) {
						pos_pos_count++;
					}
					else {
						pos_neg_count++;
					}
				}
			}
			double p1 = (double)pos_pos_count / (pos_pos_count + pos_neg_count);
			double p2 = (double)pos_neg_count / (pos_pos_count + pos_neg_count);
			double p3 = (double)neg_pos_count / (neg_pos_count + neg_neg_count);
			double p4 = (double)neg_neg_count / (neg_pos_count + neg_neg_count);
			entropy = p1 * log(p1) + p2 * log(p2) + p3 * log(p3) + p4 * log(p4);

			if (entropy < min_entropy) {
				threshold = tempThresh;
				min_id = i;
			}
		}
		feat[0].x = candidatePixelLocations(min_id, 0) / radioRadius;
		feat[0].y = candidatePixelLocations(min_id, 1) / radioRadius;
		feat[1].x = candidatePixelLocations(min_id, 2) / radioRadius;
		feat[1].y = candidatePixelLocations(min_id, 3) / radioRadius;
		lcID.clear();
		rcID.clear();
		for (int j = 0; j < sample_idx.size(); j++) {
			if (densities(min_id, j) < threshold) {
				lcID.push_back(sample_idx[j]);
			}
			else {
				rcID.push_back(sample_idx[j]);
			}
		}
	}
	else if (sType == REGRESSION) {
		Mat_<double> shape_residual((int)sample_idx.size(), 2);
		int posCount = 0;
		for (int i = 0; i < sample_idx.size(); i++) {
			if (samples[sample_idx[i]].label == 1) {
				Mat_<double> residual = Joint::GetShapeResidual(samples[sample_idx[i]], meanShape);
				shape_residual(i, 0) = residual(landmarkID, 0);
				shape_residual(i, 1) = residual(landmarkID, 1);
				posCount++;
			}
		}

		vector<double> lc1, lc2;
		vector<double> rc1, rc2;
		lc1.reserve(sample_idx.size());
		lc2.reserve(sample_idx.size());
		rc1.reserve(sample_idx.size());
		rc2.reserve(sample_idx.size());

		double varOverall = (Joint::CalculateVar(shape_residual.col(0))
			+ Joint::CalculateVar(shape_residual.col(1))) * posCount;
		double maxVarReduction = 0;
		double tempThresh;
		double varLeft = 0;
		double varRight = 0;
		double varReduction = 0;
		double max_id = 0;
		for (int i = 0; i < numFeats; i++) {
			lc1.clear();
			lc2.clear();
			rc1.clear();
			rc2.clear();
			int ind = (sample_idx.size() * randomGenerator.uniform(0.05, 0.95));
			tempThresh = densities_sorted(i, ind);
			for (int j = 0; j < sample_idx.size(); j++) {
				if (samples[sample_idx[i]].label == 1) {
					if (densities(i, j) < tempThresh) {
						lc1.push_back(shape_residual(j, 0));
						lc2.push_back(shape_residual(j, 1));
					}
					else {
						rc1.push_back(shape_residual(j, 0));
						rc2.push_back(shape_residual(j, 1));
					}
				}

			}
			varLeft = (Joint::CalculateVar(lc1) + Joint::CalculateVar(lc2)) * lc1.size();
			varRight = (Joint::CalculateVar(rc1) + Joint::CalculateVar(rc2)) * rc2.size();
			varReduction = varOverall - varLeft - varRight;
			if (varReduction > maxVarReduction) {
				maxVarReduction = varReduction;
				threshold = tempThresh;
				max_id = i;
			}
		}
		feat[0].x = candidatePixelLocations(max_id, 0) / radioRadius;
		feat[0].y = candidatePixelLocations(max_id, 1) / radioRadius;
		feat[1].x = candidatePixelLocations(max_id, 2) / radioRadius;
		feat[1].y = candidatePixelLocations(max_id, 3) / radioRadius;

		lcID.clear();
		rcID.clear();
		for (int j = 0; j < sample_idx.size(); j++) {
			if (densities(max_id, j) < threshold) {
				lcID.push_back(sample_idx[j]);
			}
			else {
				rcID.push_back(sample_idx[j]);
			}
		}
	}
}