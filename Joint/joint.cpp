#include <iostream>
#include <fstream>
#include "joint.h"

Shape Joint::loadShape(const string file) {
	ifstream fin;
	string temp;
	fin.open(file);
	getline(fin, temp);
	getline(fin, temp);
	getline(fin, temp);

	Shape shape(GlobalParams::n_landmark, 2);
	for (int i = 0; i < GlobalParams::n_landmark; ++i) {
		fin >> shape(i, 0) >> shape(i, 1);
	}
	fin.close();

	return shape;
}

bool Joint::belong(Shape &shape, BoundingBox &bb) {
	Point2d leftTop(shape(0, 0), shape(0, 1));
	Point2d rightBottom(shape(0, 0), shape(0, 1));
	double sum_x = 0;
	double sum_y = 0;
	for (int i = 0; i < shape.rows; ++i) {
		double x = shape(i, 0);
		double y = shape(i, 1);
		if (x < leftTop.x) {
			leftTop.x = x;
		}
		else if (x > rightBottom.x) {
			rightBottom.x = x;
		}
		if (y < leftTop.y) {
			leftTop.y = y;
		}
		else if (y > rightBottom.y) {
			rightBottom.y = y;
		}

		sum_x += x;
		sum_y += y;
	}

	double w = rightBottom.x - leftTop.x;
	double h = rightBottom.y - leftTop.y;
	if (w > bb.width * 1.15 || h > bb.height * 1.15) {
		return false;
	}
	if (2 * abs(sum_x / shape.rows - bb.center.x) > bb.width) {
		return false;
	}
	if (2 * abs(sum_y / shape.rows - bb.center.y) > bb.height) {
		return false;
	}
	return true;
}

void Joint::adjust(Image &image, Shape &shape, BoundingBox &bb) {
	Point2d leftTop(max(1.0, bb.center.x - bb.width * 2 / 3),
					max(1.0, bb.center.y - bb.height * 2 / 3));
	Point2d rightBottom(min(image.cols - 1.0, bb.center.x + bb.width),
						min(image.rows - 1.0, bb.center.y + bb.height));

	image = image.rowRange((int)leftTop.y, (int)rightBottom.y).colRange((int)leftTop.x, (int)rightBottom.x).clone();

	bb.x -= leftTop.x;
	bb.y -= leftTop.y;

	bb.center.x -= leftTop.x;
	bb.center.y -= leftTop.y;

	for (int i = 0; i < shape.rows; ++i){
		shape(i, 0) -= leftTop.x;
		shape(i, 1) -= leftTop.y;
	}
}

Shape Joint::Project(const Shape &shape, const BoundingBox &bb) {
	Shape ret(shape.rows, 2);
	for (int i = 0; i < shape.rows; ++i) {
		ret(i, 0) = 2 * (shape(i, 0) - bb.center.x) / bb.width;
		ret(i, 1) = 2 * (shape(i, 1) - bb.center.y) / bb.height;
	}
	return ret;
}

Shape Joint::ReProject(const Shape &shape, const BoundingBox &bb) {
	Shape ret(shape.rows, 2);
	for (int i = 0; i < shape.rows; ++i) {
		ret(i, 0) = shape(i, 0) * bb.width * 0.5 + bb.center.x;
		ret(i, 1) = shape(i, 1) * bb.height * 0.5 + bb.center.y;
	}
	return ret;
}

void Joint::loadSample(const string database) {
	ifstream fin(database);

	CascadeClassifier detector;
	detector.load("haarcascade_frontalface_alt.xml");

	double scale = 1.3;
	vector<Rect> faces;

	string name;
	while (getline(fin, name)) {
		cout << name << endl;
		Image image = imread(name + ".png", 0);
		Shape ground_truth = Joint::loadShape(name + ".pts");

		Image small(round(image.rows / scale), round(image.cols / scale));
		resize(image, small, small.size(), 0, 0, INTER_LINEAR);
		equalizeHist(small, small);
		detector.detectMultiScale(small, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (auto face : faces) {
			BoundingBox bb(face.x * scale, face.y * scale, (face.width - 1) * scale, (face.height - 1) * scale);
			
			if (belong(ground_truth, bb)) {
				adjust(image, ground_truth, bb);
				samples.push_back(Sample(image, ground_truth, bb, true));
				break;
			}
		}
	}
}

void Joint::augment() {
	RNG rng(getTickCount());
	int samples_size = samples.size();
	for (int i = 0; i < samples_size; ++i) {
		for (int j = 0; j < GlobalParams::n_initial; ++j) {
			int index = 0;
			do {
				index = rng.uniform(0, samples_size);
			} while (index == i);

			Shape projected = Joint::Project(samples[index].ground_truth, samples[index].bb);
			samples[i].current = Joint::ReProject(projected, samples[i].bb);

			augmented_samples.push_back(samples[i]);
		}
	}
}

Shape Joint::GetMeanShape(const vector<Sample>& samples) {
	Mat_<double> result = Mat::zeros(samples[0].ground_truth.rows, 2, CV_64FC1);
	int count = 0;
	for (int i = 0; i < samples.size(); i++) {
		if (samples[i].label == 1) {
			result += Project(samples[i].ground_truth, samples[i].bb);
			count++;
		}
	}
	return  1.0 / count * result;
}

Shape Joint::GetShapeResidual(const Sample &sample, const Shape &meanShape) {
	Mat_<double> rotation;
	double scale;
	Mat_<double> residual = Project(sample.ground_truth, sample.bb)
		- Project(sample.current, sample.bb);
	SimilarityTransform(meanShape, Project(sample.current, sample.bb),
		rotation, scale);
	transpose(rotation, rotation);
	return scale * residual * rotation;
}

void Joint::SimilarityTransform(const Shape &shape1, const Shape &shape2,
	Mat_<double>& rotation, double& scale) {
	rotation = Mat::zeros(2, 2, CV_64FC1);
	scale = 0;

	// centering the data
	double center_x1 = 0;
	double center_y1 = 0;
	double center_x2 = 0;
	double center_y2 = 0;

	for (int i = 0; i < shape1.rows; i++) {
		center_x1 += shape1(i, 0);
		center_y1 += shape1(i, 1);
		center_x2 += shape2(i, 0);
		center_y2 += shape2(i, 1);
	}
	center_x1 /= shape1.rows;
	center_x2 /= shape2.rows;
	center_y1 /= shape1.rows;
	center_y2 /= shape1.rows;

	Mat_<double> temp1 = shape1.clone();
	Mat_<double> temp2 = shape2.clone();
	for (int i = 0; i < shape1.rows; i++) {
		temp1(i, 0) -= center_x1;
		temp1(i, 1) -= center_y1;
		temp2(i, 0) -= center_x2;
		temp2(i, 1) -= center_y2;
	}

	Mat_<double> convariance1, convariance2;
	Mat_<double> mean1, mean2;
	// calculate covariance matries

	calcCovarMatrix(temp1, convariance1, mean1, CV_COVAR_COLS);
	calcCovarMatrix(temp2, convariance2, mean2, CV_COVAR_COLS);

	double s1 = sqrt(norm(convariance1));
	double s2 = sqrt(norm(convariance2));
	scale = s1 / s2;
	temp1 = 1.0 / s1 * temp1;
	temp2 = 1.0 / s2 * temp2;

	double num = 0;
	double den = 0;

	for (int i = 0; i < shape1.rows; i++) {
		num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
		den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
	}
	double norm = sqrt(num * num + den * den);
	double sin_theta = num / norm;
	double cos_theta = den / norm;
	rotation(0, 0) = cos_theta;
	rotation(0, 1) = -sin_theta;
	rotation(1, 0) = sin_theta;
	rotation(1, 1) = cos_theta;
}

double Joint::CalculateCovar(const vector<double>& v1, const vector<double>& v2) {
	Mat_<double> v_1(v1);
	Mat_<double> v_2(v2);
	double mean1 = mean(v_1)[0];
	double mean2 = mean(v_2)[0];
	v_1 = v_1 - mean1;
	v_2 = v_2 - mean2;
	return mean(v_1.mul(v_2))[0];
}

double Joint::CalculateVar(const vector<double>& v1) {
	if (v1.size() == 0){
		return 0;
	}
	Mat_<double> v2(v1);
	return CalculateVar(v2);
}

double Joint::CalculateVar(const Mat_<double> & v1) {
	double mean1 = mean(v1)[0];
	double mean2 = mean(v1.mul(v1))[0];
	return mean2 - mean1 * mean1;
}