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

Shape Joint::project(const Shape &shape, const BoundingBox &bb) {
	Shape ret(shape.rows, 2);
	for (int i = 0; i < shape.rows; ++i) {
		ret(i, 0) = 2 * (shape(i, 0) - bb.center.x) / bb.width;
		ret(i, 1) = 2 * (shape(i, 1) - bb.center.y) / bb.height;
	}
	return ret;
}

Shape Joint::re_project(const Shape &shape, const BoundingBox &bb) {
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

			Shape projected = Joint::project(samples[index].ground_truth, samples[index].bb);
			samples[i].current = Joint::re_project(projected, samples[i].bb);

			augmented_samples.push_back(samples[i]);
		}
	}
}