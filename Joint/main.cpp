#include "crf.h"

int main() {
	//set global parameters
	GlobalParams::landmarks = 27;

	//train a model
	CRF crf;
	crf.loadSamples("lfpw");
	crf.train();
	crf.save("model");

	return 0;
}