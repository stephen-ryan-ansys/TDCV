#ifndef FEATURE_EXTRACTION
#define FEATURE_EXTRACTION

#include <opencv2/opencv.hpp>
#include "feature_extraction.h"

using namespace std;
using namespace cv;

struct ExtractionResult {
    vector<vector<float> > results;
    vector<Mat> images;
    HOGDescriptor hog_detector;
};

ExtractionResult extract(InputArray img, bool single=false);

vector<Mat> get_rotations(Mat image);
vector<Mat> get_flips(Mat image);
vector<Mat> get_resized_crops(Mat image);

#endif
