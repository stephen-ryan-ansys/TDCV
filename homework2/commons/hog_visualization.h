#ifndef HOG_VISUALIZATION
#define HOG_VISUALIZATION

#include <opencv2/opencv.hpp>

void visualizeHOG(cv::Mat img, std::vector<float> &feats, const cv::HOGDescriptor &hog_detector, int scale_factor = 3);

#endif
