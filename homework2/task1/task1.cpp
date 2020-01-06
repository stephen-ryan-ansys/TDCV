#include <cstdio>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>
#include <utility>

#include "../commons/feature_extraction.h"
#include "../commons/hog_visualization.h"

int main() {
    char const *file = "data/task1/%s";

    Mat image;
    image = imread(format(file, "obj1000.jpg").c_str(), 1);
    printf("%d %d\n", image.size().width, image.size().height);
    imshow("original", image);
    waitKey(0);

    // Extract feature for image
    ExtractionResult extraction_result = extract(image);

    vector<vector<float> > descriptors = extraction_result.results;
    HOGDescriptor& hog_detector = extraction_result.hog_detector;
    vector<Mat> images = extraction_result.images;

    printf("%zu\n", descriptors.size());

    if (descriptors.size() > 0) {
        vector<float> sample_feats = descriptors.at(0);
        Mat sample_image = images.at(0);
        visualizeHOG(sample_image, sample_feats, hog_detector);
    } else {
        printf("No feature extracted!");
    }

    Mat padded;
    copyMakeBorder(image, padded, 12, 11, 0, 1, BORDER_REPLICATE);
    imwrite(format(file, "obj1000_padded.jpg"), padded);

    Mat gray;
    cvtColor(padded, gray, COLOR_BGR2GRAY);
    imwrite(format(file, "obj1000_gray.jpg"), gray);

    Mat resized;
    resize(padded, resized, Size(), 0.5, 0.5, 1);
    imwrite(format(file, "obj1000_resized.jpg"), resized);

    Mat rotated;
    rotate(padded, rotated, ROTATE_90_COUNTERCLOCKWISE);
    imwrite(format(file, "obj1000_rotated.jpg"), rotated);

    Mat flipped;
    flip(padded, flipped, 0);
    imwrite(format(file, "obj1000_flipped.jpg"), flipped);

    return 0;
}