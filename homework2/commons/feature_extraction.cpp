#include <utility>
#include "feature_extraction.h"

ExtractionResult extract(InputArray img, bool single) {
    ExtractionResult extraction_result;
    vector<vector<float> > results;
    vector<Mat> images;

    int _w = 128;
    int _h = 128;
    Mat resized;
    resize(img, resized, Size(_w, _h));

    HOGDescriptor hog_detector;
    hog_detector.winSize = Size(_w, _h);
    hog_detector.cellSize = Size(16, 16);
    hog_detector.blockSize = Size(32, 32);
    hog_detector.blockStride = Size(16, 16);

    vector<float> feats;
    hog_detector.compute(resized, feats, Size(), Size(), vector<Point>());
    results.push_back(feats);
    images.push_back(resized);

    if (single) {
        extraction_result.results = results;
        extraction_result.images = images;
        extraction_result.hog_detector = hog_detector;
        return extraction_result;
    }

    // Rotate 90
    Mat rotated_90;
    rotate(resized, rotated_90, ROTATE_90_CLOCKWISE);
    hog_detector.compute(rotated_90, feats, Size(), Size(), vector<Point>());
    results.push_back(feats);
    images.push_back(rotated_90);

    // Rotate 180
    Mat rotated_180;
    rotate(resized, rotated_180, ROTATE_180);
    hog_detector.compute(rotated_180, feats, Size(), Size(), vector<Point>());
    results.push_back(feats);
    images.push_back(rotated_180);

    // Rotate 270
    Mat rotated_270;
    rotate(resized, rotated_270, ROTATE_90_COUNTERCLOCKWISE);
    hog_detector.compute(rotated_270, feats, Size(), Size(), vector<Point>());
    results.push_back(feats);
    images.push_back(rotated_270);

    extraction_result.results = results;
    extraction_result.images = images;
    extraction_result.hog_detector = hog_detector;
    return extraction_result;
}