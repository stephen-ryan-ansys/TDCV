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

    vector<Mat> rotations = get_rotations(resized);

    // Rotations
    for (Mat rotation : rotations) {

        // Flips
        vector<Mat> flips = get_flips(rotation);
        for (Mat flip : flips) {

            // // Crops
            // vector<Mat> resized_crops = get_resized_crops(flip);

            // for (Mat resized_crop : resized_crops) {
                vector<float> feats;
                hog_detector.compute(flip, feats, Size(), Size(), vector<Point>());
                results.push_back(feats);
                images.push_back(resized);

                if (single) {
                    extraction_result.results = results;
                    extraction_result.images = images;
                    extraction_result.hog_detector = hog_detector;
                    return extraction_result;
                }
            // }
        }
    }

    extraction_result.results = results;
    extraction_result.images = images;
    extraction_result.hog_detector = hog_detector;
    return extraction_result;
}

vector<Mat> get_rotations(Mat image) {
    // Rotate 90
    Mat rotated_90;
    rotate(image, rotated_90, ROTATE_90_CLOCKWISE);

    // Rotate 180
    Mat rotated_180;
    rotate(image, rotated_180, ROTATE_180);

    // Rotate 270
    Mat rotated_270;
    rotate(image, rotated_270, ROTATE_90_COUNTERCLOCKWISE);

    // Append original
    return {image, rotated_90, rotated_180, rotated_270};
}

vector<Mat> get_flips(Mat image) {
    Mat flipped;
    flip(image, flipped, 0);

    return {image, flipped};
}

vector<Mat> get_resized_crops(Mat image) {
    int w = image.cols;
    int h = image.rows;

    Rect rect1(10, 10, 110, 110);
    Mat resized1;
    resize(image(rect1), resized1, Size(w, h));

    Rect rect2(15, 15, 100, 100);
    Mat resized2;
    resize(image(rect2), resized2, Size(w, h));

    Rect rect3(20, 20, 95, 95);
    Mat resized3;
    resize(image(rect3), resized3, Size(w, h));

    Rect rect4(5, 5, 115, 115);
    Mat resized4;
    resize(image(rect4), resized4, Size(w, h));

    // Rect rect5(7, 7, 112, 112);
    // Mat resized5;
    // resize(image(rect5), resized5, Size(w, h));

    // imshow("test3", resized3);
    // waitKey(0);

    return {image, resized1, resized2, resized3, resized4};
}