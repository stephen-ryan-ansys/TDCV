#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "../commons/feature_extraction.h"
#include "../task2/RandomForest.h"

using namespace cv;
using namespace cv::ml;


struct BoundingBox {
    int x1,y1,x2,y2;
};

struct DetectedObject {
    BoundingBox bounding_box;
    float confidence;
    int prediction;
};

int is_regular_file(const char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISREG(path_stat.st_mode);
}

vector<Mat> get_images(string path) {
    vector<Mat> images;
    DIR * dirp = opendir(path.c_str());
    dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        string full_path = path + dp->d_name;
        if (is_regular_file(full_path.c_str())) {
            Mat image = imread(full_path.c_str(), 1);
            images.push_back(image);
        }
    }
    (void)closedir(dirp);

    return images;
}

Ptr<TrainData> generate_data(string base_path) {
    vector<string> dirs = {"00", "01", "02", "03"};
    vector<int> categories = {0, 1, 2, 3};

    Mat input;
    Mat ground_truth;
    for (int i = 0;i < dirs.size(); i++) {
        string class_path = base_path + dirs.at(i) + "/";
        cout << "Generating data for folder: " + class_path << endl;
        vector<Mat> images = get_images(class_path);
        for (Mat image : images) {
            ExtractionResult extraction_result = extract(image);
            vector<vector<float> > features = extraction_result.results;

            int category = categories.at(i);
            for (int j = 0;j < features.size(); j++) {
                Mat feature = Mat(features.at(j)).reshape(1, 1);
                feature.convertTo(feature, CV_32F);
                input.push_back(feature);
                ground_truth.push_back(category);
            }
        }
    }

    return TrainData::create(input, ROW_SAMPLE, ground_truth);
}


void visualize_detected_objects(vector<DetectedObject> detected_objects, Mat image) {
    for (DetectedObject &d : detected_objects) {
        BoundingBox b = d.bounding_box;
        Rect rect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

        rectangle(image, rect, Scalar(0, 255, 0));
        // break;
    }

    imshow("boxes", image);
    waitKey(0);
}


double IOU(const BoundingBox &box1, const BoundingBox &box2) {
    int xA = std::max(box1.x1, box2.x1);
    int yA = std::max(box1.y1, box2.y1);
    int xB = std::min(box1.x2, box2.x2);
    int yB = std::min(box1.y2, box2.y2);
    double inter_area = max(0, xB-xA + 1) * max(0, yB-yA + 1);
    int box1_area = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    int box2_area = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);

    double iou = inter_area/(box1_area + box2_area - inter_area);
    return iou;
}


vector<DetectedObject> non_maximum_supression(vector<DetectedObject> detected_objects, double thresh) {
    vector<DetectedObject> non_supressed;

    for (DetectedObject &d : detected_objects) {
        bool is_max_confidence = true;
        for (DetectedObject &other : detected_objects) {
            if (d.prediction != other.prediction) {
                continue;
            }
            double iou = IOU(d.bounding_box, other.bounding_box);
            // cout << "iou " << iou << ", other conf: " << other.confidence << ", d.conf: " << d.confidence << endl;
            bool overlapping = iou > thresh;
            if (other.confidence > d.confidence && overlapping) {
                is_max_confidence = false;
                break;
            }
        }
        if (is_max_confidence) {
            // cout << "detected a box for something" << endl;
            non_supressed.push_back(d);
        }
    }

    return non_supressed;
}


vector<BoundingBox> get_bounding_boxes(Mat image) {
    vector<BoundingBox> bounding_boxes;
    int rows = image.rows;
    int cols = image.cols;
    int slide = 2;
    
    for (int size = 80; size <= 80; size+= 2) {
        for (int i = 0; i <= rows - size; i+= slide) {
            for (int j = 0; j <= cols - size; j+= slide) {
                BoundingBox bounding_box = {j, i, j+size, i+size};
                bounding_boxes.push_back(bounding_box);
            }
        }
    }

    return bounding_boxes;
}


vector<DetectedObject> detect_object(Mat image, Ptr<RandomForest> forest) {
    vector<DetectedObject> detected_objects;
    cout << "Generating bounding boxes..." << endl;

    vector<BoundingBox> bounding_boxes = get_bounding_boxes(image);

    cout << "Extracting features for " << bounding_boxes.size() << " boxes..." << endl;
    Mat input;
    for (BoundingBox &bounding_box : bounding_boxes) {
        int x = bounding_box.x1;
        int y = bounding_box.y1;
        int width = bounding_box.x2 - x;
        int height = bounding_box.y2 - y;
        // cout << x << " " << y << " " << width << " " << height << endl;
        Rect rect(x, y, width, height);
        Mat cropped = image(rect);
        
        vector<vector<float> > features = extract(cropped, true).results;
        Mat feature = Mat(features.at(0)).reshape(1, 1);
        feature.convertTo(feature, CV_32F);

        input.push_back(feature);
    }

    cout << input.size() << endl;

    forest->predict(input);
    vector<double> confidences = forest->getConfidences();
    vector<int> predictions = forest->getPredictions();
    vector<BoundingBox> chosen;
    for (int i = 0; i < bounding_boxes.size(); i++) {
        BoundingBox b = bounding_boxes.at(i);
        double c = confidences.at(i);
        int p = predictions.at(i);
        // cout << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2 << " " << c << endl;

        // TODO: this should be able to be selected ()
        if (c > 0.67 && p != 3) {
            detected_objects.push_back({b, c, p});
        }
    }

    return detected_objects;
}


int main() {
    bool save = true;
    bool load = false;

    Ptr<TrainData> train_data = generate_data("data/task3/train/");
    Ptr<RandomForest> forest = new RandomForest(20, 20, 0, 2, 6);

    if (load) {
        forest->load("model/task3");
    } else {
        cout << "Training classifier..." << endl;
        forest->train(train_data);
    }

    if (save) {
        forest->save("model/task3");
    }

    vector<Mat> test_images = get_images("data/task3/test/");
    cout << "Detecing object..." << endl;
    for (Mat test_image : test_images) {
        vector<DetectedObject> detected_objects = detect_object(test_image, forest);
        cout << detected_objects.size() << endl;
        visualize_detected_objects(detected_objects, test_image.clone());
        vector<DetectedObject> non_suppressed = non_maximum_supression(detected_objects, 0.5);
        std::cout << "max_suppression done" << std::endl;
        cout << non_suppressed.size() << endl;
        visualize_detected_objects(non_suppressed, test_image.clone());

        break;
    }

    return 0;
}