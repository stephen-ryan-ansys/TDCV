#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <map>

#include "../commons/feature_extraction.h"
#include "RandomForest.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

int is_regular_file(const char *path) {
    struct stat path_stat;
    stat(path, &path_stat);
    return S_ISREG(path_stat.st_mode);
}

Ptr<TrainData> generate_data(string base_path, bool is_test=false) {
    vector<string> dirs = {"00", "01", "02", "03", "04", "05"};
    vector<int> categories = {0, 1, 2, 3, 4, 5};

    Mat input;
    Mat ground_truth;
    for (int i = 0;i < dirs.size(); i++) {
        string class_path = base_path + dirs.at(i) + "/";
        cout << "Generating data for folder: " + class_path << endl;
        DIR * dirp = opendir(class_path.c_str());
        dirent * dp;
        while ((dp = readdir(dirp)) != NULL) {
            string full_path = class_path + dp->d_name;
            if (is_regular_file(full_path.c_str())) {
                Mat image = imread(full_path.c_str(), 1);
                ExtractionResult extraction_result = extract(image, is_test);
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
        (void)closedir(dirp);
    }

    return TrainData::create(input, ROW_SAMPLE, ground_truth);
}

template<class ClassifierType>
void performanceEval(cv::Ptr<ClassifierType> classifier, cv::Ptr<cv::ml::TrainData> train_data, cv::Ptr<cv::ml::TrainData> test_data, bool load = false) {
    if (!load) {
        classifier->train(train_data);
    }

    printf("Performance evaluation on training data\n");
    printf("error: %f\n", classifier->calcError(train_data, false, noArray()) );
    printf("Performance evaluation on test data\n");
    printf("error: %f\n", classifier->calcError(test_data, false, noArray()) );
};

void testDTrees() {

    int num_classes = 6;

    /* 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a single Decision Tree and evaluate the performance 
      * Experiment with the MaxDepth parameter, to see how it affects the performance

    */

    Ptr<TrainData> train_data = generate_data("data/task2/train/");
    Ptr<TrainData> test_data = generate_data("data/task2/test/", true);

    Ptr<DTrees> tree = DTrees::create();
    tree->setCVFolds(0);
    tree->setMaxCategories(6);
    tree->setMaxDepth(20);
    tree->setMinSampleCount(2);

    performanceEval<cv::ml::DTrees>(tree, train_data, test_data);
}

void testForest(){

    int num_classes = 6;

    /* 
      * 
      * Create your data (i.e Use HOG from task 1 to compute the descriptor for your images)
      * Train a Forest and evaluate the performance 
      * Experiment with the MaxDepth & TreeCount parameters, to see how it affects the performance

    */

    Ptr<TrainData> train_data = generate_data("data/task2/train/");
    Ptr<TrainData> test_data = generate_data("data/task2/test/", true);
    Ptr<RandomForest> forest = new RandomForest(5, 20, 0, 2, num_classes);

    bool load = true;
    if (load) {
        forest->load("model/task2");
    }

    performanceEval<RandomForest>(forest, train_data, test_data, load);

    forest->save("model/task2");
}

int main() {
    testDTrees();
    testForest();
    return 0;
}
