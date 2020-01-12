#include "RandomForest.h"
#include <opencv2/core/utils/filesystem.hpp>

RandomForest::RandomForest()
{
}

RandomForest::RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories)
    :mTreeCount(treeCount), mMaxDepth(maxDepth), mCVFolds(CVFolds), mMinSampleCount(minSampleCount), mMaxCategories(maxCategories)
{
   /*
     construct a forest with given number of trees and initialize all the trees with the
     given parameters
   */
    for (uint i = 0; i < treeCount; i++) {
        cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
        dtree->setCVFolds(CVFolds);
        dtree->setMaxCategories(maxCategories);
        dtree->setMaxDepth(maxDepth);
        dtree->setMinSampleCount(minSampleCount);
        mTrees.push_back(dtree);
    }
}

RandomForest::~RandomForest()
{
}

void RandomForest::setTreeCount(int treeCount)
{
    for (cv::Ptr<cv::ml::DTrees> dtrees : mTrees) {
        delete dtrees;
    }
    mTrees.clear();

    mTreeCount = treeCount;
    for (uint i = 0; i < treeCount; i++) {
        cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
        dtree->setCVFolds(mCVFolds);
        dtree->setMaxCategories(mMaxCategories);
        dtree->setMaxDepth(mMaxDepth);
        dtree->setMinSampleCount(mMinSampleCount);
        mTrees.push_back(dtree);
    }
}

void RandomForest::setMaxDepth(int maxDepth)
{
    mMaxDepth = maxDepth;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxDepth(mMaxDepth);
}

void RandomForest::setCVFolds(int cvFols)
{
    mCVFolds = cvFols;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setCVFolds(mCVFolds);
}

void RandomForest::setMinSampleCount(int minSampleCount)
{
    mMinSampleCount = minSampleCount;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMinSampleCount(mMinSampleCount);
}

void RandomForest::setMaxCategories(int maxCategories)
{
    mMaxCategories = maxCategories;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxCategories(mMaxCategories);
}

std::vector<int> RandomForest::getPredictions() const
{
    return mPredictions;
}

std::vector<double> RandomForest::getConfidences() const
{
    return mConfidences;
}

void RandomForest::train(const cv::Ptr<cv::ml::TrainData>& train_data)
{
    double train_frac = 0.5;
    int total_samples = train_data->getNTrainSamples();
    int num_train = train_frac*total_samples;
    std::cout << "training each tree with " << num_train << " out of " << total_samples << " total samples." << std::endl;

    std::vector<int> shuffled;
    for (int i = 0; i < total_samples; i++) {
        shuffled.push_back(i);
    }

    for (int i = 0; i < mTreeCount; i++) {
        std::random_shuffle(shuffled.begin(), shuffled.end());

        cv::Mat samples = train_data->getSamples();
        cv::Mat responses = train_data->getResponses();
        cv::Mat samples_shuffled(num_train, samples.cols, samples.type());
        cv::Mat responses_shuffled(num_train, 1, responses.type());

        for (int j = 0; j < num_train; j++) {
            // need to be deep copies because of how opencv handles matrices
            samples.row(shuffled[j]).copyTo(samples_shuffled.row(j));
            responses.row(shuffled[j]).copyTo(responses_shuffled.row(j));
        }
        cv::Ptr<cv::ml::TrainData> train_data_shuffled(cv::ml::TrainData::create(samples_shuffled, cv::ml::ROW_SAMPLE, responses_shuffled));

        mTrees[i]->train(train_data_shuffled);
        std::cout << "tree " << (i + 1) << " of " << mTreeCount << " trained" << std::endl;
    }
}

void RandomForest::save(const cv::String &dirpath) const {
    cv::String dirpath_new = dirpath;
    if (dirpath_new.back() != '/') {
        dirpath_new.push_back('/');
    }

    cv::utils::fs::createDirectories(dirpath_new);
    int i = 0;
    for (auto tree : mTrees) {
        const std::string savepath = dirpath_new + std::to_string(i) + ".model";
        tree->save(savepath);
        std::cout << "saved tree " << savepath << std::endl;
        i++;
    }
}

void RandomForest::load(const cv::String &dirpath) {
    cv::String dirpath_new = dirpath;
    if (dirpath_new.back() != '/') {
        dirpath_new.push_back('/');
    }

    for (auto& dtrees : mTrees) {
        dtrees.reset();
        delete dtrees;
    }
    mTrees.clear();

    for (int i = 0; i < mTreeCount; i++) {
        const std::string loadpath = dirpath_new + std::to_string(i) + ".model";
        mTrees.push_back(cv::ml::DTrees::load(loadpath));
        std::cout << "loaded tree " << loadpath << std::endl;
        if (i == 0) {
            mCVFolds = mTrees.at(i)->getCVFolds();
            mMaxCategories = mTrees.at(i)->getMaxCategories();
            mMaxDepth = mTrees.at(i)->getMaxDepth();
            mMinSampleCount = mTrees.at(i)->getMinSampleCount();
        }
    }
}

std::vector<int> RandomForest::predict(cv::InputArray samples)
{
    // Fill
    std::vector<std::vector<float> > trees_results;
    for (int i = 0; i < mTreeCount; i++) {
        std::vector<float> results;
        mTrees[i]->predict(samples, results);
        trees_results.push_back(results);
    }

    std::vector<int> predictions;
    std::vector<double> confidences;
    for (int i = 0; i < samples.rows(); i++) {
        int final_prediction = -1;
        int max = -1;
        std::map<int, int> count;

        for (int j = 0; j < mTreeCount; j++) {
            int prediction = trees_results.at(j).at(i); // TO INT
            count[prediction]++;
            if (count[prediction] > max) {
                max = count[prediction];
                final_prediction = prediction;
            }
        }
        double confidence = static_cast<double>(count[final_prediction])/mTreeCount;

        confidences.push_back(confidence);
        predictions.push_back(final_prediction);
    }
    mPredictions = predictions;
    mConfidences = confidences;

    return predictions;
}

float RandomForest::calcError(cv::Ptr<cv::ml::TrainData>& data, bool test, cv::OutputArray resp) {
    std::vector<int> results = RandomForest::predict(data->getSamples());
    std::vector<int> ground_truth = data->getResponses();

    int correct = 0;
    for (int i = 0; i < results.size(); i++) {
        if (ground_truth.at(i) == results.at(i)) {
            correct++;
        } else {
            // std::cout << ground_truth.at(i) << " " << results.at(i) << std::endl;
        }
    }

    // std::cout << "Correct: " << correct << std::endl;
    // std::cout << "Total: " << results.size() << std::endl;
    return (1 - ((correct + 0.0) / results.size())) * 100;
}
