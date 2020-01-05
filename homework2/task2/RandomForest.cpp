#include "RandomForest.h"

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
    // Fill
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
    // Fill
    mCVFolds = cvFols;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setCVFolds(mCVFolds);
}

void RandomForest::setMinSampleCount(int minSampleCount)
{
    // Fill
    mMinSampleCount = minSampleCount;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMinSampleCount(mMinSampleCount);
}

void RandomForest::setMaxCategories(int maxCategories)
{
    // Fill
    mMaxCategories = maxCategories;
    for(uint treeIdx=0; treeIdx<mTreeCount; treeIdx++)
        mTrees[treeIdx]->setMaxCategories(mMaxCategories);
}



void RandomForest::train(const cv::Ptr<cv::ml::TrainData>& train_data)
{
    // Fill
    for (int i = 0; i < mTreeCount; i++) {
        mTrees[i]->train(train_data);
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

        predictions.push_back(final_prediction);
    }

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
            // cout << ground_truth.at(i) << " " << results.at(i) << endl;
        }
    }

    // cout << "Correct: " << correct << endl;
    // cout << "Total: " << results.size() << endl;
    return (1 - ((correct + 0.0) / results.size())) * 100;
}
