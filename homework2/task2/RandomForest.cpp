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
    // TODO: what is a good value? Is this the expected amount of inliers?
    double train_frac = 0.9;
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

        for (int i = 0; i < num_train; i++) {
            // need to be deep copies because of how opencv handles matrices
            samples.row(shuffled[i]).copyTo(samples_shuffled.row(i));
            responses.row(shuffled[i]).copyTo(responses_shuffled.row(i));
        }
        cv::Ptr<cv::ml::TrainData> train_data_shuffled(cv::ml::TrainData::create(samples_shuffled, cv::ml::ROW_SAMPLE, responses_shuffled));

        mTrees[i]->train(train_data_shuffled);
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
            // std::cout << ground_truth.at(i) << " " << results.at(i) << std::endl;
        }
    }

    // std::cout << "Correct: " << correct << std::endl;
    // std::cout << "Total: " << results.size() << std::endl;
    return (1 - ((correct + 0.0) / results.size())) * 100;
}
