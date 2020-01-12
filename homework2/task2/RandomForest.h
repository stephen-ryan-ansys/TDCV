

#ifndef RF_RANDOMFOREST_H
#define RF_RANDOMFOREST_H


#include <opencv2/opencv.hpp>
#include <vector>

class RandomForest
{
public:
	RandomForest();

    // You can create the forest directly in the constructor or create an empty forest and use the below methods to populate it
	RandomForest(int treeCount, int maxDepth, int CVFolds, int minSampleCount, int maxCategories);
    
    ~RandomForest();

    void setTreeCount(int treeCount);
    void setMaxDepth(int maxDepth);
    void setCVFolds(int cvFols);
    void setMinSampleCount(int minSampleCount);
    void setMaxCategories(int maxCategories);

    std::vector<int> getPredictions();
    std::vector<double> getConfidences();

    void train(const cv::Ptr<cv::ml::TrainData>& train_data);

    // float predict(/* Fill */);
    std::vector<int> predict(cv::InputArray samples);

    float calcError(cv::Ptr<cv::ml::TrainData>& data, bool test, cv::OutputArray resp);

private:
	int mTreeCount;
	int mMaxDepth;
	int mCVFolds;
	int mMinSampleCount;
	int mMaxCategories;

    std::vector<int> mPredictions;
    std::vector<double> mConfidences;

    // M-Trees for constructing the forest
    std::vector<cv::Ptr<cv::ml::DTrees> > mTrees;
};

#endif //RF_RANDOMFOREST_H
