#include "featureExtractor.hpp"

void featureExtractor::GetFeature(Mat &image)
{
	detector.detect(image, keypoints);
}

vector<KeyPoint> featureExtractor::GetKeyPoint()
{
	return keypoints;
}

void featureExtractor::descriptor(Mat &image)
{
	extractor.compute(image, keypoints, destriptor);
}


void featureExtractor::compute(Mat &image)
{
	GetFeature(image);
	descriptor(image);
	
}
Mat featureExtractor::GetDescriptor()
{
	return destriptor;
}

featureExtractor::featureExtractor(featureExtractor &tmp)
{
	tmp.destriptor.copyTo(destriptor);
}
featureExtractor::featureExtractor()
{
}