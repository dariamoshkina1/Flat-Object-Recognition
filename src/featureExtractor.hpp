#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include <iostream>

using namespace cv;

class featureExtractor
{
private:
	SurfFeatureDetector detector;
	SurfDescriptorExtractor extractor;
	vector<KeyPoint> keypoints;
	Mat destriptor;
	
public:
	featureExtractor();
	featureExtractor(featureExtractor &tmp);
	void GetFeature(Mat &image);
	vector<KeyPoint> GetKeyPoint();
	void descriptor(Mat &image);
	void compute(Mat &image);
	Mat GetDescriptor();

};