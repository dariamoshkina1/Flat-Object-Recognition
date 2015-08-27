#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"

#include <fstream>
#include <iostream>
#include "ctime"
#include "featureExtractor.hpp"

using namespace cv;



#if 1
  #define TS(name) int64 t_##name = getTickCount()
  #define TE(name) printf("TIMER_" #name ": %.2fms\n", \
    1000.f * ((getTickCount() - t_##name) / getTickFrequency()))
#else
  #define TS(name)
  #define TE(name)
#endif

const char* params =
     "{ h | help          | false | print usage                                   }"
     "{   | sample-list   |       | path to list with image classes names         }"
     "{   | samples       |       | path to samples                               }"
     "{   | image         |       | image to detect objects on                    }"
     "{   | camera        | false | whether to detect on video stream from camera }";


float calculateTriangleArea(Point2f p1, Point2f p2, Point2f p3) {
	float a = sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
	float b = sqrt((p3.x-p2.x)*(p3.x-p2.x) + (p3.y-p2.y)*(p3.y-p2.y));
	float c = sqrt((p1.x-p3.x)*(p1.x-p3.x) + (p1.y-p3.y)*(p1.y-p3.y));
	float p = (a+b+c)/2;
	float area = sqrt (p*(p-a)*(p-b)*(p-c));
	return area;
}


float fourPointsArea(Point2f p1, Point2f p2, Point2f p3, Point2f p4) {
	float tr1 = calculateTriangleArea(p1,p2,p3);
	float tr2 = calculateTriangleArea(p1,p3,p4);
	return tr1+tr2;
}

void DrawContours(const Mat image, Mat& test_image, const Mat homography, Scalar color ) {
	std::vector<Point2f> startcorners, newcorners;
	std::vector<float> distances;
	startcorners.push_back(Point2f(0,0));
	startcorners.push_back(Point2f(image.cols,0));
	startcorners.push_back(Point2f( image.cols, image.rows));
	startcorners.push_back(Point2f( 0, image.rows));

	perspectiveTransform(startcorners, newcorners, homography);

	float areaOrig = fourPointsArea(startcorners[0], startcorners[1], startcorners[2], startcorners[3]);
	float areaFound = fourPointsArea(newcorners[0], newcorners[1], newcorners[2], newcorners[3]);
	if (areaFound/areaOrig>0.2) {
	    line(test_image, Point2f(newcorners[0].x, newcorners[0].y), Point2f(newcorners[1].x, newcorners[1].y), color, 4);
	    line(test_image, Point2f(newcorners[1].x, newcorners[1].y), Point2f(newcorners[2].x, newcorners[2].y), color, 4);
	    line(test_image, Point2f(newcorners[2].x, newcorners[2].y), Point2f(newcorners[3].x, newcorners[3].y), color, 4);
	    line(test_image, Point2f(newcorners[3].x, newcorners[3].y), Point2f(newcorners[0].x, newcorners[0].y), color, 4);

	}
	
}


void compute(Mat &image, featureExtractor &extractor)
{
	extractor.compute(image);
}

vector<DMatch> matches(featureExtractor& object, featureExtractor& test)
{
	BFMatcher matcher( NORM_L1 );
	vector< DMatch > match;
	matcher.match( object.GetDescriptor() , test.GetDescriptor() ,match );
	
	return match;

}

Mat Homography(vector<DMatch> matches,featureExtractor &object, featureExtractor &test, double ransacThreshold, Mat& H )
{
	vector<Point2f> obj;
	vector<Point2f> scene;
	for( int i = 0; i < matches.size(); i++ )
	{
		obj.push_back( object.GetKeyPoint()[i].pt );
		scene.push_back(  test.GetKeyPoint()[ matches[i].trainIdx ].pt ); 
	}
	H =  findHomography ( Mat(obj), Mat(scene), CV_RANSAC, ransacThreshold);
	Mat scene_corners;
	perspectiveTransform(Mat(obj), scene_corners, H);
	return scene_corners;

}

void inliers(vector< DMatch > matches, Mat &scene_corners,featureExtractor &test, double ransacThreshold, vector<DMatch>  &inliers)
{
	for (int i=0; i<matches.size(); i++)
	{
		Point2f p1 = test.GetKeyPoint().at( matches[i].trainIdx ).pt;
		Point2f p2 = scene_corners.at<Point2f>(matches[i].queryIdx);
		if ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) < 
			ransacThreshold * ransacThreshold)
		{
			inliers.push_back(matches[i]);
		}
	}
}

int main(int argc, const char **argv)
{ 
	vector<SurfDescriptorExtractor> detector;
	Mat image, test_image;
    CommandLineParser parser(argc, argv, params);
    string sampleListFile = parser.get<string>("sample-list");
	string testImage = parser.get<string>("image");
	string _path = parser.get<string>("samples");
    std::ifstream sampleListFileReader(sampleListFile);
    char buff[50];
	test_image = imread(testImage);
	if (test_image.empty())
	{
		std::cout << "Image is empty." << std::endl;
		return 1;
	}	
	TS();
	featureExtractor test,
		tmp;

	compute(test_image,test);

	Mat H,
		scene;

	vector< DMatch > matcher, inlier;


	Mat img;

    while (sampleListFileReader.getline(buff, 50))
    {
        string str(buff);
        string image_file = str.substr(0,str.find(" "));
		image = imread(_path + image_file);
		compute(image,tmp);
		matcher = matches(tmp, test);
		scene = Homography(	matches(tmp, test), tmp, test, 3.0, H);
		inliers(matcher,scene,test,3.0,inlier);
		DrawContours(image, test_image, H, Scalar(0,255,0));	
    }

	TE();

	imshow("image",test_image);
	waitKey();
    return 0;
}