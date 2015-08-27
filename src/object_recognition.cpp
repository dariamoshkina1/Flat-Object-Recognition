
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\nonfree\features2d.hpp"
#include "opencv2\calib3d\calib3d.hpp"

#include <fstream>
#include <iostream>

using namespace cv;

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

Mat Descriptor(const Mat image, Mat test_image)
{
	const int ransacThreshold = 3;

	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints_object, keypoint_test;
	detector.detect( image, keypoints_object );
	detector.detect( test_image, keypoint_test );

	SurfDescriptorExtractor extractor;
	Mat destriptor_object, destriptor_test;	
	extractor.compute(image,  keypoints_object ,destriptor_object);
	extractor.compute(test_image, keypoint_test ,destriptor_test);

	BFMatcher matcher( NORM_L1 );
	vector< DMatch > matches;
	matcher.match( destriptor_object, destriptor_test,matches );


	Mat img_matches; 
    drawMatches(image,keypoints_object,test_image, keypoint_test, matches,img_matches);
	waitKey();

	vector<Point2f> obj;
	vector<Point2f> scene;
	for( int i = 0; i < matches.size(); i++ )
	{
		obj.push_back( keypoints_object[i].pt );
		scene.push_back(  keypoint_test[ matches[i].trainIdx ].pt ); 
	}
	
	Mat H = findHomography ( Mat(obj), Mat(scene), CV_RANSAC, ransacThreshold);

	Mat  scene_corners;
	perspectiveTransform(Mat(obj), scene_corners, H);

	vector< DMatch > inliers;
	for (int i=0; i<matches.size(); i++)
	{
		Point2f p1 = keypoint_test.at( matches[i].trainIdx ).pt;
		Point2f p2 = scene_corners.at<Point2f>(matches[i].queryIdx);
		if ((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) < 
			ransacThreshold * ransacThreshold)
		{
			inliers.push_back(matches[i]);
		}
	}

	Mat img_matches_a;
	drawMatches(image, keypoints_object,test_image, keypoint_test, inliers,img_matches_a);
	Mat final_test_image;
	test_image.copyTo(final_test_image);
	DrawContours(image, test_image, H, Scalar(0,255,0));

	return test_image;
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
	test_image = imread(testImage,0);
    while (sampleListFileReader.getline(buff, 50))
    {
        string str(buff);
        string image_file = str.substr(0,str.find(" "));

		image = imread(_path + image_file,0);

		test_image = Descriptor(image,test_image);
    }
	imshow("image",test_image);
	waitKey();
    return 0;
}
