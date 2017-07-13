#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>

#include <ctime>
#include <sstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std;
using namespace cv;


// Optical Flow with OpenCV

class OV7251 {
	float phi, the, psi;
	float dist_x, dist_y;

public:
	float roll(Mat R) {
		phi = (float)atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		return phi;}
	float pitch(Mat R) {
		the = (float)atan2(-1*R.at<double>(2, 0), sqrt( pow(R.at<double>(2, 1), 2) + pow(R.at<double>(2, 2), 2) ));
		return the;}
	float yaw(Mat R) {
		psi = (float)atan2(R.at<double>(1, 0), R.at<double>(0, 0));
		return psi;}

	float x(Mat T,float distanceInCM) {
		float w = (float)T.at<double>(2);
		dist_x = T.at<double>(0)/w*distanceInCM;
		return dist_x;}
	float y(Mat T,float distanceInCM) {
		float w = (float)T.at<double>(2);
		dist_y = T.at<double>(1)/w*distanceInCM;
		return dist_y;}

};

static void process_image() {

	Mat raw1 = imread("i1.jpg",0);
	Mat raw2 = imread("i2.jpg",0);
	int fpixel = 430;
	float fmm = 1.3;
	Rect myROI(0,0,630,470);
	Mat next = raw2(myROI);
	Mat prev = raw1(myROI); 

        int maxCorners = 10;
	double qualityLevel = 0.01;
	double minDistance = 5.0;
	int blockSize = 3;
	Size win_size(30,30);
	int distanceInMM=300;
	float distanceInCM=distanceInMM/10.0;

	std::vector<cv::Point2f> cornersA;
	cornersA.reserve(maxCorners);
	std::vector<cv::Point2f> cornersB;
	cornersB.reserve(maxCorners);


	goodFeaturesToTrack( prev,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat(),blockSize,false,0.04);

	Mat features_found, feature_errors;
	vector<Mat> prevPyr,nextPyr;
	buildOpticalFlowPyramid(prev, prevPyr, win_size, 3, true);
	buildOpticalFlowPyramid(next, nextPyr, win_size, 3, true);
	calcOpticalFlowPyrLK( prevPyr, nextPyr, cornersA, cornersB, features_found, feature_errors,
	      win_size);

//-- Quick calculation of max and min distances between features
	float min_dist = 100;
	for( int i = 0; i < cornersA.size(); i++ )
	{
		float dist = feature_errors.at<float>(i);
		if( dist < min_dist || dist != 0 ) min_dist = dist;
	}

	
//cout << "A : " << cornersA.size() << "  B : " << cornersB.size() << endl;
//cout << " min_dist " << min_dist << endl;

// If points status are ok and distance not negligible keep the point
	for( int i=0; i < cornersA.size(); i++ )
	{
		if (!features_found.at<unsigned char>(i) || feature_errors.at<float>(i) > 3*min_dist || feature_errors.at<float>(i)==0 )
		{		
		   cornersA.erase(cornersA.begin() +i);
		   cornersB.erase(cornersB.begin() +i);
		}
	}

//cout << "A : " << cornersA.size() << "  B : " << cornersB.size() << endl;

	Mat H = findHomography( cornersA, cornersB, CV_RANSAC,3);

	Mat Camera_mat = (Mat_<double>(3,3) << 430.546, 0, 309.724,
   			      0, 429.713, 248.413,		
   			      0, 0, 1);
	std::vector<Mat> R, T, N;
	decomposeHomographyMat(H,Camera_mat,R,T,N);

	OV7251 attitude;
	float phi=attitude.roll(R[0]);
	float y=attitude.y(T[0],distanceInCM);

//cout << Deg[2] << endl;
 
// Showing image
	Mat buf;
	cvtColor(next, buf, CV_GRAY2BGR);
	for( int i=0; i < cornersA.size(); i++ ){
	       Point p0( ceil( cornersA[i].x ), ceil( cornersA[i].y ) );
	       Point p1( ceil( cornersB[i].x ), ceil( cornersB[i].y ) );
	       line( buf, p0, p1, CV_RGB(0,255,0), 2 );
	   }
	imshow("Camera Preview", buf);
	//waitKey(0);
}

int main(int argc, char **argv)
{
        namedWindow( "Camera Preview", WINDOW_AUTOSIZE );// Create a window for display.
	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
	auto t0 = std::chrono::high_resolution_clock::now();
 	while(1){
	//check for FPS(Frame Per Second)
	auto t1 = std::chrono::high_resolution_clock::now();
	time += std::chrono::duration<float>(t1-t0).count();
	t0 = t1;
	++frames;
	if(time > 0.5f)
	{
	    fps = frames / time;
	    cout << fps << endl;
	    frames = 0;
	    time = 0;
	}
	process_image();}
}

