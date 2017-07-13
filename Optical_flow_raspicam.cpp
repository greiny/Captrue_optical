// to compile: g++ -o main main.cpp -lraspicam_cv -lraspicam `pkg-config --libs --cflags opencv`
#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <stdio.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdlib.h>
#include <unistd.h>
#include <wiringPi.h>
#include <wiringPiI2C.h>


using namespace std;
using namespace cv;


int main ( int argc,char **argv ) {

/////I2C///////
    int devieAddress = 0x55; 
    int fd = wiringPiI2CSetup(devieAddress); 
    int distanceInCM;
    float distanceInM;
    if (fd == -1) {
        printf("I2C Bus file could not be opened\n");}
  


    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Cv Camera;
    cv::Mat imgA, imgB;
    int nCount=1;
    clock_t begin, end;

    //set camera params
    Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
    Camera.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    Camera.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    //Open camera
    cout<<"Opening Camera..."<<endl;

    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
    //Start capture
    cout<<"Capturing "<<nCount<<" frames ...."<<endl;
    
    
    

    for ( ;; ) {

            unsigned char byte[2];
            int res = read(fd, byte, 2);
            

            if (res == -1)
            {
                printf("I2C Device with address %d was not available\n", devieAddress);
            }
            else
            {
                distanceInCM = (byte[0] << 8) | byte[1];
                distanceInM = distanceInCM/100;
                //printf("Distance: %dcm\n", distanceInCM);
            }

            delay(50);


        begin = clock();
        Camera.grab();
        Camera.retrieve (imgB);
        //if ( i%5==0 )  cout<<"\r captured "<<i<<" images"<<std::flush;
        //imshow("raw_image", image);
 	   
	   //resize(imgB, imgB, Size(320, 240));

           cvtColor(imgB, imgB, COLOR_BGR2GRAY);
           Size img_sz = imgB.size();

           Mat imgC(img_sz,1);
           imgB.copyTo(imgC);

           int win_size = 10;
           int maxCorners = 300;
           double qualityLevel = 0.01;
           double minDistance = 5.0;
           int blockSize = 3;
           double k = 0.04;
           std::vector<cv::Point2f> cornersA;
           cornersA.reserve(maxCorners);
           std::vector<cv::Point2f> cornersB;
           cornersB.reserve(maxCorners);

           if (imgA.empty() == false ) {

           goodFeaturesToTrack( imgA,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat(),blockSize,false,0.04);
           goodFeaturesToTrack( imgB,cornersB,maxCorners,qualityLevel,minDistance,cv::Mat());

           /*cornerSubPix( imgA, cornersA, Size( win_size, win_size ), Size( -1, -1 ),
                         TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

           cornerSubPix( imgB, cornersB, Size( win_size, win_size ), Size( -1, -1 ),
                         TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );*/ //요놈을주의깊게보자 정확도와 연결된듯!

           // Call Lucas Kanade algorithm

           //CvSize pyr_sz = Size( img_sz.width+8, img_sz.height/3 );
           //Mat pyrA(pyr_sz,1);
           //Mat pyrB(pyr_sz,1);

           std::vector<uchar> features_found;
           features_found.reserve(maxCorners);
           std::vector<float> feature_errors;
           feature_errors.reserve(maxCorners);

           calcOpticalFlowPyrLK( imgA, imgB, cornersA, cornersB, features_found, feature_errors ,
               Size( win_size, win_size ), 5,
                cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );
                                                                                                                                              
           // Find the homography transformation between the frames.
           Mat H= findHomography( cornersA, cornersB, CV_RANSAC,3);
           //Mat K = (Mat_<float>(3,3) << 6.1059846357523986e+02,0,3.1950000000000000e+02,0,6.1059846357523986e+02,1.7950000000000000e+02,0,0,1);
           //Matx33d K = Matx33d(6.1059846357523986e+02,0,3.1950000000000000e+02,
           //            0,6.1059846357523986e+02,1.7950000000000000e+02,
           //            0,0,1);
           Matx33d K = Matx33d(1,0,0,
                       0,1,0,
                       0,0,1);
           vector<Mat> R, T, N;      
             
           decomposeHomographyMat(H,K,R,T,N);
//printf("1");
           //vector<float> T1 = T[0];
           Mat T1 = T[0];
           //vector<float> T1;


           Mat a(T1.row(0));
           //cout<<"x : "<< a <<endl;

           //float value = T(0,0);
//printf("2");  
           //cout<<"Homography : "<< H <<endl;
           //cout<<"Rotation : "<< R[0] <<endl;
           //cout<<"Translation : "<< T[0] <<endl;
           //cout<<"Normal vector : "<< N[0] <<endl;

           //cout<<"Distance : "<< (float)distanceInCM/100 <<endl;
           cout<<"Distance : "<< distanceInCM <<endl;


           //T1.at<float>(0,0) result in strange result!!!!!!
           //cout<<"x : "<< T1.at<double>(0,0) << endl;
           cout<<"y : "<< (T1.at<double>(1,0))*(distanceInCM+5) << endl;
           //cout<<"y : "<< (T1.at<double>(1,0)) << endl;
           //cout<<"z : "<< T1.at<double>(2,0) << endl;


           //cout<<"value : "<< T1.size() <<endl;
           //cout<<"value : "<< T1 <<endl;
           //cout<<"value2 : "<< T1.at(1,1) <<endl;
           //printf("%f\n",T1.at(1,1));



           // Make an image of the results

           for( int i=0; i < features_found.size(); i++ ){
                   //cout<<"Error is "<<feature_errors[i]<<endl;
                   //continue;

               //cout<<"Got it"<<endl;
           if( features_found[i] == 0 || feature_errors[i] > 550 )
           {
                  //printf("Error is %f\n", feature_errors[i]);
                  continue;
           }
           //printf("Got it\n");
               Point p0( ceil( cornersA[i].x ), ceil( cornersA[i].y ) );
               Point p1( ceil( cornersB[i].x ), ceil( cornersB[i].y ) );
               line( imgC, p0, p1, CV_RGB(255,255,255), 2 );
           }
           //time(&end);

           //namedWindow( "LKpyr_OpticalFlow",  WINDOW_AUTOSIZE );
           //imshow( "LKpyr_OpticalFlow", imgC );

           imgB.copyTo(imgA);
           //cvtColor(imgA, imgA, COLOR_BGR2GRAY);




          }
          else {
           // fill previous image in case prevgray.empty() == true
           imgB.copyTo(imgA);
           //cvtColor(imgA, imgA, COLOR_BGR2GRAY);

          }        


        char chKey = cvWaitKey(5);
        if(chKey == 27) //ESC
           break;

        end = clock();
        double tt = (end - begin);//CLOCKS_PER_SEC;
        double FPS = 1/tt;
        //cout<<"time(sec) : "<< tt/CLOCKS_PER_SEC <<endl;
        cout<<"FPS : "<< FPS*CLOCKS_PER_SEC <<endl;
   
    }


    //cout<<"Stop camera..."<<endl;
    Camera.release();

}
