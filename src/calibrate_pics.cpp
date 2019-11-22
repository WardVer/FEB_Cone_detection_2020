
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <thread>

using namespace cv;
using namespace std;

using namespace cv;

int main(int, char** argv)
{

  VideoCapture cap1;
  VideoCapture cap2;

  cap1.open(0);
  cap2.open(2);

  FileStorage fs("piclist.xml", FileStorage::WRITE);

  std::string picpath = "./calibrationpics/pic";
  std::string picext = ".jpg";

  Mat frame1;
  Mat frame2;

  Mat concatframe;

  int sectime = 0;

  for(int i = 0; i < sectime*30; i++)
  {
    cap1 >> frame1;
    cap2 >> frame2;

        
    hconcat(frame1, frame2, concatframe);
    flip(concatframe, concatframe, 1);
    putText(concatframe, to_string(sectime-i/30), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
  }

  int multiplier = 20;
  int pics = 20;

  fs << "strings" << "["; 

  for(int i = 0; i < pics*multiplier; i++)
  {
    std::thread t_cap1;
    std::thread t_cap2;

    t_cap2 = std::thread([&]()
    {
      cap2 >> frame2; 
    });

    t_cap1 = std::thread([&]()
    {
      cap1 >> frame1; 
    });

    

    t_cap1.join();
    t_cap2.join();

    if(i%multiplier == 0){
      std::string nr = to_string(i*2/multiplier);
      std::string picnr = std::string(5 - nr.length(), '0') + nr;
      imwrite(picpath + picnr + picext, frame1);
      fs << (picpath + picnr + picext).c_str();


      nr = to_string(i*2/multiplier+1);
      picnr = std::string(5 - nr.length(), '0') + nr;
      imwrite(picpath + picnr + picext, frame2);
      fs << (picpath + picnr + picext).c_str();
    }


    hconcat(frame1, frame2, concatframe);
    flip(concatframe, concatframe, 1);
    putText(concatframe, to_string(i/multiplier), Point(50,50), FONT_HERSHEY_PLAIN, 4, cv::Scalar(0,255,100), 2);
    imshow("yeet", concatframe);
    waitKey(1);
    

  }

  fs << "]";

  fs.release();


  return 0;
}