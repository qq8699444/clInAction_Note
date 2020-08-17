
#include <opencv2/opencv.hpp>
#include "imgShow.h"

void showImage(uint8_t* imgData, const int width, const int height,int channel )
{
    int type;
    if (channel == 1)
    {
        type = CV_8UC1;
    }
    else if (channel == 3)
    {
        type = CV_8UC3;
    }
    else
    {
        type = CV_8UC4;
    }


    cv::Mat img = cv::Mat(cv::Size(width, height), type, imgData);

    cv::imshow("img", img);

    //imwrite("111.jpg", img);

    cv::waitKey(0);
}