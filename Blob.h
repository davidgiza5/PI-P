// Blob.h

#ifndef MY_BLOB
#define MY_BLOB

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class Blob {

public:

    std::vector<cv::Point> contur_curent;

    cv::Rect currentBoundingRect;

    vector<Point> centerPositions;

    double dblCurrentDiagonalSize;
    double dblCurrentAspectRatio;

    bool matchORnew;

    bool still_tracked;

    int intNumOfConsecutiveFramesWithoutAMatch;

    cv::Point predictedNextPosition;


    Blob(std::vector<cv::Point> _contur);
    void predictNextPosition(void);

};

#endif
