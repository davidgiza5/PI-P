// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           

#include "Blob.h"

#define SHOW_STEPS            


using namespace cv;
using namespace std;


const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);



void potrivire_cadru_curent(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs);
void actualizare_blob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex);
void adaugare_blob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs);
double distanta_dintre2puncte(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool verificare_trecere_linie(std::vector<Blob>& blobs, int& poz_linie_orizontala, int& nr_masini);
void afisare_informatii_blob(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy);
void afisare_nr_masini(int& nr_masini, cv::Mat& imgFrame2Copy, int& nr_masini_2);
bool verificare_trecere_linie_contrasens(std::vector<Blob>& blobs, int& poz_linie_orizontala_1, int& nr_masini_2);



int main(void) {

    VideoCapture capVideo;

    Mat imgFrame1;
    Mat imgFrame2;

    vector<Blob> blobs;

    Point linie_detectie[2];
    Point linie_detectie_1[2];

    int nr_masini = 0;
    int nr_masini_2 = 0;

    capVideo.open("Roads - 1952.mp4");

    if (!capVideo.isOpened()) {                                                 
        std::cout << "Eroare la citire video" << std::endl << std::endl;      
        _getch();                   
        return(0);                                                              
    }

    if (capVideo.get(cv::CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "Eroare: video-ul trebuie sa aiba cel putin 2 cadre";
        _getch();                   
        return(0);
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    int poz_linie_orizontala = (int)std::round((double)imgFrame1.rows * 0.75);
    int poz_linie_orizontala_1= (int)std::round((double)imgFrame1.rows * 0.75);

    linie_detectie[0].x = 675;
    linie_detectie[0].y = poz_linie_orizontala;

    linie_detectie[1].x = imgFrame1.cols - 200;
    linie_detectie[1].y = poz_linie_orizontala;

    linie_detectie_1[0].x = 25;
    linie_detectie_1[0].y = poz_linie_orizontala_1;
    linie_detectie_1[1].x = imgFrame1.cols - 775;
    linie_detectie_1[1].y = poz_linie_orizontala_1;

    char verificare_esc = 0;

    bool primulCadru = true;

    int nrCadre = 2;

    while (capVideo.isOpened() && verificare_esc != 27) {

        vector<Blob> currentFrameBlobs;

        Mat imgFrame1Copy = imgFrame1.clone();
        Mat imgFrame2Copy = imgFrame2.clone();

        Mat imgDifference;
        Mat imgThresh;

        cvtColor(imgFrame1Copy, imgFrame1Copy, COLOR_BGR2GRAY);
        cvtColor(imgFrame2Copy, imgFrame2Copy, COLOR_BGR2GRAY);

        GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        threshold(imgDifference, imgThresh, 30, 255.0, THRESH_BINARY);

        imshow("Segmentare", imgThresh);

        Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        //Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        //Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        //Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement3x3);
            cv::dilate(imgThresh, imgThresh, structuringElement3x3);
            cv::erode(imgThresh, imgThresh, structuringElement3x3);
        }

        Mat imgThreshCopy = imgThresh.clone();

        vector<vector<cv::Point> > contours;

        findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "Contururi");

        vector<vector<Point>> convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "ConvexHulls");

        for (auto& convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
                (contourArea(possibleBlob.contur_curent) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) 
            {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "Blobs_cadru_curent");

        if (primulCadru == true) {
            for (auto& currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        }
        else {
            potrivire_cadru_curent(blobs, currentFrameBlobs);
        }

        drawAndShowContours(imgThresh.size(), blobs, "Blobs");

        imgFrame2Copy = imgFrame2.clone();          

        afisare_informatii_blob(blobs, imgFrame2Copy);

        bool OK = verificare_trecere_linie(blobs, poz_linie_orizontala, nr_masini);
        bool OK_2 = verificare_trecere_linie_contrasens(blobs, poz_linie_orizontala_1, nr_masini_2);

        if (OK == true) {
            line(imgFrame2Copy, linie_detectie[0], linie_detectie[1], SCALAR_GREEN, 2);
        }
        else {
            line(imgFrame2Copy, linie_detectie[0], linie_detectie[1], SCALAR_RED, 2);
        }

        if (OK_2 == true) {
            line(imgFrame2Copy, linie_detectie_1[0], linie_detectie_1[1], SCALAR_GREEN, 2);
        }
        else {
            line(imgFrame2Copy, linie_detectie_1[0], linie_detectie_1[1], SCALAR_YELLOW, 2);
        }


        afisare_nr_masini(nr_masini, imgFrame2Copy,nr_masini_2);

        imshow("Supraveghere trafic", imgFrame2Copy);



        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           

        if ((capVideo.get(CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
        }
        else {
            cout << "sfarsitul video-ului\n";
            break;
        }

        primulCadru = false;
        nrCadre++;
        verificare_esc = cv::waitKey(1);
    }
    

    if (verificare_esc != 27) {              
        waitKey(0);                         
    }

    
    

    return(0);
}


void potrivire_cadru_curent(std::vector<Blob>& existingBlobs, std::vector<Blob>& currentFrameBlobs) {

    for (auto& existingBlob : existingBlobs) {

        existingBlob.matchORnew = false;

        existingBlob.predictNextPosition();
    }


    for (auto& currentFrameBlob : currentFrameBlobs) {

        int index_dist_min = 0;
        double distanta_minima = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].still_tracked == true) {

                double distanta = distanta_dintre2puncte(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (distanta < distanta_minima) {
                    distanta_minima = distanta;
                    index_dist_min = i;
                }
            }
        }

        if (distanta_minima < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            actualizare_blob(currentFrameBlob, existingBlobs, index_dist_min);
        }
        else {
            adaugare_blob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto& existingBlob : existingBlobs) {

        if (existingBlob.matchORnew == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.still_tracked = false;
        }

    }

}

/// <summary>
/// adaugare blob la blob existent
/// </summary>
/// <param name="currentFrameBlob"></param>
/// <param name="existingBlobs"></param>
/// <param name="intIndex"></param>
void actualizare_blob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs, int& intIndex) {

    existingBlobs[intIndex].contur_curent = currentFrameBlob.contur_curent;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].still_tracked = true;
    existingBlobs[intIndex].matchORnew = true;
}

/// <summary>
/// adaugare blob nou
/// </summary>
/// <param name="currentFrameBlob"></param>
/// <param name="existingBlobs"></param>
void adaugare_blob(Blob& currentFrameBlob, std::vector<Blob>& existingBlobs) {

    currentFrameBlob.matchORnew = true;

    existingBlobs.push_back(currentFrameBlob);
}

/// <summary>
/// distanta dintre 2 puncte
/// </summary>
/// <param name="point1"></param>
/// <param name="point2"></param>
/// <returns></returns>
double distanta_dintre2puncte(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}


void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

/// <summary>
/// 
/// </summary>
/// <param name="imageSize"></param>
/// <param name="blobs"></param>
/// <param name="strImageName"></param>
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto& blob : blobs) {
        if (blob.still_tracked == true) {
            contours.push_back(blob.contur_curent);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

/// <summary>
/// verifcare trecere linie si contorizare masini
/// </summary>
/// <param name="blobs"></param>
/// <param name="poz_linie_orizontala"></param>
/// <param name="nr_masini"></param>
/// <returns></returns>
bool verificare_trecere_linie(std::vector<Blob>& blobs, int& poz_linie_orizontala, int& nr_masini) {
    bool OK = false;
    

    for (auto blob : blobs) {

        if (blob.still_tracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size()-2;
            int currFrameIndex = (int)blob.centerPositions.size()-1;

            if (blob.centerPositions[prevFrameIndex].y <= poz_linie_orizontala && blob.centerPositions[currFrameIndex].y > poz_linie_orizontala) {
                nr_masini++;
                OK = true;
            }
        }

    }

    return OK;
}

bool verificare_trecere_linie_contrasens(std::vector<Blob>& blobs, int& poz_linie_orizontala_1, int& nr_masini_2) {
    bool OK_2 = false;

    for (auto blob : blobs) {

        if (blob.still_tracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

              if (blob.centerPositions[prevFrameIndex].y > poz_linie_orizontala_1 && blob.centerPositions[currFrameIndex].y <= poz_linie_orizontala_1)
            {
                nr_masini_2++;
                OK_2 = true;
            }
        }

    }

    return OK_2;
}

/// <summary>
/// afisare informatii blobs - id, boundingrect
/// </summary>
/// <param name="blobs"></param>
/// <param name="imgFrame2Copy"></param>
void afisare_informatii_blob(std::vector<Blob>& blobs, cv::Mat& imgFrame2Copy) {

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].still_tracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

            int font = FONT_HERSHEY_SIMPLEX;
            double dimensiune_font = blobs[i].dblCurrentDiagonalSize / 60.0;
            int grosime_font = (int)std::round(dimensiune_font * 1.0);

            //cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), font, dimensiune_font, SCALAR_GREEN, grosime_font);
        }
    }
}

/// <summary>
/// afisare numar de masini
/// </summary>
/// <param name="nr_masini"></param>
/// <param name="imgFrame2Copy"></param>
void afisare_nr_masini(int& nr_masini, cv::Mat& imgFrame2Copy,int& nr_masini_2) {

    int font = FONT_HERSHEY_SIMPLEX;
    double dimensiune_font = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int grosime_font = (int)std::round(dimensiune_font * 1.5);

    Size textSize = getTextSize(std::to_string(nr_masini), font, dimensiune_font, grosime_font, 0);

    Point ptTextBottomLeftPosition;
    Point nr2;

    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 50 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);
    nr2.x=imgFrame2Copy.cols-50 - (int)((double)textSize.width * 1.25);
    nr2.y= (int)((double)textSize.height * 2.5);

    putText(imgFrame2Copy, "Sens 1:",Point(850,75), FONT_HERSHEY_SIMPLEX,2,Scalar(0,200,0),3);
    putText(imgFrame2Copy, to_string(nr_masini), ptTextBottomLeftPosition, font, dimensiune_font, SCALAR_GREEN, grosime_font);
    putText(imgFrame2Copy, "Sens 2:", Point(850, 160), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 200, 0), 3);
    putText(imgFrame2Copy, to_string(nr_masini_2), nr2, font, dimensiune_font, SCALAR_GREEN, grosime_font);

}
