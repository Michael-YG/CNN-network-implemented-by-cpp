#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_binary_cls.hpp"
#include "MyMatrix.hpp"

using namespace std;

class CNN
{
public:
    string FileName;
    size_t ImageRows;
    size_t ImageCols;
    size_t ImageChannels;
    MyMatrix * matrix = NULL;

    CNN(string file,size_t r,size_t c,size_t ch)
    {
        FileName = file;
        ImageRows = r;
        ImageCols = c;
        ImageChannels = ch;
        matrix = (MyMatrix*)malloc(sizeof(MyMatrix));
        matrix->num = (float*)malloc(sizeof(float)*ImageRows*ImageCols*ImageChannels);
        matrix->row = r;
        matrix->col = c;
        matrix->channel = ch;
    }
    ~CNN()
    {
        delete []matrix->num;
    }

    MyMatrix * setInput(MyMatrix * mt);
    float * conv_relu(const float * in,conv_param & param,size_t row,size_t col);
    float * maxPooling(float * in,size_t row,size_t col,size_t channel);
    void execute();
};

