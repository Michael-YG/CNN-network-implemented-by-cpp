#pragma once
#include <iostream>

using namespace std;

class MyMatrix
{
public:
    size_t row;
    size_t col;
    size_t * ref_count;
    size_t channel;
    float * num;

    MyMatrix()
    {
        // cout<<"a smy matrix is created"<<endl;
        ref_count = (size_t*)malloc(sizeof(size_t));
        *ref_count = 0;
    }
    MyMatrix(size_t r, size_t c, size_t ch)
    {
        // cout<<"a smy matrix is created"<<endl;
        ref_count = (size_t*)malloc(sizeof(size_t));
        *ref_count = 0;
        row = r;
        col = c;
        channel = ch;
        num = new float[row*col*channel]();
    }
    ~MyMatrix()
    {
        if(*ref_count==0)
        {
            delete []num;
            cout<<"a matrix is freed"<<endl;
        }
        *ref_count = *ref_count - 1;
    }

    // MyMatrix* operator=(const MyMatrix * mat);
};

bool add(MyMatrix * mat1, MyMatrix * mat2, MyMatrix * outcome);
bool mul(MyMatrix * mat1, MyMatrix * mat2, float * outcome,size_t check);
