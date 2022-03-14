#include "MyMatrix.hpp"
#include <iostream>

using namespace std;

// MyMatrix* MyMatrix::operator=(const MyMatrix * mat)
// {
//     cout<<"MyMatrix& MyMatrix::operator=(const MyMatrix & mat)"<<endl;
//     if(this == mat)
//         return this;
//     this->row = mat->row;
//     this->col = mat->col;
//     this->channel = mat->channel;
//     this->num = mat->num;
//     this->ref_count = mat->ref_count;
//     *mat->ref_count = *mat->ref_count + 1;
//     return this;
// }

bool add(MyMatrix * mat1, MyMatrix * mat2, MyMatrix * outcome)
{
    if(mat1 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The mat1 parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if(mat2 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The mat2 parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if(outcome == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The outcome parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if((mat1->channel!=mat2->channel)||(mat2->channel!=outcome->channel))
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The channel of inputs does not match.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if((mat1->row!=mat2->row)||(mat1->col!=mat2->col))
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The dimension of mat1 and mat2 does not match!\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if((outcome->row!=mat1->row)||(outcome->col!=mat1->col))
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The dimension of mat1 and outcome does not match!\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    size_t length = mat1->row*mat1->col*mat1->channel;
    float * p1 = mat1->num;
    float * p2 = mat2->num;
    float * po = outcome->num;
    for(size_t i=0;i<length;i++)
    {
        *(po++) = *(p1++) + *(p2++);
    }
    return true;
}

bool mul(MyMatrix * mat1, MyMatrix * mat2, float * outcome,size_t check)
{
    if(mat1 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The mat1 parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if(mat2 == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The mat2 parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if(outcome == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The outcome parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    if((mat1->col!=mat2->row))
    {
        cout<<mat1->col<<" "<<mat2->row<<endl;
        fprintf(stderr, "File %s, Line %d, Function %s(): The dimension of mat1 and mat2 does not match!\n", __FILE__, __LINE__, __FUNCTION__);
        return false;
    }
    float * p1 = mat1->num;
    float * p2 = mat2->num;
    float * po = outcome;
    size_t rows = mat1->row;
    size_t cols = mat2->col;
    size_t channels = mat1->channel;
    float var = 0;
    for(size_t c=0;c<channels;c++)
    {
        for(size_t i=0;i<rows;i++)
        {
            for(size_t j=0;j<cols;j++)
            {
                for(size_t k=0;k<mat1->col;k++)
                {
                    po[i*cols+j]+=(p1[i*cols+k]*p2[k*cols+j]);
                    // if(check == 1)
                    // {
                    //     cout<<p1[i*cols+k]<<" "<<p2[k*cols+j]<<endl;
                    // }
                }
            }
        }
    }
    return true;
}

