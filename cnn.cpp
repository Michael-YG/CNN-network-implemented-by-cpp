#include "cnn.hpp"
#include <iostream>
#include "time.h"

using namespace std;

clock_t start,finish;

MyMatrix * CNN::setInput(MyMatrix * mt)
{
    if(mt == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The mt parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return NULL;
    }
    cv::Mat InputImage = cv::imread(FileName);
    if(InputImage.data==nullptr)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The image does not exist!\n", __FILE__, __LINE__, __FUNCTION__);
        return NULL;
    }else
    {
        float base = 255.0f;
        size_t step = mt->col*mt->row;
        float * mat_p = mt->num;

        if((InputImage.rows==ImageRows)&&(InputImage.cols==ImageCols))
        {
            for(size_t i=0;i<ImageRows;i++)
            {
                uchar * p = InputImage.ptr<uchar>(i);
                size_t var1 = i*ImageCols;
                #pragma omp parallel for
                for(size_t j=0;j<ImageCols;j++)
                {
                    mat_p[var1+j] = (float)p[3*j+1]/base;
                    mat_p[step+var1+j] = (float)p[3*j+2]/base;
                    mat_p[2*step+var1+j] = (float)p[3*j]/base;
                }
            }
        }else
        {
            cv::resize(InputImage, InputImage, cv::Size(ImageRows, ImageCols));
            for(size_t i=0;i<ImageRows;i++)
            {
                uchar * p = InputImage.ptr<uchar>(i);
                size_t var1 = i*ImageCols;
                #pragma omp parallel for
                for(size_t j=0;j<ImageCols;j++)
                {
                    mat_p[var1+j] = (float)p[3*j+1]/base;
                    mat_p[step+var1+j] = (float)p[3*j+2]/base;
                    mat_p[2*step+var1+j] = (float)p[3*j]/base;
                }
            }
        }
    }
    return mt;
}

float * CNN::conv_relu(const float * in,conv_param & param,size_t row,size_t col)
{
    if(in == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The in parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return NULL;
    }
    const size_t pad = param.pad;
    const size_t stride = param.stride;
    const size_t kernel_size = param.kernel_size;
    const size_t in_channels = param.in_channels;
    const size_t out_channels = param.out_channels;
    float * p_weight = param.p_weight;
    float * p_bias = param.p_bias;
    size_t length1 = kernel_size*kernel_size;
    size_t length2 = ((row+2*pad-kernel_size)/stride+1)*((col+2*pad-kernel_size)/stride+1);
    MyMatrix * mtdata = NULL;
    MyMatrix * mtweights = NULL;
    float * out = new float[out_channels*length2]();
    float * temp_p = new float[in_channels*length2]();
    float * tempp = new float[length2]();
    mtweights = new MyMatrix(1,length1,1);
    mtdata = new MyMatrix(length1,length2,1);
    size_t check = 0;
    if(in_channels==16)
        check = 1;

    if(pad == 0)
    {
        for(size_t oc=0;oc<out_channels;oc++)
        {
            for(size_t ic=0;ic<in_channels;ic++)
            {
                // temp_p = out+oc*in_channels*length2+ic*length2;
                tempp = temp_p+ic*length2;
                mtweights->num = p_weight+ic*length1+oc*length1*in_channels;
                //set the matrix of weights
                size_t k = 0;
                for(size_t i=1;i<=row-2;i+=stride)
                {
                    for(size_t j=1;j<=col-2;j+=stride)
                    {
                        size_t var1 = ic*row*col+i*col+j;
                        mtdata->num[k+length2*0] = in[var1-col-1];
                        mtdata->num[k+length2*1] = in[var1-col];
                        mtdata->num[k+length2*2] = in[var1-col+1];
                        mtdata->num[k+length2*3] = in[var1-1];
                        mtdata->num[k+length2*4] = in[var1];
                        mtdata->num[k+length2*5] = in[var1+1];
                        mtdata->num[k+length2*6] = in[var1+col-1];
                        mtdata->num[k+length2*7] = in[var1+col];
                        mtdata->num[k+length2*8] = in[var1+col+1];
                        k++;
                    }
                }
                //set the matrix of data

                mul(mtweights,mtdata,tempp,check);   
            }

            for(size_t i=0;i<length2;i++)
            {
                for(size_t j=0;j<in_channels;j++)
                {
                    out[oc*length2+i] += temp_p[j*length2+i];
                }
                out[oc*length2+i] += p_bias[oc];
            }
        }
        delete []temp_p;
        delete []mtdata->num;
    }else
    {
        for(size_t oc=0;oc<out_channels;oc++)
        {
            for(size_t ic=0;ic<in_channels;ic++)
            {
                // temp_p = out+oc*in_channels*length2+ic*length2;
                tempp = temp_p+ic*length2;
                // cout<<temp_p[ic*length2]<<endl;
                mtweights->num = p_weight+ic*length1+oc*length1*in_channels;
                size_t k = 0;
                for(size_t i=0;i<=row-1;i+=stride)
                {
                    for(size_t j=0;j<=col-1;j+=stride)
                    {
                        size_t var1 = ic*row*col+i*col+j;
                            mtdata->num[k+length2*0] = in[var1-col-1];
                            mtdata->num[k+length2*1] = in[var1-col];
                            mtdata->num[k+length2*2] = in[var1-col+1];
                            mtdata->num[k+length2*3] = in[var1-1];
                            mtdata->num[k+length2*4] = in[var1];
                            mtdata->num[k+length2*5] = in[var1+1];
                            mtdata->num[k+length2*6] = in[var1+col-1];
                            mtdata->num[k+length2*7] = in[var1+col];
                            mtdata->num[k+length2*8] = in[var1+col+1];
                            if(i==0)
                            {
                                mtdata->num[k+length2*0] = 0.0f;
                                mtdata->num[k+length2*1] = 0.0f;
                                mtdata->num[k+length2*2] = 0.0f;
                            }
                            if(i==row-1)
                            {
                                mtdata->num[k+length2*6] = 0.0f;
                                mtdata->num[k+length2*7] = 0.0f;
                                mtdata->num[k+length2*8] = 0.0f;
                            }
                            if(j==0)
                            {
                                mtdata->num[k+length2*0] = 0;
                                mtdata->num[k+length2*3] = 0;
                                mtdata->num[k+length2*6] = 0;                                
                            }
                            if(j==col-1)
                            {
                                mtdata->num[k+length2*2] = 0;
                                mtdata->num[k+length2*5] = 0;
                                mtdata->num[k+length2*8] = 0;
                            }
                            k++;
                    }
                }
                mul(mtweights,mtdata,tempp,check);
            }

            for(size_t i=0;i<length2;i++)
            {
                for(size_t j=0;j<in_channels;j++)
                {
                    out[oc*length2+i] += temp_p[j*length2+i];
                }
                out[oc*length2+i] += p_bias[oc];
            }
        }
        delete []temp_p;
        delete []mtdata->num;
    }
    float * po = out;
    for(size_t i=0;i<out_channels*length2;i++)
    {
        if(*po<0)
            *po = 0;
        po++;
    }
    return out;
}

float * CNN::maxPooling(float * in,size_t row,size_t col,size_t channel)
{
    if(in == NULL)
    {
        fprintf(stderr, "File %s, Line %d, Function %s(): The in parameter is NULL.\n", __FILE__, __LINE__, __FUNCTION__);
        return NULL;
    }
    float * output = new float[row/2*col/2*channel]();
    for(size_t c=0;c<channel;c++)
    {
        size_t var2 = c*row*col;
        for(size_t i=0;i<row;i+=2)
        {
            size_t var1 = i/2*col/2+c*row/2*col/2;
            for(size_t j=0;j<col;j+=2)
            {
                float var3=in[i*col+j+var2];
                if(var3<in[i*col+j+var2+1])
                    var3=in[i*col+j+var2+1];
                if(var3<in[i*col+j+var2+col])
                    var3=in[i*col+j+var2+col];
                if(var3<in[i*col+j+var2+col+1])
                    var3=in[i*col+j+var2+col+1];
                output[var1+j/2] = var3;
            }
        }
    }
    return output;
}

void CNN::execute()
{
    MyMatrix * input = NULL;
    input = new MyMatrix(128,128,3);
    input = setInput(input);
    float * c1 = NULL;
    c1 = conv_relu(input->num,conv_params[0],128,128);
    float * m1 = NULL;
    m1 = maxPooling(c1,64,64,16);
    float * c2 = NULL;
    c2 = conv_relu(m1,conv_params[1],32,32);
    float * m2 = NULL;
    m2 = maxPooling(c2,30,30,32);
    float * c3 = NULL;
    c3 = conv_relu(m2,conv_params[2],16,16);

    float * result = NULL;
    result = new float[2]();
	for (size_t i=0;i<2048;i++)
	{
		result[0] += (fc_params[0].p_weight[i]*c3[i]);
		result[1] += (fc_params[0].p_weight[i + 2048]*c3[i]);
	}
	result[0] += fc_params[0].p_bias[0];
	result[1] += fc_params[0].p_bias[1];
	float background = 0;
    background = exp(result[0]) / (exp(result[0]) + exp(result[1]));
	float face = 0;
    face = exp(result[1]) / (exp(result[0]) + exp(result[1]));
    cout<<"Judging "<<FileName<<endl;
	cout << "background probability: " << background << endl;
	cout << "face probability: " << face << endl;
}

int main()
{
    double duration = 0;
    start = clock();
    CNN p5("../1.jpg",128,128,3);
    p5.execute();
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
    cout<<"time consumption: "<<duration<<"ms"<<endl;
}

//neurosim
//arcone convergetostate