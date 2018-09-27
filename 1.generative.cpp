#include <iostream>
using namespace std;
#include<fstream>
#include<sstream>
#include<string>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include<vector>
#include <Eigen/LU>
#include <Eigen/Dense>
using Eigen::MatrixXd;

float sigmoid(float x)
{
     float exp_value;
     float return_value;

     /*** Exponential calculation ***/
     exp_value = exp((double) -x);

     /*** Final sigmoid value ***/
     return_value = 1 / (1 + exp_value);

     return return_value;
}

int main()
{

    float data[960][5];
    float testdata[960][5];
    std::ifstream file("train.txt");

    //Importing training data

    for(int row = 0; row < 960; ++row)
    {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);
        
        for (int col = 0; col < 5; ++col)
        {
            std::string val;

            std::getline(iss, val, ',');
            
            std::stringstream convertor(val);
            convertor >> data[row][col];
        	
        }
    }

    //Importing testing data.

    std::ifstream file2("test.txt");

    for(int row = 0; row < 412; ++row)
    {
        std::string line;
        std::getline(file2, line);
        if ( !file2.good() )
            break;

        std::stringstream iss(line);
        
        for (int col = 0; col < 5; ++col)
        {
            std::string val;

            std::getline(iss, val, ',');
            
            std::stringstream convertor(val);
            convertor >> testdata[row][col];
        	
        }
    }

	MatrixXd alldata(960,5);
	MatrixXd x(960,4);
	MatrixXd t(960,1);

	for(int i=0;i<960;i++){
		for(int j=0;j<5;j++){
			alldata(i,j)=data[i][j];
		}
		for(int j=0;j<4;j++){
			x(i,j)=data[i][j];
		}
	}

	for(int i=0;i<960;i++){
		t(i,0)=data[i][4];
	}

	MatrixXd alldatatest(412,5);
	MatrixXd xtest(412,4);
	MatrixXd ttest(412,1);

	for(int i=0;i<412;i++){
		for(int j=0;j<5;j++){
			alldatatest(i,j)=testdata[i][j];
		}
		for(int j=0;j<4;j++){
			xtest(i,j)=testdata[i][j];
		}
	}

	for(int i=0;i<412;i++){
		ttest(i,0)=testdata[i][4];
	}

	t.transposeInPlace();
	ttest.transposeInPlace();
	

	int positiveCases=425;
	int negativeCases=535;
	int totalCases=900;

	float pi=(float)positiveCases/(float)totalCases;

	float pc1=pi;
	float pc2=1-pi;

	MatrixXd mean1(1,4);
	mean1=t*x;
	mean1=mean1/positiveCases;
	

	MatrixXd ones(1,960);
	ones.setOnes();
	MatrixXd mean2(1,4);
	mean2=(ones-t)*x;
	mean2=mean2/negativeCases;

	MatrixXd mean1temp(4,960);
	mean1temp=mean1.replicate(960,1);
	MatrixXd val1(4,960);
	MatrixXd val1t(4,960);
	MatrixXd s1(4,4);
	val1=x-mean1temp;

	for(int i =0;i<960;i++){
		if(alldata(i,4)==0){
			val1.row(i).setZero();
		}
	}

	val1t=val1.transpose();
	s1=val1t*val1;

	MatrixXd mean2temp(4,960);
	mean2temp=mean2.replicate(960,1);
	MatrixXd val2(4,960);
	MatrixXd val2t(4,960);
	MatrixXd s2(4,4);
	val2=x-mean2temp;

	for(int i =0;i<960;i++){
		if(alldata(i,4)==1){
			val2.row(i).setZero();
		}
	}

	val2t=val2.transpose();
	s2=val2t*val2;

	MatrixXd sigma(4,4);
	sigma=(s1+s2)/960;

	MatrixXd sigmainv(4,4);
	sigmainv=sigma.inverse();

	mean1.transposeInPlace(); //4x1
	mean2.transposeInPlace(); //4x1

	MatrixXd w(4,1);
	w=sigmainv*(mean1-mean2); //4x1

	MatrixXd w0(1,4);

	w0=(-0.5*mean1.transpose()*sigmainv*mean1)+(0.5*mean2.transpose()*sigmainv*mean2);
	float fw0=w0(0,0)+log(pc1/pc2);

	cout<<"w0 value:"<<fw0<<"\n";
	cout<<"w value:"<<"\n"<<w<<"\n";
	
	xtest.transposeInPlace();
	
	float truepos=0;
	float trueneg=0;
	float falsepos=0;
	float falseneg=0;

	for(int i =0;i<412;i++){
		MatrixXd temp(1,1);
		temp=w.transpose()*xtest.col(i);
		float value=temp(0,0)+fw0;
		if(sigmoid(value)>0.5 && ttest(0,i)==1){
			truepos++;
		}else if(sigmoid(value)<0.5 && ttest(0,i)==0){
			trueneg++;
		}else if(sigmoid(value)<0.5 && ttest(0,i)==1){
			falsepos++;
		}else if(sigmoid(value)>0.5 && ttest(0,i)==0){
			falseneg++;
		}
	}

	float accuracy=(truepos+trueneg)/(truepos+trueneg+falsepos+falseneg);
	float precisionpos=(truepos)/(truepos+falsepos);
	float precisionneg=(trueneg)/(trueneg+falseneg);
	float recallpos=(truepos)/(truepos+falseneg);
	float recallneg=(trueneg)/(trueneg+falsepos);

	cout<<"Accuracy on testing data is:"<<accuracy*100<<"%\n";
	cout<<"Precision pos on testing data is:"<<precisionpos*100<<"%\n";
	cout<<"Recall pos on testing data is:"<<recallpos*100<<"%\n";
	cout<<"Precision neg on testing  data is:"<<precisionneg*100<<"%\n";
	cout<<"Recall neg on testing  data is:"<<recallneg*100<<"%\n";
	cout<<falsepos+falseneg<<"\n";

	x.transposeInPlace();
	
	truepos=0;
	trueneg=0;
	falsepos=0;
	falseneg=0;

	for(int i =0;i<960;i++){
		MatrixXd temp(1,1);
		temp=w.transpose()*x.col(i);
		float value=temp(0,0)+fw0;
		if(sigmoid(value)>0.5 && t(0,i)==1){
			truepos++;
		}else if(sigmoid(value)<0.5 && t(0,i)==0){
			trueneg++;
		}else if(sigmoid(value)<0.5 && t(0,i)==1){
			falseneg++;
		}else if(sigmoid(value)>0.5 && t(0,i)==0){
			falsepos++;
		}
	}

	accuracy=(truepos+trueneg)/(truepos+trueneg+falsepos+falseneg);
	precisionpos=(truepos)/(truepos+falsepos);
	precisionneg=(trueneg)/(trueneg+falseneg);
	recallpos=(truepos)/(truepos+falseneg);
	recallneg=(trueneg)/(trueneg+falsepos);

	cout<<"Accuracy on training data is:"<<accuracy*100<<"%\n";
	cout<<"Precision pos on training data is:"<<precisionpos*100<<"%\n";
	cout<<"Recall pos on training data is:"<<recallpos*100<<"%\n";
	cout<<"Precision neg on training data is:"<<precisionneg*100<<"%\n";
	cout<<"Recall neg on training data is:"<<recallneg*100<<"%\n";
	cout<<falsepos+falseneg<<"\n";


	return 0;
}