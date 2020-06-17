#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <"getIQR.h">
#include <"getSD.h">

//enable this line to test
//using namespace std;

#ifndef KERNELFUNCTIONS_H
#define KERNELFUNCTIONS_H
double getKernelDensityftn (double Yvalue, vector<double> Xvector) {
    vectorLength = Xvector.size();
    double h;
    h = getKernelHftn(Xvector);
    double KofUsum;
    KofUsum = 0.0;
    double U;
    for (int i = 0; i < vectorLength; ++i) {
        U = (Yvalue - Xvector[i]) / h;
        KofUsum += getKofUftn(U);
    }
    double kernelDensity;
    kernelDensity = KofUsum / (h * vectorLength);
}

double getKernelHftn (vector<double> X)
{
    double sd;
    double iqr;
    sd  = getSDftn(X);
    cout << "The sd is: " << sd << endl;
    iqr = getIQRftn(X);
    cout << "The iqr is: " << iqr << endl;
    double minspread;
    cout << "The minspread is: " << iqr << endl;
    minspread = min(sd,iqr);
    double n;
    n = X.size();
    double fifthRootOfn;
    fifthRootOfn = pow(n,0.2);
    cout << "The fifth root of n is: " << fifthRootOfn << endl;
    double h;
    h = 0.9*minspread/fifthRootOfn;
    return(h);
}

double getKofUftn (double U){
    double KofU;
    if (U >= -1.0 && U <= 1.0) }
        KofU = 3.0*(1-U*U)/4.0;
    }
    else {
        KofU = 0.0;
    }
    cout << "K(u) is: " << KofU << endl;
    return(KofU);
}

#endif
// code to test
//int main(){
//    vector<double> datavector;
//    datavector = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}; 
//    double hTest;
//    hTest = getKernelHftn(datavector);
//    cout << "The value of h is: " << hTest << endl;
//}
