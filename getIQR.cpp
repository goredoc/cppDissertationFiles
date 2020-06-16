#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

double getIQRftn (vector<double> v)
{
    sort(v.begin(), v.end());
    for (const auto &i: v)
        cout << i << ' '<<endl;
    int L = v.size();
    cout << "The size of the vector is: " << L << endl;
    cout << "Element zero is: " << v[0] << endl;
    cout << "Element one is: " << v[1] << endl;
    int one = 1;
    int two = 2;
    int three = 3;
    int four = 4;
    int remainder = (L+one) % four;
    cout << "We are in case: " << remainder << endl;
    int iQ1;
    int iQ3;
    double IQR;
    double Q1;
    double Q3;
    switch (remainder) {
        case 0:
            cout << "Begin case 0 section" << endl;
            iQ1 = (L+one)/four;
            cout << "iQ1: " << iQ1 << endl;
            iQ3 = (L+one)-iQ1;
            cout << "iQ3: " << iQ3 << endl;
            IQR = v[iQ3-1] - v[iQ1-1];
            break;
        case 1:
            cout << "Begin case 1 section" << endl;
            iQ1 = L/4;
            iQ3 = L - iQ1;
            cout << "iQ1: " << iQ1 << endl;
            cout << "iQ3: " << iQ3 << endl;
            Q1 = 0.5 * (v[iQ1-1] + v[iQ1]);
            Q3 = 0.5 * (v[iQ3-1] + v[iQ3]);
            IQR = Q3 - Q1;
            break;
        case 2:
            cout << "Begin case 2 section" << endl;
            iQ1 = (L+3) / 4;
            iQ3 = (L+1) - iQ1;
            IQR = v[iQ3-1] - v[iQ1-1];
            break;
        case 3:
            cout << "Begin case 3 section" << endl;
            iQ1 = (L+two) / four;
            iQ3 = (L+one) - iQ1;
            IQR = v[iQ3] - v[iQ1];
            break;
    }
return(IQR);
}

//
//int main(){
//    vector<double> datavector;
//    datavector = {10,9,8,7,6,5,4,3,2,1};
//    double iqr;
//    iqr=getIQRftn(datavector);
//    cout << "The interquartile range is: " << iqr << endl << endl;
//}

