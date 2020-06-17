#ifndef GETSDFTN_H
#define GETSDFTN_H
double getSDftn (vector<double> xVector) {
    double sumX; // sum of x values
    sumX = 0.0;
    
    double sumX2; // sum of x squared values
    sumX2 = 0.0;

    double squaredSumX; // squared sum of x
    squaredSumX = 0.0;

    double variance; // with n in denominator
    variance = 0.0;

    double sd; // standard deviation
    sd = 0.0;

    int vectorLength;
    vectorLength = xVector.size();

    for (int i=0; i < vectorLength ; ++i)
    {
        sumX  += xVector[i];
        sumX2 += xVector[i]*xVector[i];
    }

    squaredSumX = sumX * sumX;

    variance = (sumX2 - squaredSumX/vectorLength)/vectorLength;
    sd = sqrt(variance);

    return(sd);
}
#endif
