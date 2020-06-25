#ifndef EXPONENTIALFUNCTIONS_H
#define EXPONENTIALFUNCTIONS_H

// Matthew Lang
// 6-23-2020
// This file contains two functions involving exponential distributions

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

using namespace std;
/*
double exponential_pdf(double x_score, double lambda);
double exponential_cdf(double x_score, double lambda);
*/
// Main that tests both functions
/*
int main()
{
    double pdf_x_score;
    double pdf_lambda;
    double pdf_answer;

    cout << "\nTesting PDF Function" << endl;
    cout << "Enter a score X: ";
    cin >> pdf_x_score;
    cout << "Enter Lambda: ";
    cin >> pdf_lambda;
    pdf_answer = exponential_pdf(pdf_x_score, pdf_lambda);
    cout << "The PDF value of the given values is: " << setprecision(10) << pdf_answer << endl;

    double cdf_x_score;
    double cdf_lambda;
    double cdf_answer;

    cout << "\nTesting CDF Function" << endl;
    cout << "Enter a score X: ";
    cin >> cdf_x_score;
    cout << "Enter Lambda: ";
    cin >> cdf_lambda;
    cdf_answer = exponential_cdf(cdf_x_score, cdf_lambda);
    cout << "The CDF value of the given values is: " << setprecision(10) << cdf_answer << endl << endl;

    return 0;
}
*/
double exponential_pdf(double x_score, double lambda)
{
    double pdf_answer;
    pdf_answer = lambda * exp(-1 * lambda * x_score);
    return pdf_answer;
}

double exponential_cdf(double x_score, double lambda)
{
    double cdf_answer;
    cdf_answer = 1 - exp(-1 * lambda * x_score);
    return cdf_answer;
}

#endif
