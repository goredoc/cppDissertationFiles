// Matthew Lang
// 6-19-2020
// This file contains three more functions, with a main that tests them.

//// g++ -Wall -I /Users/matthewlang/documents/CodingProjects/boost_1_73_0 morefunctions.cpp

// Since the first payment....
// 40 min zoom meeting
// Two hours working on last project (fillingvectors.cpp file)
// 30 min zoom meeting
// 3 hours on current project (morefunctions.cpp file)

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <boost/random.hpp>
#include <vector>
#include <fstream>
#include <algorithm>

using namespace std;

// Global variable declarations
vector <int> St;
vector <double> Vc;
vector <double> Vw;
vector <double> alpha;
vector <double> beta;

// (Old) Function Protocalls
double random_beta(double alpha, double beta);

double random_normal(double mean, double standard_deviation);

void filling(double rho, double A, double A2b, double meanVc, double meanVw, double stdV, int N);

// NEW FUNCTIONS ADDED Protocalls
double pdf_height(double mu, double sigma, double x);
double cdf_height(double mu, double sigma, double x);
void shuffle(vector <double> &x_vector);

// Main that tests
int main()
{
/*
    // Testing pdf_height function
    double mu1;
    double sigma1;
    double x1;
    double pdf;
    cout << "\nTESTING PDF FUNCTION" << endl;
    cout << "Enter a mu value: ";
    cin >> mu1;
    cout << "Enter a sigma value: ";
    cin >> sigma1;
    cout << "Enter a X value: ";
    cin >> x1;
    pdf = pdf_height (mu1, sigma1, x1);
    cout << "PDF Value: " << setprecision(10) << pdf << endl;

    // Testing cdf_height function
    double mu2;
    double sigma2;
    double x2;
    double cdf;
    cout << "\nTESTING CDF FUNCTION" << endl;
    cout << "Enter a mu value: ";
    cin >> mu2;
    cout << "Enter a sigma value: ";
    cin >> sigma2;
    cout << "Enter a X value: ";
    cin >> x2;
    cdf = cdf_height(mu2, sigma2, x2);
    cout << "CDF Value: " << setprecision(10) << cdf << endl;
*/

/*
    double rho;
    double A;
    double A2b;
    double meanVc;
    double meanVw;
    double stdV;
    int N;

    cout << "Enter your rho value: ";
    cin >> rho;
    cout << "Enter your A value: ";
    cin >> A;
    cout << "Enter your A2b value: ";
    cin >> A2b;
    cout << "Enter your meanVc value: ";
    cin >> meanVc;
    cout << "Enter your meanVw value: ";
    cin >> meanVw;
    cout << "Enter your stdV value: ";
    cin >> stdV;
    cout << "Enter your N value: ";
    cin >> N;

    filling(rho, A, A2b, meanVc, meanVw, stdV, N);

    cout << "\nElements in St vector" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << St[i];
        cout << endl;
    }

    cout << "\nElements in Vc vector" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << Vc[i];
        cout << endl;
    }

    cout << "\nElements in Vw vector" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << Vw[i];
        cout << endl;
    }

    cout << "\nElements in alpha vector" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << alpha[i];
        cout << endl;
    }

    cout << "\nElements in beta vector" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << beta[i];
        cout << endl;
    }
*/

/*
   // Testing Shuffle Function
    cout << "\nElements in Vc vector (origional)" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << Vc[i];
        cout << endl;
    }
    
    shuffle(Vc);

    cout << "\nElements in Vc vector (Shuffled once)" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << Vc[i];
        cout << endl;
    }
    
    shuffle(Vc);

    cout << "\nElements in Vc vector (Shuffled twice)" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << i << " element in vector: "; 
        cout << Vc[i];
        cout << endl;
    }
*/

/*
    ofstream outstream;
    outstream.open("numberstest.txt");
    for (int i = 0; i < N; i++)
    {
        outstream << setprecision(10) << Vc[i] << endl;
    }
    return 0;
*/
}

double random_beta(double alpha, double beta)
{
    double number;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    typedef boost::random::mt19937 RandomNumberGenerator;
    typedef boost::random::beta_distribution<> BetaDistribution;
    RandomNumberGenerator Rng(seed);
    BetaDistribution distribution(alpha, beta);
    number = distribution(Rng);
    return number;
}

double random_normal(double mean, double standard_deviation)
{
    double number;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    default_random_engine generator(seed);
    normal_distribution<double> distribution(mean, standard_deviation);
    number = distribution(generator);
    return number;
}


void filling(double rho, double A, double A2b, double meanVc, double meanVw, double stdV, int N)
{
    // Filling St vector
    int st_randnum;
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    srand(chrono::system_clock::now().time_since_epoch().count());
    for (int i = 0; i < N; i++)
    {
        st_randnum = rand() % 2;
        St.push_back(st_randnum);
    }

    // Filling Vc vector
    double vc_distribution;
    for (int i = 0; i < N; i++)
    {
        vc_distribution = random_normal(meanVc, stdV);
        while (vc_distribution < 0)
        {
            vc_distribution = random_normal(meanVc, stdV);
        }
        Vc.push_back(vc_distribution);
    }

    // Filling Vw vector
    double vw_distribution;
    for (int i = 0; i < N; i++)
    {
        vw_distribution = random_normal(meanVw, stdV);
        while (vw_distribution < 0)
        {
            vw_distribution = random_normal(meanVw, stdV);
        }
        Vw.push_back(vw_distribution);
    }

    // Filling Alpha vector
    int alpha_randnum;
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    srand(chrono::system_clock::now().time_since_epoch().count());
    for (int i = 0; i < N; i++)
    {
        alpha_randnum = rand() % 2;
        alpha.push_back(alpha_randnum);
    }

    // Filling Beta vector
    int beta_randnum;
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    srand(chrono::system_clock::now().time_since_epoch().count());
    for (int i = 0; i < N; i++)
    {
        beta_randnum = rand() % 2;
        beta.push_back(beta_randnum);
    }
}

double pdf_height(double mu, double sigma, double x)
{
    double part1;
    double part2;
    double answer;

    part1 = 1 / sqrt(2 * M_PI * sigma * sigma);
    part2 = exp( (-1 * (x - mu) * (x - mu) ) / ( 2 * sigma * sigma ));
    answer = part1 * part2;
    return answer;
}

double cdf_height(double mu, double sigma, double x)
{
    double answer;
    answer = erf( (x - mu ) / ( sigma * sqrt(2) ) );
    answer ++;
    answer = answer / 2;
    return answer;
}
void shuffle(vector <double> &x_vector)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    default_random_engine generator(seed);
    shuffle(x_vector.begin(), x_vector.end(), generator);
}
