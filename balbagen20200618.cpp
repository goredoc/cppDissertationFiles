// Robert Gore with help from Matthew Lang
// June 18, 2020

// g++ -Wall -I /Users/matthewlang/documents/CodingProjects/boost_1_73_0 fillingvectors.cpp
// g++ -Wall -I /Users/OUConline/Desktop/Dissertation/boost_1_73_0 fillingvectors.cpp


#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
#include <boost/random.hpp>
#include <vector>

using namespace std;

// Global variable declarations

// Parameters to recover
// These are for generating a data set Y
// yA: end of range of possible start points
// yb: decision boundary
// yrho: memory retention autoregression parameter for starting point beta distribution
// ymuVc: average for drift rate distribution toward correct responses
// ymuVw: average for drift rate distribution toward wrong responses
// ysigmaV: standard deviation for both drift rate distributions
// yT0: non decision time (zero for now)


vector <int> St;
vector <double> Vc;
vector <double> Vw;
vector <double> alpha;
vector <double> beta;


// Declare the number of groups of particles
const int nGroups=10;

// Declare the number of particles in each group
const int nParticlesPerGroup=10;

// Declare the number of trials in a target Y dataset
//  to be created in simulation
const int nTrialsPerStudy=10000;

// Declare the number of trials in an X dataset
const int nTrialsPerSimulation=10000;

// Declare the number of iterations
const int nIterations=1000;

// Declare the number of variables to record
// sequence: 
// 0  empty for now
// 1  St(imulus right=1, left=0)
// 2  Ch(oice right=1, left=0)
// 3  RT - response time
// 4  corrYN (correct=1, wrong=0)
// 5  alphaL (alpha parameter of beta start distribution left response)
// 6  alphaR (alpha parameter for right response)
// 7  betaL  (beta parameter for left response)
// 8  betaR  (beta parameter for right response)
// 9  Vc (drift rate toward correct on this trial)
// 10 Vw (drift rate toward wrong response)
// 11 Zc (starting point toward correct response)
// 12 Zw (starting point toward wrong response)
const int  nRecordedVariables=13;

// Declare a multidimensional array to hold chains
// nGroups - number of groups
// nParticlesPerGroup
// nRecordedVariables - number of variables recorded
// nIterations - number of iterations
double mcmc.chains [nGroups][nParticlesPerGroup][nRecordedVariables][nIterations];

// Declare a multidimensional array for temporary holding
//  during migration (swapping)
// nGroups - number of groups
// nParticlesPerGroup
// nRecordedVariables - number of variables recorded
double holdingPen [nGroups][nParticlesPerGroup][nRecordedVariables];

// FclearHoldingPen fills the holding pen with zeroes
void FclearHoldingPen()
{
    for (i5416 = 0; i5416 < nGroups; i5416++)
        {
            for (j5094=0; j5094 < nParticlesPerGroup; j5094++)
            {
                for (k2380=0; k2380 < nRecordedVariables; k2380++)
                {
                    holdingPen[i5416][j5094][k2380]=0.0;
                }
            }
        }
}

// Declare a multidimensional array to hold simulation data
// nGroups - number of groups
// nParticlesPerGroup
// 2: 0=existing 1=proposed
// nRecordedVariables
// nTrialsPerSimulation
double xVector [nGroups][nParticlesPerGroup][2][nRecordedVariables][nTrialsPerSimulation]; 

// Function Protocalls
vector<double> Fswaparoo (vector<double> invector);

double random_beta(double alpha, double beta);

double random_normal(double mean, double standard_deviation);

void filling(double rho, double A, double A2b, double meanVc, double meanVw, double stdV, int N, int group, int particle, int proposalYN);

// Main that tests filling function
int main()
{
    double rho;
    double t0;
    double A;
    double A2b;
    double meanVc;
    double meanVw;
    double stdV;
    int N;


    // declare global variables
    // for creating target dataset Y
    // and declare default values
    //   rrho defaults to zero which is the LBA
    double yrho;
    yrho = 0.0; 

    //   yt0 defaults to zero
    double yt0;
    yt0 = 0.0;

    double yA;
    yA = 1.0;

    double yb;
    yb = 1.2;

    double ymuVc;
    ymuVc = 1.7;

    double ymuVw;
    ymuVw = 1.0;

    double ysigmaV;
    ysigmaV = 0.35;

    int createTargetSetYN ;
    createTargetSetYN = 9 ;
    while (createTargetSetYN != 0 && createTargetSetYN != 1)
      {
        cout << "Questions will be repeated if you enter invalid values\n\n\n";
        cout << "Do you want to create a target data set for fitting? \n\n";
        cout << "1=yes \n";
        cout << "0=no  \n";
        cin  >> createTargetSetYN;
        cout << endl;
      }  

    int useDefaultParametersYN ;
    useDefaultParametersYN = 9 ;
    while (useDefaultParametersYN != 0 && useDefaultParametersYN != 1)
      {
        cout << "Questions will be repeated if you enter invalid values\n\n\n";
        cout << "Do you want to use the default parameters to create target data?\n\n"
        cout << "The default parameters are as follows: \n";
        cout << "yrho = 0\n";
        cout << "yt0  = 0\n";
        cout << "yA = 1\n";
        cout << "yb = 1.2\n";
        cout << "ymuVc = 1.7\n";
        cout << "ymuVw = 1.0\n";
        cout << "ysigmaV = .35\n";
        cin  >> useDefaultParametersYN;
        cout << endl;

        if (useDefaultParameters == 0)
          {
            cout << "Questions will be repeated if you enter invalid values\n\n\n";
            yrho = -1;
            while (yrho < 0 || yrho > 1)
              {
                cout << "RHO: what is the memory parameter, yrho?\n";
                cout << "-----default: 0, the LBA-----\n";
                cin  << yrho;
                cout << endl;
              }
            yt0 = -1;
            while (yt0 < 0 || yt0 > 10000)
              {
                cout << "t0: what is the non-decision time, yt0?\n";
                cout << "-----default: 0-----\n";
                cout << "BE CAREFUL ABOUT WHETHER YOU ENTER SEC OR MSEC!";
                cin  << yt0;
                cout << endl;
              }
            yA = -1;
            while (yA < 0 || yA > 10000)
              {
                cout << "A: what is the end point of the range of possible starting values, yA?\n";
                cout << "-----default: 1.0-----\n";
                cin  >> yA;
                cout << endl;
              }
            yb = -1;
            while (yb < 0 || yb > 10000 || yb < yA)
              {
                cout << "b: what is the decision boundary (must be at least A), yb? \n";
                cout << "-----default: 1.2-----\n";
                cin  >> yb;
                cout << endl;
              }
            ymuVc = -1;
            while (ymuVc <= 0 || ymuVc > 10000)
              {
                cout << "muVc: mean of the correct drift rate, ymuVc? \n";
                cout << "-----default: 1.7-----\n";
                cin  >> ymuVc;
                cout << endl;
              }
            ymuVw = -1;
            while (ymuVw <=0 || ymuVw > 10000 || ymuVw > ymuVc)
              {
                cout << "muVw: mean of the wrong drift rate, ymuVw? \n";
                cout << "-----default: 1.0-----\n";
                cin  >> ymuVw;
                cout << endl;
              }
            ysigmaV = -1;
            while (ysigmaV < 0 || ysigmaV > 10000)
              {
                cout << "sigmaV: standard dev of drift rates (c and w), ysigmaV? \n";
                cout << "-----default: 0.35-----\n";
                cin  >> ysigmaV;
                cout >> endl;
              }
          }



      }

    //cout << "Enter your rho value: ";
    //cin >> rho;
    rho = .7;

    //cout << "Enter your A value: ";
    //cin >> A;
    A = 1.0;

    //cout << "Enter your A2b value: ";
    //cin >> A2b;
    A2b = .5;

    //cout << "Enter your meanVc value: ";
    //cin >> meanVc;
    meanVc = 1.2;

    //cout << "Enter your meanVw value: ";
    //cin >> meanVw;
    meanVw = .8;

    //cout << "Enter your stdV value: ";
    //cin >> stdV;
    stdV = .2;

    //cout << "Enter your N value: ";
    //cin >> N;
    N = nTrialsPerSimulation;

    filling(rho, A, A2b, meanVc, meanVw, stdV, N, 1, 1, 0);

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

    for (int j = 0; j < nGroups; j++)
    {
        for (int k = 0; k < nParticlesPerGroup; k++)
        {
            filling (rho, A, A2b, meanVc, meanVw, stdV, N, j, k, 0);
        }
    }

    // write a csv file with results
    ofstream output_file;
    output_file.open("simData.csv");
    output_file << "group, particle, trial, St, Ch, RT, corrYN, alphaL, alphaR, betaL, betaR, Vc, Vw, Zc, Zw \r";

    for (int j = 0; j < nGroups; j++)
    {
        for (int k = 0; k < nParticlesPerGroup; k++)
        {
            for (int m = 0; m < N; m++)
            {
                output_file << j << ",";
                output_file << k << ",";
                output_file << m << ",";
                output_file << xVector[j][k][0][1][m] << ","; 
                output_file << xVector[j][k][0][2][m] << ",";
                output_file << xVector[j][k][0][3][m] << ",";
                output_file << xVector[j][k][0][4][m] << ",";
                output_file << xVector[j][k][0][5][m] << ",";
                output_file << xVector[j][k][0][6][m] << ",";
                output_file << xVector[j][k][0][7][m] << ",";
                output_file << xVector[j][k][0][8][m] << ",";
                output_file << xVector[j][k][0][9][m] << ",";
                output_file << xVector[j][k][0][10][m] <<",";
                output_file << xVector[j][k][0][11][m] <<",";
                output_file << xVector[j][k][0][12][m] <<"\r";
            }
        }
    }
    output_file.close();

    return 0;
}

double random_beta(double alpha, double beta)
{
    double number;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::microseconds(1));
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
    this_thread::sleep_for(std::chrono::microseconds(1));
    default_random_engine generator(seed);
    normal_distribution<double> distribution(mean, standard_deviation);
    number = distribution(generator);
    return number;
}


void filling(double rho, double A, double A2b, double meanVc, double meanVw, double stdV, int N, int group, int particle, int proposalYN)
{
    // first passage time wrong
    double fptWrong;

    // first passage time correct
    double fptCorrect;

    // Filling St vector
    double st_randnum;
    this_thread::sleep_for(std::chrono::microseconds(1));
    srand(chrono::system_clock::now().time_since_epoch().count());

    for (int i = 0; i < N; i++)
    {
        st_randnum = rand() % 2;
        St.push_back(st_randnum);
        xVector[group][particle][proposalYN][1][i] = St[i];
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
        xVector[group][particle][proposalYN][9][i] = Vc[i];;
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
        xVector[group][particle][proposalYN][10][i]=Vw[i];
    }

    // Filling race matrix
    for (int i = 0; i < N ; i++)
    {
        if (i==1)
        {
        xVector[group][particle][proposalYN][5][i]=1; // alphaL
        xVector[group][particle][proposalYN][7][i]=1; // betaL
        xVector[group][particle][proposalYN][6][i]=1; // alphaR
        xVector[group][particle][proposalYN][8][i]=1; // betaR
        }

        // update alphaLR and betaLR if last response correct
        if (xVector[group][particle][proposalYN][4][i-1]==1.0)
        {
            // last stimulus was left side
            if (xVector[group][particle][proposalYN][1][i-1]==0.0)
            {
                xVector[group][particle][proposalYN][5][i]=rho*xVector[group][particle][proposalYN][5][i-1] + 1;
                xVector[group][particle][proposalYN][6][i]=rho*xVector[group][particle][proposalYN][6][i-1];
                xVector[group][particle][proposalYN][7][i]=rho*xVector[group][particle][proposalYN][7][i-1];
                xVector[group][particle][proposalYN][8][i]=rho*xVector[group][particle][proposalYN][8][i-1];
            }
            else // we know last stimulus was on the right side
            {
                xVector[group][particle][proposalYN][5][i]=rho*xVector[group][particle][proposalYN][5][i-1];
                xVector[group][particle][proposalYN][6][i]=rho*xVector[group][particle][proposalYN][6][i-1] + 1;
                xVector[group][particle][proposalYN][7][i]=rho*xVector[group][particle][proposalYN][7][i-1];
                xVector[group][particle][proposalYN][8][i]=rho*xVector[group][particle][proposalYN][8][i-1];
            }
        }
        // update alphaLR and betaLR if last response wrong
        else // we know the last response was wrong
        {
            xVector[group][particle][proposalYN][5][i]=1;
            xVector[group][particle][proposalYN][6][i]=1;
            xVector[group][particle][proposalYN][7][i]=1;
            xVector[group][particle][proposalYN][8][i]=1;
        }
        if (xVector[group][particle][proposalYN][1][i]==0)
        // stimulus is on the left
        {
            //Zc
            xVector[group][particle][proposalYN][11][i]=random_beta(xVector[group][particle][proposalYN][5][i],xVector[group][particle][proposalYN][7][i]);

            //Zw
            xVector[group][particle][proposalYN][12][i]=random_beta(xVector[group][particle][proposalYN][6][i],xVector[group][particle][proposalYN][8][i]);
        }
        else // stimulus is on the right
        {
            //Zc
            xVector[group][particle][proposalYN][11][i]=random_beta(xVector[group][particle][proposalYN][6][i],xVector[group][particle][proposalYN][8][i]);

            //Zw
            xVector[group][particle][proposalYN][12][i]=random_beta(xVector[group][particle][proposalYN][5][i], xVector[group][particle][proposalYN][7][i]);
        }
        fptCorrect = ( (A + A2b) - xVector[group][particle][proposalYN][11][i] ) / xVector[group][particle][proposalYN] [9][i]; 
        fptWrong   = ( (A + A2b) - xVector[group][particle][proposalYN][12][i] ) / xVector[group][particle][proposalYN][10][i];
        xVector[group][particle][proposalYN][3][i] = min(fptCorrect, fptWrong);
        if (xVector[group][particle][proposalYN][3][i] == fptCorrect)
        {  // response is correct
            xVector[group][particle][proposalYN][4][i] = 1.0; // corrYN =1
            xVector[group][particle][proposalYN][2][i] = xVector[group][particle][proposalYN][1][i]; // Ch = St
        }
        else // response is wrong
        {
            xVector[group][particle][proposalYN][4][i] = 0.0; // corrYN =0
            xVector[group][particle][proposalYN][2][i] = 1.0 - xVector[group][particle][proposalYN][1][i]; // Ch = 1-St
        } 
    }    

    // Filling Alpha vector
    int alpha_randnum;
    this_thread::sleep_for(std::chrono::microseconds(1));
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        alpha_randnum = rand() % 2;
        alpha.push_back(alpha_randnum);
    }

    // Filling Alpha vector
    int beta_randnum;
    srand(chrono::system_clock::now().time_since_epoch().count());
    this_thread::sleep_for(std::chrono::microseconds(1));
    for (int i = 0; i < N; i++)
    {
        beta_randnum = rand() % 2;
        beta.push_back(beta_randnum);
    }
}

// Input to this function is a vector of particle identifiers
// Left of the decimal is the group
// Right of the decimal is the particle
// Assume no more than 1000 groups and 1000 particles
// Vector can be only as long as the number of groups
// And must be at least 3
// Output is this vector having been rotated

vector<double> Fswaparoo (vector<double> invector)
    {
        int vecsize;
        vecsize = invector.size();
        vector<double> outvector (vecsize);
        vector<int> inIndex  (vecsize);
        vector<int> outIndex (vecsize);
        for (int i = 0; i < vecsize; i++)
        {
            inIndex[i]  = i;
            outIndex[i] = (i + 1) % vecsize;
            int outIndexScalar;
            outIndexScalar = outIndex[i];
            outvector[outIndexScalar] = invector[i];
            cout << "i: " << i << "\n";
            cout << "inIndex[i]: " << inIndex[i] << "\n\n";
            cout << "outIndex[i]: " << outIndex[i] << "\n\n";
            cout << "invector[i]: " << invector[i] << "\n\n";
            cout << "outvector: " << outvector[outIndexScalar] << "\n\n";
            cout << "---------------------------------------------\n\n\n";
        }
    return(outvector);
    }

// code to test Fswaparoo
//int main()
//    {
//        vector<double> inputVector;
//        double vecinput;
//        vecinput = 0;
//        int j;
//        j = -1;
//        while (vecinput != -999)
//        {
//            j++;
//            cout << "Enter a vector element (or -999 to discontinue): " << "\n\n";
//            cin  >> vecinput;
//            if (vecinput != -999)
//            {
//                inputVector.push_back(vecinput);
//            }
//        }
//        vector<double> outputVector( j );
//        outputVector = Fswaparoo (inputVector);
//        for (int i = 0; i < j; i++)
//        {
//            cout << "outputVector: " << outputVector[i] << "\n\n";
//        }
//        int sizeOfOutputVector;
//        sizeOfOutputVector = outputVector.size();
//        cout << "Size of Output Vector: " << sizeOfOutputVector << "\n\n";
//    }
// end of code to test Fswaparoo
