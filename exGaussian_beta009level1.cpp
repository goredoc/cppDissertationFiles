//  We aren't taking covariance into consideration in q-prob

// 009: last version did not converge on right values
//      try disabling the exponential aspect of the model
//      so it is just trying to fit a normal

#include<algorithm>
#include<boost/random.hpp>
#include<chrono>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<fstream>
#include<iomanip>
#include<iostream>
#include<math.h>
#include<random>
#include<sstream>
#include<string>
#include<thread>
#include<vector>

#include "cFrandom_normal.h"

using namespace std;

const int NGROUPS = 5;
const int NPARTICLESPERGROUP = 5;
const int NITERATIONS = 10000;
const int BURNIN      = NITERATIONS / 2; // at end of burn in crossover is simplified
const int NPARAMETERS = 3;
const int NPARAMETERSANDONE = NPARAMETERS + 1; // adds a slot for the PLL
const int NPARAMETERSLOTS = NPARAMETERS + 2;
const int NMCMCPARAMETERS = 5;
const int PLLSLOT = NPARAMETERS + 1;

class Holdingclass
{
public:
    double parameter [::NGROUPS][::NPARTICLESPERGROUP][::NPARAMETERS][::NITERATIONS];
    double pll[::NGROUPS][::NPARTICLESPERGROUP][::NITERATIONS];
    double usedYN[::NGROUPS][::NPARTICLESPERGROUP][::NITERATIONS];
    double newYN[::NGROUPS][::NPARTICLESPERGROUP][::NITERATIONS];
    double failedYN[::NGROUPS][::NPARTICLESPERGROUP][::NITERATIONS];
    double prior[::NGROUPS][::NPARTICLESPERGROUP][::NITERATIONS];
};

Holdingclass HMIGRATION;
Holdingclass HMUTATION;
Holdingclass HCROSSOVER;
Holdingclass HMCMCCHAINS;

// Function protocalls
double cdf_normal(double mu, double sigma, double x);
double computeMHprob (int group, int particle, int iteration, int longVersionYN, double logQratio, int stage);
double computeTransProbQs (int group, int particle, int iteration, int stage);

vector<int> FchooseGroups2Migrate ();
void   Fcrossover (int group, int particle, int iteration, int burninYN);
//void   FdefaultHoldingPen (int iteration);
void   FdefaultHoldingPen3 (int iteration, int whichpen);
void FdecideOnJump (int group, int particle, int iteration, int stage, int longversionYN);
double FexGaussian (double mu, double sigma, double lambda);
void   fFilling (int iteration);
double FholdingPenPriorCalculation (int group, int particle, int iteration, int stage);
double FlogPriorProb (int group, int particle, int iteration);
double Fmad (vector<double> v);
void   fMakeMCMC ();
double FMCMCpriorCalculation (int group, int particle, int iteration);
void   Fmutation (int groupNumber, int particleNumber, int iterationID);
void   FPLL (int iterationi);
double FPLLparticle (vector<double> xdata);
double Frandom01();
double FrandomExponential(double lambda);
double FsampleLambdaFromPrior ();
double FsampleMuFromPrior ();
double FsampleSigmaFromPrior ();
int    FselectParticleXFitness (vector<double> FitnessVector, int inverseYN, int logYN);
vector<double> Fswaparoo (vector<double> invector);
void   FupdateMCMC (int group, int particle, int iteration, int stage);
void FwriteParticlePLL (int group, int particle, int iteration, int mcmcchainsYN, int stage);

double getKernelDensityftn (double Yvalue, vector<double> Xvector);
double getKernelHftn (vector<double> X);
double getKofUftn (double U);
double getIQRftn (vector<double> v);
double getSDftn (vector<double> xVector);

double pdf_normal(double mu, double sigma, double x);

void shuffle(vector <double> &x_vector);
void shuffle_int(vector <int> &x_vector);
void shuffle_double(vector <int> &x_vector);

// Overall Control Paramters
const double PMIGRATE = 0.10;
const double PMUTATE = 0.10;
const double KAPPA = .90; // prob of a particular particle crossing over in a crossover step

// Define parameters to be recovered
const double MU     = 100.0;
const double SIGMA  = 15.0;
const double LAMBDA = 0.0;

const int NTRIALSPERSTUDY = 100;
const int NTRIALSPERSIMULATION = 1000;

// Priors parameters
// prior is normal on all variables
const double PRIORMEANOFMU     = 100;
const double PRIORSDOFMU       = 10;
const double PRIORMEANOFSIGMA  = 15;
const double PRIORSDOFSIGMA    = 5;
const double PRIORMEANOFLAMBDA = 10;
const double PRIORSDOFLAMBDA   = 3;
const double PRIORMEANSVEC[3]  = {PRIORMEANOFMU, PRIORMEANOFSIGMA, PRIORMEANOFLAMBDA};
const double PRIORSDSVEC[3]    = {PRIORSDOFMU  , PRIORSDOFSIGMA,   PRIORSDOFLAMBDA  };

//const int EXTREMELOG     =  250;
const int NEGEXTREMELOG  = -250;
const double EXTREMEZ    =  22.18; // for 250 as extremelog
const double NEGEXTREMEZ = -22.18;

const int NEGEXTREMEPLL = NEGEXTREMELOG * NTRIALSPERSTUDY;

// extra slots for likelihood
// and whether this is new or old parameter set

double MCMCCHAINS[NGROUPS][NPARTICLESPERGROUP][NPARAMETERSLOTS][NITERATIONS];
double HOLDINGPEN[NGROUPS][NPARTICLESPERGROUP][NPARAMETERSLOTS];
double MCMCPRIORS[NGROUPS][NPARTICLESPERGROUP][NITERATIONS];
double HOLDINGPENPRIORS[NGROUPS][NPARTICLESPERGROUP];
double XDATA[NGROUPS][NPARTICLESPERGROUP][NTRIALSPERSIMULATION];
double XPARTICLE[NTRIALSPERSIMULATION];
double YDATA[NTRIALSPERSTUDY];
double TEMPXVECTOR[NTRIALSPERSIMULATION];

// mutation step parameters
double MUTATESDMU = 15;
double MUTATESDSIGMA = 3;
double MUTATESDLAMBDA = .01;
// The line below was the problem preventing optimization
// No values had been set in the lines above
double SDMUTATEVECTOR[3] = {MUTATESDMU, MUTATESDSIGMA, MUTATESDLAMBDA};
double LOLIMITS[2][3] = {{1,1,1},{0,0,0}};
// first we have whether or not a lower limit then the vector of limits
// 999 means no limits but not important
double HILIMITS[2][3] = {{0,0,0},{999,999,999}};

int main() 
{
    //    test getKernelDensityftn
    //    double testYvalue = 500.0;
    //    vector<double> testXvector;
    //    testXvector.push_back(209);
    //    testXvector.push_back(872);
    //    testXvector.push_back(618);
    //    testXvector.push_back(788);
    //    testXvector.push_back(116);
    //
    //    double kernelH;
    //    kernelH = getKernelHftn(testXvector);
    //    cout << kernelH << endl;
    //
    //    cout << "K(u) values" << endl;
    //    cout << getKofUftn( 1.56 ) << endl;
    //    cout << getKofUftn(-1.99 ) << endl;
    //    cout << getKofUftn(-.6314) << endl;
    //    cout << getKofUftn(-1.54 ) << endl;
    //    cout << getKofUftn( 2.05 ) << endl;
    //
    //    double kerneldensity;
    //    kerneldensity = getKernelDensityftn(testYvalue, testXvector);
    //    cout << kerneldensity << endl;
    //    return 0;
    
    // Main control loop
    //   Executes an exGaussian parameter recovery
    
    
    string runLogFile = "runlog.txt";
    ofstream myLogFile;
    myLogFile.open(runLogFile.c_str());
    
    vector<double> yVector;
    double gaussian;
    double exponential;
    double exgaussian;
    int goodMutateFound = 0;
    int loopcounter = 0;
    int looplimit;
    double newMutPLL;
    // Generate data based on parameters to be recovered
    // yVector is the target data for parameter recovery
    // It has length NTRIALSPERSTUDY
    for (int i = 0; i < ::NTRIALSPERSTUDY; i++)
    {
        gaussian = Frandom_normal(::MU,::SIGMA);
        //exponential = FrandomExponential(::LAMBDA) ;
        exgaussian = gaussian;
        // exgaussian = exponential + gaussian;
        yVector.push_back(exgaussian);
    }
    // Sort yVector to avoid unnecessary computations
    // when doing kernel based PDA
    sort(yVector.begin(),yVector.end());
    // The next loop makes YDATA (array) a copy of yVector (vector)
    for (int i = 0; i < ::NTRIALSPERSTUDY; i++)
    {
        ::YDATA[i] = yVector[i];
    }
    // Loop through the particles and groups choosing parameters based on priors
    // for iteration zero
    for (int i = 0; i < ::NGROUPS; i++)
    {
        for (int j = 0; j < ::NPARTICLESPERGROUP; j++)
        {
            ::HMCMCCHAINS.newYN[i][j][0] = 1;
            int loopdone = 0;
            while (loopdone == 0)
            {
                ::HMCMCCHAINS.parameter[i][j][0][0] = FsampleMuFromPrior();
                ::HMCMCCHAINS.parameter[i][j][1][0] = FsampleSigmaFromPrior();
                ::HMCMCCHAINS.parameter[i][j][2][0] = FsampleLambdaFromPrior();
                FwriteParticlePLL(i,j,0,1,0);
                double thisPLL = ::HMCMCCHAINS.pll[i][j][0];
                if (thisPLL > ::NEGEXTREMEPLL)
                {
                    loopdone = 1;
                }
            }
        }
    }
    
    fFilling(0); // fills X with data at iteration
    
    FPLL(0); // puts PLL in MCMC at iteration 0
    cout << "HMCMCCHAINS.newYN[0][0][0][0]=" << ::HMCMCCHAINS.newYN[0][0][0];
    cout << "HMCMCCHAINS.parameter[0][0][0][0]=" << ::HMCMCCHAINS.parameter[1][0][0][0];
    cout << "HMCMCCHAINS.parameter[0][0][1][0]=" << ::HMCMCCHAINS.parameter[2][0][1][0];
    cout << "HMCMCCHAINS.parameter[0][0][2][0]=" << ::HMCMCCHAINS.parameter[3][0][2][0];
    
    int iterationMinusOne;
    cout << "Entering loop through iterations. " << endl;
    for (int i88 = 1; i88 < ::NITERATIONS; i88++)
    {
        myLogFile << "BEGIN ITERATION \t \t \t" << i88 << "." << endl;
        cout << "Beginning iteration: " << i88 << endl;
        iterationMinusOne = i88 - 1;
        // initialize the MCMC chains values
        cout << "MCMC. Group: ";
        for (int g = 0; g < ::NGROUPS; g++)
        {
            cout << g << endl;
            for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
            {
                cout << "\n Particle: " << p << endl;
                ::HMCMCCHAINS.newYN[g][p][i88] = 0;
                ::HMCMCCHAINS.pll[g][p][i88] = ::HMCMCCHAINS.pll[g][p][iterationMinusOne];
                for (int x = 0; x < ::NPARAMETERS; x++)
                {
                    ::HMCMCCHAINS.parameter[g][p][x][i88] = ::HMCMCCHAINS.parameter[g][p][x][iterationMinusOne];
                    cout << "\n Parameter " << x << "=" << ::HMCMCCHAINS.parameter[g][p][x][iterationMinusOne] << endl;
                }
            }
        }
        cout << "MCMC chains have now been initialized." << endl;
        // Initialize the holding pen values to MCMC chains at previous iteration
        FdefaultHoldingPen3(i88,1);
        FdefaultHoldingPen3(i88,2);
        FdefaultHoldingPen3(i88,3);
        
        // Once each iteration decide whether to do a migrate step
        double prob01 = Frandom01(); // NOTE THAT THE RANDOM FUNCTION NEEDS TO BE INCLUDED
        // ALSO NOTE THAT THE RANDOM FUNCTION REQUIRES CHRON PACKAGES
        if (prob01 < ::PMIGRATE)
        {
            cout << "A decision was made to migrate. Beginning migration." << endl;
            myLogFile << "Iteration: " << i88 << "\t BEGIN MIGRATION STEP " ;
            // EXECUTE THE MIGRATION STEP
            vector<int> groups2migrate;
            groups2migrate = FchooseGroups2Migrate();
            int number2migrate;
            number2migrate = groups2migrate.size();
            myLogFile << "WITH " << number2migrate << "GROUPS. " << endl;
            vector<double> groupAndParticleVector;
            groupAndParticleVector.clear();
            //double groupAndParticleVector[number2migrate];
            for (int g = 0; g < number2migrate; g++)
            {
                vector<double> fitnessvector;
                int chosenParticle;
                for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
                {
                    fitnessvector.push_back(::HMCMCCHAINS.pll[g][p][i88]);
                }
                chosenParticle = FselectParticleXFitness(fitnessvector,1,1);
                double tempvar198;
                tempvar198 = g + (chosenParticle/1000.0);
                groupAndParticleVector.push_back(tempvar198);
                fitnessvector.clear();
            }
            vector<double> postSwapVector;
            postSwapVector = Fswaparoo(groupAndParticleVector);
            // we are looping over groupAndParticleVector (post swaps)
            for (int g = 0; g < number2migrate; g++)
            {
                int hpgroup;
                int hpparticle;
                int mcmcgroup;
                int mcmcparticle;
                hpgroup    = trunc(groupAndParticleVector[g]);
                hpparticle = 1000 * (groupAndParticleVector[g] - hpgroup);
                mcmcgroup = trunc(postSwapVector[g]);
                mcmcparticle = 1000 * (postSwapVector[g] - mcmcgroup);
                
                ::HMIGRATION.usedYN[hpgroup][hpparticle][i88] = 1;
                for (int eloop = 0; eloop < ::NPARAMETERS; eloop++)
                {
                    ::HMIGRATION.parameter[hpgroup][hpparticle][eloop][i88] = ::HMCMCCHAINS.parameter[hpgroup][hpparticle][eloop][i88];
                }
                ::HMIGRATION.pll[hpgroup][hpparticle][i88] = ::HMCMCCHAINS.pll[hpgroup][hpparticle][i88];
                
                for (int eloop = 0; eloop < ::NPARAMETERS; eloop++)
                {
                    ::HMCMCCHAINS.parameter[mcmcgroup][mcmcparticle][eloop][i88] = ::HMIGRATION.parameter[hpgroup][hpparticle][eloop][i88];
                    if (eloop > 0)
                    {
                        ::HMCMCCHAINS.parameter[mcmcgroup][mcmcparticle][eloop][i88] = :: HMIGRATION.parameter[hpgroup][hpparticle][eloop][i88];
                    }
                }
                ::HMCMCCHAINS.pll[mcmcgroup][mcmcparticle][i88] = :: HMIGRATION.pll [hpgroup][hpparticle][i88];
                ::HMCMCCHAINS.newYN[mcmcgroup][mcmcparticle][i88] = 1; // says this is a new set of values
            }
            cout << "Migration step has ended." << endl;
            myLogFile << "END MIGRATION STEP. " << endl;
        }
        
        // Loop across groups
        // For each group decide whether or not to MIGRATE
        
        for (int g = 0; g < ::NGROUPS; g++)
        {
            myLogFile << "BEGIN GROUP LOOP: \t" << g << "." << endl;
            //for (int p = 0; p < ::NPARTICLESPERGROUP; p++) // we should not be iterating over p
            //{
            int p = 0;
            p = rand() % ::NPARTICLESPERGROUP;
            double tempLogQratio;
            // DECIDE WHETHER TO MUTATE
            double prob02 = Frandom01();
            if (prob02 < ::PMUTATE)
            {
                myLogFile << "BEGIN MUTATION. " << endl;
                myLogFile << "  EXISTING MCMCCHAINS VALUES ARE: " << endl;
                myLogFile << "\t \t Mu:" << HMCMCCHAINS.parameter[g][p][0][i88] << "\t Sigma: " << HMCMCCHAINS.parameter[g][p][1][i88] << "\t PLL: " << HMCMCCHAINS.pll[g][p][i88] << endl;
                cout << "A decision has been made to mutate." << endl;
                cout << "checkpoint andy" << endl;
                goodMutateFound = 0;
                loopcounter = 0;
                looplimit = 25;
                while (goodMutateFound == 0)
                {
                    loopcounter++;
                    Fmutation (g, p, i88);
                    FwriteParticlePLL(g,p,i88,0,1);
                    newMutPLL = HMUTATION.pll[g][p][i88];
                    myLogFile << "SOME MUTATION VALUES FOUND. " << endl;
                    myLogFile << "\t \t Mu:" << HMUTATION.parameter[g][p][0][i88] << "\t Sigma: " << HMUTATION.parameter[g][p][1][i88] << "\t PLL: " << HMUTATION.pll[g][p][i88] << endl;
                    if (newMutPLL > ::NEGEXTREMEPLL)
                    {
                        goodMutateFound = 1;
                        myLogFile << "GOOD MUTATION VALUES FOUND. " << endl;
                        myLogFile << "\t \t Mu:" << HMUTATION.parameter[g][p][0][i88] << "\t Sigma: " << HMUTATION.parameter[g][p][1][i88] << "\t PLL: " << HMUTATION.pll[g][p][i88] << endl;
                        FdecideOnJump(g, p, i88, 2, 1);
                    }
                    else
                    {
                        if (loopcounter == looplimit)
                        {
                            HMUTATION.failedYN[g][p][i88] = 1;
                            myLogFile << "MUTATION STEP ENDED BECAUSE LOOPLIMIT WAS REACHED. " << endl;
                            goodMutateFound  = 1;
                        }
                    }
                }
                // mutation stetp has ended;
            }
            // IF NOT MUTATE THEN CROSSOVER
            else
            {
                myLogFile << "BEGIN CROSSOVER. " << endl;
                myLogFile << "  EXISTING MCMCCHAINS VALUES ARE: " << endl;
                myLogFile << "\t \t Mu:" << HMCMCCHAINS.parameter[g][p][0][i88] << "\t Sigma: " << HMCMCCHAINS.parameter[g][p][1][i88] << "\t PLL: " << HMCMCCHAINS.pll[g][p][i88] << endl;
                cout << "A decision has been made to crossover." << endl;
                // g is group
                // i88 is iteration
                // CROSSOVER STEP
                // EXECUTE THE CROSSOVER STEP
                // DO THIS DIFFERENTLY FOR BURNIN VERSUS REST OF THEM
                // DURING BURNIN USE 4
                // DURING REMAINING USE 3
                // SELECT A PARTICLE IN DIRECT PROPORTION TO ITS FITNESS
                // SELECT THREE MORE PARTICLES
                // LOOP THROUGH THE PARAMETERS
                // DECIDE WHETHER TO CROSSOVER BASED ON KAPPA
                // CROSSOVER WHEN NEEDED
                // int triple[3];
                //int baseparticle;
                //int burninYN;
                int goodCrossValues = 0;
                loopcounter = 0;
                looplimit = 25;
                double crossPLL;
                if (i88 < ::BURNIN)
                {
                    goodCrossValues = 0;
                    while (goodCrossValues == 0)
                    {
                        loopcounter++;
                        Fcrossover (g,p,i88,0);
                        FwriteParticlePLL(g,p,i88,0,3);
                        crossPLL = HCROSSOVER.pll[g][p][i88];
                        myLogFile << "SOME CROSSOVER VALUES FOUND. " << endl;
                        myLogFile << "\t \t Mu:" << HCROSSOVER.parameter[g][p][0][i88] << "\t Sigma: " << HCROSSOVER.parameter[g][p][1][i88] << "\t PLL: " << HCROSSOVER.pll[g][p][i88] << endl;
                        if (crossPLL > ::NEGEXTREMEPLL)
                        {
                            goodCrossValues = 1;
                            myLogFile << "GOOD CROSSOVER VALUES FOUND. " << endl;
                            myLogFile << "\t \t Mu:" << HCROSSOVER.parameter[g][p][0][i88] << "\t Sigma: " << HCROSSOVER.parameter[g][p][1][i88] << "\t PLL: " << HCROSSOVER.pll[g][p][i88] << endl;
                            FdecideOnJump (g, p, i88, 3, 1);
                        }
                        else
                        {
                            if (loopcounter == looplimit)
                            {
                                myLogFile << "CROSSOVER LOOP LIMIT REACHED. ENDING CROSSOVER WITH FAILURE STATE. " << endl;
                                HCROSSOVER.failedYN[g][p][i88] = 1;
                                goodCrossValues = 1;
                            }
                        }
                    }
                }
                else // what to do if not in burnin
                {
                    goodCrossValues = 0;
                    while (goodCrossValues == 0)
                    {
                        loopcounter++;
                        Fcrossover (g,p,i88,1);
                        myLogFile << "SOME CROSSOVER VALUES FOUND. " << endl;
                        FwriteParticlePLL(g,p,i88,0,3);
                        crossPLL = HCROSSOVER.pll[g][p][i88];
                        if (crossPLL > ::NEGEXTREMEPLL)
                        {
                            goodCrossValues = 1;
                            myLogFile << "GOOD CROSSOVER VALUES FOUND. " << endl;
                            FdecideOnJump(g, p, i88, 3, 1);
                        }
                        else
                        {
                            if (loopcounter == looplimit)
                            {
                                myLogFile << "LOOP LIMIT REACHED. ENDING CROSSOVER. " << endl;
                                HCROSSOVER.failedYN[g][p][i88] = 1;
                                goodCrossValues = 1;
                            }
                        }
                    }
                }
                //--not in burnin   }
                // new PLL needs to be calculated based on new values
                cout << "Checkpoint VALERIE. " << endl;
                cout << "g (group): " << g << endl;
                // why                   cout << "thetaTindex (particle): " << thetaTindex << endl;
                cout << "i88 (iteration): " << i88 << endl;
                // why                   FwriteParticlePLL(g,thetaTindex,i88);
                cout << "Checkpoint WILLIAM. " << endl;
                // why                   ::MCMCCHAINS[g][thetaTindex][0][i88]=1; // update new particle indicator
            }
            //}
        }
    }
    
    
    // WRITE MCMC FILES WITH RESULTS
    cout << "Checkpoint XAVIER. " << endl;
    myLogFile.close();
    fMakeMCMC();
}

// FdefaultHoldingPen fills holding pen with contents of mcmcchains at start of iteration loop
// This function is superceded by FdefaultHoldingPen3
//void FdefaultHoldingPen (int iteration)
//{
//    for (int g = 0; g < NGROUPS; g++)
//    {
//        for (int p = 0; p < NPARTICLESPERGROUP; p++)
//        {
//            for (int t = 0; t < NPARAMETERSLOTS; t++)
//            {
//                ::HOLDINGPEN[g][p][t] = ::MCMCCHAINS[g][p][t][iteration-1];
//            }
//        }
//    }
//}

void FdefaultHoldingPen3 (int iteration, int whichpen)
{
    // whichpen == 1 MIGRATION
    // whichpen == 2 MUTATION
    // whichpen == 3 CROSSOVER
    for (int g = 0; g < NGROUPS; g++)
    {
        for (int p = 0; p < NPARTICLESPERGROUP; p++)
        {
            switch (whichpen)
            {
                case 1:
                    ::HMIGRATION.usedYN[g][p][iteration] = 0;
                    ::HMIGRATION.pll[g][p][iteration] = ::HMCMCCHAINS.pll[g][p][iteration - 1];
                case 2:
                    ::HMUTATION.usedYN[g][p][iteration] = 0;
                    ::HMUTATION.pll[g][p][iteration] = ::HMCMCCHAINS.pll[g][p][iteration - 1];
                case 3:
                    ::HCROSSOVER.usedYN[g][p][iteration] = 0;
                    ::HCROSSOVER.pll[g][p][iteration] = ::HMCMCCHAINS.pll[g][p][iteration - 1];
            }
            for (int t = 0; t < NPARAMETERS; t++)
            {
                switch (whichpen)
                {
                    case 1:
                        ::HMIGRATION.parameter[g][p][t][iteration] = ::HMCMCCHAINS.parameter[g][p][t][iteration-1];
                    case 2:
                        ::HMUTATION.parameter[g][p][t][iteration] = ::HMCMCCHAINS.parameter[g][p][t][iteration-1];
                    case 3:
                        ::HCROSSOVER.parameter[g][p][t][iteration] = ::HMCMCCHAINS.parameter[g][p][t][iteration-1];
                }
            }
        }
    }
}


void fFilling (int iteration)
{
    // Loop through particles, groups, and trials per simulation simulating observations
    for (int i = 0; i < ::NGROUPS; i++)
    {
        for (int j = 0; j < ::NPARTICLESPERGROUP; j++)
        {
            for (int k = 0; k < ::NTRIALSPERSIMULATION; k++)
            {
                double gaussian;
                double exponential;
                gaussian = Frandom_normal(::HMCMCCHAINS.parameter[i][j][0][iteration],::HMCMCCHAINS.parameter[i][j][1][iteration]);
                while (gaussian <= 0.0)
                {
                    gaussian = Frandom_normal(::HMCMCCHAINS.parameter[i][j][0][iteration],::HMCMCCHAINS.parameter[i][j][1][iteration]);
                }
                exponential = 0.0;
                //exponential = FrandomExponential(::HMCMCCHAINS.parameter[i][j][2][iteration]);
                ::XDATA [i][j][k] = gaussian + exponential;
                cout << "XDATA: ITERATION="<< iteration <<", group="<< i <<", particle="<< j <<", trial="<< k << endl;
            }
        }
    }
}

double FexGaussian (double mu, double sigma, double lambda)
{
    double gaussian    = Frandom_normal(mu, sigma);
    while (gaussian < 0)
    {
        gaussian = Frandom_normal(mu, sigma);
    }
    //double exponential = FrandomExponential (lambda);
    double exponential = 0.0;
    double exgaussian;
    exgaussian = exponential + gaussian;
    return exgaussian;
}

void FPLL (int iterationi)
{// Compute pseudo likelihood for each of the particles
    double particlePLL = 0.0;
    int everposYN = 0;
    double tempelement = 0.0;
    int sizeTtempxvector;
    vector<double> Ttempxvector;
    int exitswitch = 0;
    for (int i = 0; i < ::NGROUPS; i++)
    {
        for (int j = 0; j < ::NPARTICLESPERGROUP; j++)
        {
            particlePLL = 0;
            exitswitch = 0;
            for (int k = 0; k < ::NTRIALSPERSTUDY; k++)
            {
                Ttempxvector.clear();
                for (int m = 0; m < ::NTRIALSPERSIMULATION; m++)
                {
                    ::TEMPXVECTOR[m] = ::XDATA[i][j][m];
                }
                if (sizeof(TEMPXVECTOR[0]) == 0)
                {
                    cout << "Warning: division by zero in FPLL !" << endl;
                }
                sizeTtempxvector = sizeof(TEMPXVECTOR)/sizeof(TEMPXVECTOR[0]);
                // POTENTIAL ERROR DIVISION BY ZERO ABOVE
                for (int n = 0; n < sizeTtempxvector; n++)
                {
                    Ttempxvector.push_back(TEMPXVECTOR[n]);
                }
                cout << "Computing kernel density for group " << i << ", particle " << j << ", iteration " << i << endl;
                tempelement = getKernelDensityftn (::YDATA[k],Ttempxvector);
                //cout << "tempelement is: " << tempelement << endl;
                particlePLL = particlePLL + tempelement;
                //cout << "particlePLL is: " << particlePLL ;
                if (tempelement > ::NEGEXTREMELOG)
                {
                    everposYN = 1;
                }
                if (tempelement == ::NEGEXTREMELOG && everposYN == 1)
                {
                    cout << "Checkpoint Florence: we should exit the Y/X PLL loop" << endl;
                    particlePLL = particlePLL + (::NTRIALSPERSTUDY - k - 1)*::NEGEXTREMELOG;
                    exitswitch = 1;
                    k = ::NTRIALSPERSTUDY;
                }
            }
            cout << "Checkpoint Grayson: we should not just have passed through Florence. " << endl;
            //::MCMCCHAINS[i][j][::PLLSLOT][iterationi] = particlePLL;
            ::HMCMCCHAINS.pll[i][j][iterationi] = particlePLL;
            everposYN = 0;
        }
    }
}

#ifndef FPLLPARTICLE_H
#define FPLLPARTICLE_H
double FPLLparticle (vector<double> xdata)
{
    // this is a place where code could be made more efficient
    double particlePLL = 0.0;
    double tempelement = 0.0;
    for (int k125 = 0; k125 < ::NTRIALSPERSTUDY; k125++)
    {
        tempelement = getKernelDensityftn (::YDATA[k125],xdata);
        particlePLL = particlePLL + tempelement;
    }
    return particlePLL;
}
#endif

#ifndef FWRITEPARTICLEPLL_H
#define FWRITEPARTICLEPLL_H
void FwriteParticlePLL (int group, int particle, int iteration, int mcmcchainsYN, int stage)
{
    double pllOfParticle;
    vector<double> Txparticle;
    int lengthTxparticle;
    
    cout << "Checkpoint: entering FwriteParticlePLL. " << endl;
    cout << "Group: "     << group     << endl;
    cout << "Particle: "  << particle  << endl;
    cout << "Iteration: " << iteration << endl;
    
    double    tempvar450;
    double    mu;
    double    sigma;
    double    lambda;
    // algebra instead of if statements below
    switch (stage)
    {
        case 0: // iteration zero - pll of first draw from priors
            mu     = ::HMCMCCHAINS.parameter[group][particle][0][0];
            sigma  = ::HMCMCCHAINS.parameter[group][particle][1][0];
            lambda = ::HMCMCCHAINS.parameter[group][particle][2][0];
        case 1:
            mu     = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][0][iteration] + (1-mcmcchainsYN)*::HMIGRATION.parameter[group][particle][0][iteration];
            sigma  = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][1][iteration] + (1-mcmcchainsYN)*::HMIGRATION.parameter[group][particle][1][iteration];
            lambda = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][2][iteration] + (1-mcmcchainsYN)*::HMIGRATION.parameter[group][particle][2][iteration];
        case 2:
            mu     = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][0][iteration] + (1-mcmcchainsYN)*::HMUTATION.parameter[group][particle][0][iteration];
            sigma  = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][1][iteration] + (1-mcmcchainsYN)*::HMUTATION.parameter[group][particle][1][iteration];
            lambda = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][2][iteration] + (1-mcmcchainsYN)*::HMUTATION.parameter[group][particle][2][iteration];
        case 3:
            mu     = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][0][iteration] + (1-mcmcchainsYN)*::HCROSSOVER.parameter[group][particle][0][iteration];
            sigma  = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][1][iteration] + (1-mcmcchainsYN)*::HCROSSOVER.parameter[group][particle][1][iteration];
            lambda = mcmcchainsYN*::HMCMCCHAINS.parameter[group][particle][2][iteration] + (1-mcmcchainsYN)*::HCROSSOVER.parameter[group][particle][2][iteration];
    }
    //mu     = mcmcchainsYN*::HMCMCCHAINS[group][particle][1][iteration] + (1-mcmcchainsYN)*::HOLDINGPEN[group][particle][1];
    //sigma  = mcmcchainsYN*::HMCMCCHAINS[group][particle][2][iteration] + (1-mcmcchainsYN)*::HOLDINGPEN[group][particle][2];
    //lambda = mcmcchainsYN*::HMCMCCHAINS[group][particle][3][iteration] + (1-mcmcchainsYN)*::HOLDINGPEN[group][particle][3];
    for (int i = 0; i < NTRIALSPERSIMULATION; i++)
    {
        tempvar450 = FexGaussian (mu, sigma, lambda);
        ::XPARTICLE[i] = tempvar450;
    }
    lengthTxparticle = sizeof(::XPARTICLE)/sizeof(::XPARTICLE[0]);
    for (int j = 0; j < lengthTxparticle; j++)
    {
        Txparticle.push_back(::XPARTICLE[j]);
    }
    pllOfParticle = FPLLparticle(Txparticle);
    switch (mcmcchainsYN)
    {
        case 0:
            switch (stage)
            {
                case 0:
                    ::HMCMCCHAINS.pll[group][particle][0] = pllOfParticle;
                case 1:
                    ::HMUTATION.pll[group][particle][iteration] = pllOfParticle;
                case 2:
                    ::HMIGRATION.pll[group][particle][iteration] = pllOfParticle;
                case 3:
                    ::HCROSSOVER.pll[group][particle][iteration] = pllOfParticle;
            }
        case 1:
            ::HMCMCCHAINS.pll[group][particle][iteration] = pllOfParticle;
    }
    // the code below seems redundant
    // DELETE after code is working
    //    switch (stage)
    //    {
    //        case 0:
    //            // nothing to do here
    //        case 1:
    //            ::HMUTATION.pll[group][particle][iteration] = pllOfParticle;
    //        case 2:
    //            ::HMIGRATION.pll[group][particle][iteration] = pllOfParticle;
    //        case 3:
    //            ::HCROSSOVER.pll[group][particle][iteration] = pllOfParticle;
    //    }
}
#endif

void Fmutation (int groupNumber, int particleNumber, int iterationID)
{
    double    tempvar230;
    double    mu230;
    double    sigma230;
    double    lambda230;
    ::HMUTATION.usedYN[groupNumber][particleNumber][iterationID] = 1;
    int loopcounterminusone;
    for (int q = 0; q < ::NPARAMETERS; q++)
    {
        cout << "Checkpoint Fmutation, group=" << groupNumber << ", particle=" << particleNumber << ", iteration=" << iterationID << endl;
        double tempvar207 = ::HMCMCCHAINS.parameter[groupNumber][particleNumber][q][iterationID] + Frandom_normal(0,.1);
        if (::LOLIMITS[0][q] == 0)
        {
            if (::HILIMITS[0][q] == 0)
            {
                ::HMUTATION.parameter[groupNumber][particleNumber][q][iterationID] = tempvar207;
                cout << "Checkpoint Alonzo in Fmutation reached. " << endl;
            }
            else // upper limit but no lower one
            {
                while (tempvar207 > ::HILIMITS[1][q])
                {
                    tempvar207 = ::HMCMCCHAINS.parameter[groupNumber][particleNumber][q][iterationID] + Frandom_normal(0,.1);
                }
                ::HMUTATION.parameter[groupNumber][particleNumber][q][iterationID] = tempvar207;
                cout << "Checkpoint Bernardo in Fmutation reached. " << endl;
            }
        }
        else // there is a lower limit
        {
            while (tempvar207 < ::LOLIMITS[1][q])
            {
                tempvar207 = ::HMCMCCHAINS.parameter[groupNumber][particleNumber][q][iterationID] + Frandom_normal(0,.1);
                cout << "Checkpoint Carlo in Fmutation reached. " << endl;
            }
            if (::HILIMITS[0][q] == 1)
            {
                while (tempvar207 > ::HILIMITS[1][q])
                {
                    tempvar207 = ::HMCMCCHAINS.parameter[groupNumber][particleNumber][q][iterationID] + Frandom_normal(0,.1);
                }
            }
            ::HMUTATION.parameter[groupNumber][particleNumber][q][iterationID] = tempvar207;
            cout << "Checkpoint Daniella in Fmutation reached. " << endl;
        }
        loopcounterminusone = q - 1;
        HMUTATION.parameter[groupNumber][particleNumber][loopcounterminusone][iterationID] = tempvar207;
    }
    cout << "Checkpoint Eduardo in Fmutation reached. " << endl;
    mu230     = ::HMUTATION.parameter[groupNumber][particleNumber][0][iterationID];
    sigma230  = ::HMUTATION.parameter[groupNumber][particleNumber][1][iterationID];
    lambda230 = ::HMUTATION.parameter[groupNumber][particleNumber][2][iterationID];
    for (int i = 0; i < NTRIALSPERSIMULATION; i++)
    {
        tempvar230 = FexGaussian (mu230, sigma230, lambda230);
        ::XPARTICLE[i] = tempvar230;
    }
    cout << "Checkpoint Fernando in Fmutation reached. " << endl;
    double pllOfParticle;
    vector<double> Txparticle2;
    int lengthTxparticle2 = sizeof(::XPARTICLE)/sizeof(::XPARTICLE[0]);
    for (int i = 0; i < lengthTxparticle2; i++)
    {
        Txparticle2.push_back(::XPARTICLE[i]);
    }
    cout << "Checkpoint Gandolph in Fmutation reached. " << endl;
    pllOfParticle = FPLLparticle(Txparticle2);
    ::HMUTATION.pll[groupNumber][particleNumber][iterationID]=pllOfParticle;
    // decide whether to make the jump
    if (::HMUTATION.pll[groupNumber][particleNumber][iterationID] > ::HMCMCCHAINS.pll[groupNumber][particleNumber][iterationID])
    {
        for (int i240 = 0; i240 < ::NPARAMETERS; i240++)
        {
            ::HMCMCCHAINS.parameter[groupNumber][particleNumber][i240][iterationID] = ::HMUTATION.parameter[groupNumber][particleNumber][i240][iterationID];
            cout << "Checkpoint Harold in Fmutation reached. " << endl;
        }
        ::HMCMCCHAINS.newYN[groupNumber][particleNumber][iterationID] = 1;
        cout << "Checkpoint Ismereldo in Fmutation reached. " << endl;
    }
    else
    { // here we jump with an MH probability
        double testprob;
        testprob = Frandom01();
        double Ltestprob;
        Ltestprob = log(testprob);
        double LogDifference = ::HMUTATION.pll[groupNumber][particleNumber][iterationID] - ::HMCMCCHAINS.pll[groupNumber][particleNumber][iterationID];
        cout << "Checkpoint Jannine in Fmutation reached. " << endl;
        // SEGFAULT AFTER JANNINE
        cout << "::HMCMCCHAINS.newYN[0][0][" << iterationID << "] " <<  ::HMCMCCHAINS.newYN[0][0][iterationID] << endl;
        cout << "::HMUTATION.newYN[0][0][" << iterationID << "] " << ::HMUTATION.newYN[0][0][iterationID] << endl;
        cout << "Ltestprob " << Ltestprob << endl;
        cout << "LogDifference " << LogDifference << endl;
        if (Ltestprob < LogDifference)
        {
            cout << "Checkpoint JUMP " << endl;
            ::HMCMCCHAINS.pll[groupNumber][particleNumber][iterationID] = ::HMUTATION.pll[groupNumber][particleNumber][iterationID];
            for (int i = 0; i < ::NPARAMETERS; i++)
            {
                ::HMCMCCHAINS.parameter[groupNumber][particleNumber][i][iterationID] = ::HMUTATION.parameter[groupNumber][particleNumber][i][iterationID];
                cout << "Checkpoint Klondike in Fmutation reached. " << endl;
            }
            cout << "Checkpoint Kali reached. " << endl;
            ::HMCMCCHAINS.newYN[groupNumber][particleNumber][iterationID] = 1;
            cout << "Checkpoint Lyle in Fmutation reached. " << endl;
        }
    }
}


double FlogPriorProb (int group, int particle, int iteration)
{
    double logmupriorprob     ;
    double logsigmapriorprob  ;
    double loglambdapriorprob ;
    double mupriorprob     = pdf_normal (::PRIORMEANOFMU, ::PRIORSDOFMU ,::HMCMCCHAINS.parameter[group][particle][0][iteration]);
    if (mupriorprob == 0)
    {logmupriorprob = ::NEGEXTREMELOG;}
    else
    {
        logmupriorprob = log(mupriorprob);
        if (logmupriorprob < ::NEGEXTREMELOG)
        {logmupriorprob = ::NEGEXTREMELOG;}
    }
    double sigmapriorprob  = pdf_normal (::PRIORMEANOFSIGMA, ::PRIORSDOFSIGMA , ::HMCMCCHAINS.parameter[group][particle][1][iteration]);
    if (sigmapriorprob == 0)
    {logsigmapriorprob = ::NEGEXTREMELOG;}
    else
    {
        logsigmapriorprob = log(sigmapriorprob);
        if (logsigmapriorprob < ::NEGEXTREMELOG)
        {logsigmapriorprob = ::NEGEXTREMELOG;}
    }
    double lambdapriorprob = pdf_normal (::PRIORMEANOFLAMBDA,::PRIORSDOFLAMBDA,::HMCMCCHAINS.parameter[group][particle][2][iteration]);
    if (lambdapriorprob == 0)
    {loglambdapriorprob = ::NEGEXTREMELOG;}
    else
    {
        loglambdapriorprob = log(lambdapriorprob);
        if (loglambdapriorprob < ::NEGEXTREMELOG)
        {loglambdapriorprob = ::NEGEXTREMELOG;}
    }
    double logprior;
    logprior = logmupriorprob + logsigmapriorprob;
    //logprior = logmupriorprob + logsigmapriorprob + loglambdapriorprob;
    return logprior;
}

double FsampleMuFromPrior ()
{
    // select a mu from prior distribution
    double sampledmu;
    sampledmu = Frandom_normal (::PRIORMEANOFMU,::PRIORSDOFMU);
    while (sampledmu <= 0.0)
    {
        sampledmu = Frandom_normal (::PRIORMEANOFMU,::PRIORSDOFMU);
    }
    return sampledmu;
}

double FsampleSigmaFromPrior ()
{
    double sampledSigma;
    // select a sigma from prior distribution
    sampledSigma = Frandom_normal (::PRIORMEANOFSIGMA,::PRIORSDOFSIGMA);
    while (sampledSigma <= 0.0)
    {
        sampledSigma = Frandom_normal (::PRIORMEANOFSIGMA,::PRIORSDOFSIGMA);
    }
    return sampledSigma;
}

double FsampleLambdaFromPrior ()
{
    double sampledLambda;
    sampledLambda = -1;
    // select a lambda from prior distribution
    while (sampledLambda <= 0.0)
    {
        sampledLambda = Frandom_normal (::PRIORMEANOFLAMBDA,::PRIORSDOFLAMBDA);
    }
    return sampledLambda;
}

double computeTransProbQs (int group, int particle, int iteration, int stage)
{
    //return 0;
    vector<double> zOldMutate;
    vector<double> zNewMutate;
    double numerSum=0;
    double denomSum=0;
    
    // Changed to int i = 1, to int i = 0 in the for loop
    //for (int i = 1; i <= ::NPARAMETERS; i++)
    for (int i = 0; i < ::NPARAMETERS; i++)
    {
        cout << "Test 1" << endl;
        int lohiBoundyn = ::LOLIMITS[0][i]*10 + HILIMITS[0][i];
        cout << "Test 2" << endl;
        double oldMutateElement;
        oldMutateElement = ::HMCMCCHAINS.parameter[group][particle][i][iteration];
        double newMutateElement;
        switch (stage)
        {
            case 1:
                newMutateElement = ::HMIGRATION.parameter[group][particle][i][iteration];
            case 2:
                newMutateElement = ::HMUTATION.parameter[group][particle][i][iteration];
            case 3:
                newMutateElement = ::HCROSSOVER.parameter[group][particle][i][iteration];
        }
        cout << "Test 3" << endl;
        cout << "Size of zOldMutateVector: " << zOldMutate.size() << endl;
        
        // Changed these 2 lines to push_back
        zOldMutate.push_back((oldMutateElement - newMutateElement) / ::SDMUTATEVECTOR[i]);
        zNewMutate.push_back((newMutateElement - oldMutateElement) / ::SDMUTATEVECTOR[i]);
        
        
        cout << "Test 4" << endl;
        if      (zOldMutate[i] < ::NEGEXTREMEZ) {numerSum = numerSum + ::NEGEXTREMEZ;}
        else if (zOldMutate[i] >  ::EXTREMEZ) {numerSum = numerSum +    0.0;}
        else
        {
            switch(lohiBoundyn) // need to redo this
            {
                case 0: // bounded on neither side so num and denom cancel
                    break;
                case 1: // bounded only from above
                    // may not work due to taking log of a function return
                    numerSum = numerSum + log(cdf_normal (oldMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]));
                case 10: // bounded only from below
                    numerSum = numerSum + log(cdf_normal (::LOLIMITS[1][i], ::SDMUTATEVECTOR[i],oldMutateElement));
                case 11: // bounded on both sides
                    numerSum = numerSum + log (cdf_normal(oldMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]) - cdf_normal(oldMutateElement,::SDMUTATEVECTOR[i],::LOLIMITS[1][i]));
            }
        }
        if      (zNewMutate[i] < ::NEGEXTREMEZ) {denomSum = denomSum + ::NEGEXTREMEZ;}
        else if (zNewMutate[i] >  ::EXTREMEZ) {denomSum = denomSum +    0.0;}
        else
        {
            switch(lohiBoundyn)
            {
                case 0: // bounded on neither side so num and denom cancel
                    denomSum = denomSum + 0.0;
                case 1: // bounded only from above
                    // may not work due to taking log of a function return
                    denomSum = denomSum + log(cdf_normal (newMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]));
                case 10: // bounded only from below
                    denomSum = denomSum + log(cdf_normal (::LOLIMITS[1][i],::SDMUTATEVECTOR[i],newMutateElement));
                case 11: // bounded on both sides
                    denomSum = denomSum + log (cdf_normal(newMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]) - cdf_normal(newMutateElement,::SDMUTATEVECTOR[i],::LOLIMITS[1][i]));
            }
        }
    }
    return numerSum - denomSum; // returns the log of (q(thetaT|thetaStar)/q(thetaStar|thetaT))
}




#ifndef FWRITEMCMC_H
#define FWRITEMCMC_H
// Globally declared constants
void fMakeMCMC ()
{
    string filename;
    for (int g = 0; g < ::NGROUPS; g++)
    {
        for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
        {
            filename = "mcmc_" + std::to_string(g) + "_" + std::to_string(p) + ".csv";
            ofstream output_file;
            output_file.open(filename.c_str());
            output_file << "newInMCMC, MCMCMu, MCMCSigma, MCMCLambda, MCMCPLL, MigrationUsed, MigrateMu, MigrateSigma, MigrateLambda, MigratePLL, MutationUsed, MutateMu, MutateSigma, MutateLambda, MutatePLL, CrossUsed, CrossMu, CrossSigma, CrossLambda, CrossPLL  \r";
            for (int i = 0; i < ::NITERATIONS; i++)
            {
                output_file << ::HMCMCCHAINS.newYN[g][p][i] << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][0][i] << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][1][i] << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][2][i] << ",";
                output_file << ::HMCMCCHAINS.pll[g][p][i] << ",";
                output_file << ::HMIGRATION.usedYN[g][p][i] << ",";
                output_file << ::HMIGRATION.parameter[g][p][0][i] << ",";
                output_file << ::HMIGRATION.parameter[g][p][1][i] << ",";
                output_file << ::HMIGRATION.parameter[g][p][2][i] << ",";
                output_file << ::HMIGRATION.pll[g][p][i] << ",";
                output_file << ::HMUTATION.usedYN[g][p][i] << ",";
                output_file << ::HMUTATION.parameter[g][p][0][i] << ",";
                output_file << ::HMUTATION.parameter[g][p][1][i] << ",";
                output_file << ::HMUTATION.parameter[g][p][2][i] << ",";
                output_file << ::HMUTATION.pll[g][p][i] << ",";
                output_file << ::HCROSSOVER.usedYN[g][p][i] << ",";
                output_file << ::HCROSSOVER.parameter[g][p][0][i] << ",";
                output_file << ::HCROSSOVER.parameter[g][p][1][i] << ",";
                output_file << ::HCROSSOVER.parameter[g][p][2][i] << ",";
                output_file << ::HCROSSOVER.pll[g][p][i] << "\r";
            }
            output_file.close();
        }
    }
    filename = "mcmcOnefile.csv";
    ofstream output_file;
    output_file.open(filename.c_str());
    output_file << "group, particle, iteration, Mu, Sigma, Lambda, PLL, migrationUsed, mutationUsed, crossoverUsed  \r";
    for (int g = 0; g < ::NGROUPS; g++)
    {
        for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
        {
            for (int i = 0; i < :: NITERATIONS; i++)
            {
                output_file << g << ",";
                output_file << p << ",";
                output_file << i << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][0][i] << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][1][i] << ",";
                output_file << ::HMCMCCHAINS.parameter[g][p][2][i] << ",";
                output_file << ::HMCMCCHAINS.pll[g][p][i] << ",";
                output_file << ::HMIGRATION.usedYN[g][p][i] << ",";
                output_file << ::HMUTATION.usedYN[g][p][i] << "," ;
                output_file << ::HCROSSOVER.usedYN[g][p][i] << "\r";
            }
        }
    }
    output_file.close();
}
#endif


#ifndef FRANDOMEXPONENTIAL_H
#define FRANDOMEXPONENTIAL_H
double FrandomExponential(double lambda)
{
    double number;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::microseconds(1));
    default_random_engine generator(seed);
    exponential_distribution<double> distribution(lambda);
    number = distribution(generator);
    return number;
}

#endif

#ifndef FRANDOM01_H
#define FRANDOM01_H

double Frandom01()
{
    double number;
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::microseconds(1));
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0, 1);
    number = distribution(generator);
    return number;
}
#endif

#ifndef COMPUTEMHPROB_H
#define COMPUTEMHPROB_H
// This function assumes the existence of a normal CDF function tentatively called cnorm (score, mean, sd)
// The return is assumed to be a probability not a log probability
// computeMHprob returns a log probability in keeping with the principle of working always in logs
double computeMHprob (int group, int particle, int iteration, int longVersionYN, double logQratio, int stage)
{
    double logMHprob;
    double logPriorRatio;
    double logPLLratio;
    //::HOLDINGPENPRIORS[group][particle]      = FholdingPenPriorCalculation(group, particle);
    //::MCMCPRIORS[group][particle][iteration] = FMCMCpriorCalculation(group, particle, iteration);
    HMCMCCHAINS.prior[group][particle][iteration] = FMCMCpriorCalculation(group,particle,iteration);
    switch (stage)
    {
        case 1:
            ::HMIGRATION.prior[group][particle][iteration] = FholdingPenPriorCalculation(group,particle,iteration,1);
            logPriorRatio = ::HMIGRATION.prior[group][particle][iteration] - ::HMCMCCHAINS.prior[group][particle][iteration];
            logPLLratio = ::HMIGRATION.pll[group][particle][iteration] - ::HMCMCCHAINS.pll[group][particle][iteration];
        case 2:
            ::HMUTATION.prior[group][particle][iteration] = FholdingPenPriorCalculation(group,particle,iteration,2);
            logPriorRatio = ::HMUTATION.prior[group][particle][iteration] - ::HMCMCCHAINS.prior[group][particle][iteration];
            logPLLratio = ::HMIGRATION.pll[group][particle][iteration] - ::HMCMCCHAINS.pll[group][particle][iteration];
        case 3:
            ::HCROSSOVER.prior[group][particle][iteration] = FholdingPenPriorCalculation(group,particle,iteration,3);
            logPriorRatio = ::HCROSSOVER.prior[group][particle][iteration] - ::HMCMCCHAINS.prior[group][particle][iteration];
            logPLLratio = ::HCROSSOVER.pll[group][particle][iteration] - ::HMCMCCHAINS.pll[group][particle][iteration];
    }
    
    switch(longVersionYN)
    {
        case 0: // implements TS12 equation 6
            logMHprob = logPriorRatio + logPLLratio;
        case 1: // implements TS12 equation 3
            logMHprob = logPriorRatio +  logPLLratio + logQratio;
    }
    
    return logMHprob;
}
#endif

#ifndef FHOLDINGPENPRIORCALCULATION_H
#define FHOLDINGPENPRIORCALCULATION_H
double FholdingPenPriorCalculation (int group, int particle, int iteration, int stage)
{
    double logtempprob;
    double sumlogtempprob = 0.0;
    double tempprob = 0.0;
    double supportArea = 1.0;
    //    for (int i500 = 1; i500 <= 3; i500++)
    for (int i500 = 0; i500 < 2; i500++)
    {
        tempprob = 0.0;
        supportArea = 1.0;
        switch (stage)
        {
            case 1:
                tempprob = pdf_normal (::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HMIGRATION.parameter[group][particle][i500][iteration]);
            case 2:
                tempprob = pdf_normal (::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HMUTATION.parameter[group][particle][i500][iteration]);
            case 3:
                tempprob = pdf_normal (::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HCROSSOVER.parameter[group][particle][i500][iteration]);
        }
        //tempprob = pdf_normal (::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HOLDINGPEN[group][particle][i500]);
        if (LOLIMITS[0][i500] == 1)
        {
            supportArea = supportArea - cdf_normal(::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::LOLIMITS[1][i500]);
        }
        if (HILIMITS[0][i500] == 1)
        {
            supportArea = supportArea - (1.0 - cdf_normal(::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HILIMITS[1][i500]));
        }
        tempprob = tempprob / supportArea;
        if (tempprob > 0)
        {
            logtempprob = log(tempprob);
        }
        if (tempprob == 0)
        {
            logtempprob = ::NEGEXTREMELOG;
        }
        if (tempprob < 0)
        {
            logtempprob = ::NEGEXTREMELOG;
            cout << "WARNING: NEGATIVE PROBABILITY GENERATED IN FHOLDINGPENPRIORCALCULATION. " << endl;
        }
        if (logtempprob < ::NEGEXTREMELOG)
        {
            logtempprob = ::NEGEXTREMELOG;
        }
        sumlogtempprob = sumlogtempprob + logtempprob;
    }
    return sumlogtempprob;
}
#endif

#ifndef FMCMCPRIORCALCULATION_H
#define FMCMCPRIORCALCULATION_H
double FMCMCpriorCalculation (int group, int particle, int iteration)
{
    double sumlogtempprob = 0.0;
    double tempprob = 0.0;
    double logtempprob;
    double supportArea = 1.0;
    //    for (int i501 = 1; i501 <= 3; i501++)
    for (int i = 0; i < 2; i++)
    {
        tempprob = 0.0;
        supportArea = 1.0;
        tempprob = pdf_normal (::PRIORMEANSVEC[i],::PRIORSDSVEC[i],::MCMCCHAINS[group][particle][i][iteration]);
        if (LOLIMITS[0][i] == 1)
        {
            supportArea = supportArea - cdf_normal(::PRIORMEANSVEC[i],::PRIORSDSVEC[i],::LOLIMITS[1][i]);
        }
        if (HILIMITS[0][i] == 1)
        {
            supportArea = supportArea - (1.0 - cdf_normal(::PRIORMEANSVEC[i],::PRIORSDSVEC[i],::HILIMITS[1][i]));
        }
        tempprob = tempprob / supportArea;
        if (tempprob > 0)
        {
            logtempprob = log(tempprob);
        }
        if (tempprob == 0)
        {
            logtempprob = ::NEGEXTREMELOG;
        }
        if (tempprob < 0)
        {
            logtempprob = ::NEGEXTREMELOG;
            cout << "WARNING: NEGATIVE PROBABILITY GENERATED IN FMCMCPRIORCALCULATION! " << endl;
        }
        if (logtempprob < ::NEGEXTREMELOG)
        {
            logtempprob = ::NEGEXTREMELOG;
        }
        sumlogtempprob = sumlogtempprob + logtempprob;
    }
    return sumlogtempprob;
}
#endif


#ifndef FRANDDISCRETEUNIFORM_H
#define FRANDDISCRETEUNIFORM_H
int FrandDiscreteUniform (int Lower, int Upper)
{
    int modrange = Upper - Lower + 1;
    int output;
    output = Lower + (rand() % modrange);
    return output;
}
#endif

#ifndef FCHOOSEGROUPS2MIGRATE_H
#define FCHOOSEGROUPS2MIGRATE_H
vector<int> FchooseGroups2Migrate ()
{
    vector<double> baseGroupVector;
    vector<double> groupIDvector; // why is this double not int?
    vector<int> INTgroupIDvector;
    int numberOfGroups2Migrate = FrandDiscreteUniform (3, ::NGROUPS);
    for (int i = 0; i < ::NGROUPS; i++)
    {
        baseGroupVector.push_back(i);
    }
    shuffle(baseGroupVector);
    for (int j = 0; j < numberOfGroups2Migrate; j++)
    {
        groupIDvector.push_back(baseGroupVector[j]);
    }
    for (int k = 0; k < numberOfGroups2Migrate; k++) // this seems redundant
    {
        INTgroupIDvector.push_back(groupIDvector[k]);
    }
    return INTgroupIDvector;
}
#endif


#ifndef FSELECTPARTICLEXFITNESS_H
#define FSELECTPARTICLEXFITNESS_H
int FselectParticleXFitness (vector<double> FitnessVector, int inverseYN, int logYN)
{
    // random uniform normalized to sum of fitnesses in vector
    double normalizedRandom;
    // sum of the values in the fitness vector
    double uFitnessVectorSum;
    
    // length variable
    int lengthvec;
    lengthvec = FitnessVector.size();
    
    // converted away from log scale u is for use this
    // unlogged = u
    vector<double> uFitnessVector(lengthvec);
    
    // set to zero to start
    uFitnessVectorSum = 0;
    
    // cumulative total of uFitnessVector
    // to determine which is selected
    // based on random number chosen
    double cumuFitnessVector;
    
    // initialize to zero
    cumuFitnessVector = 0;
    
    // convert away from log scale if needed
    if (logYN==1)
    {
        for (int i = 0; i < FitnessVector.size(); ++i)
        {
            if (FitnessVector[i] > ::NEGEXTREMELOG)
            {
                uFitnessVector[i] = exp(FitnessVector[i]);
            }
            else
            {
                uFitnessVector[i] = 0.0;
            }
        }
    }
    else
    {
        for (int j=0; j < FitnessVector.size(); ++j)
        {
            uFitnessVector[j] = FitnessVector[j];
        }
    }
    // choose on inverse scale if needed
    if (inverseYN==1)
    {
        for (int k = 0; k < uFitnessVector.size(); ++k)
        {
            uFitnessVector[k] = 1 / (uFitnessVector[k]+exp(::NEGEXTREMELOG));
        }
    }
    // compute cumulative sum
    for (int m = 0; m < uFitnessVector.size(); m++)
    {
        uFitnessVectorSum = uFitnessVectorSum + uFitnessVector[m];
    }
    // at this point uFitnessVectorSum is available;
    //cout << "uFitnessVectorSum: " << uFitnessVectorSum << "\n\n";
    // normalizedRandom is between 0 and uFitnessVectorSum;
    // normalizedRandom = uFitnessVectorSum * rand();
    double randomvar = Frandom01();
    //cout << "randomvar: " << randomvar << "\n\n";
    normalizedRandom = randomvar * uFitnessVectorSum;
    //cout << "normalizedRandom: " << normalizedRandom << "\n\n";
    
    // define exit status for while loop
    int exitStatusTrue = 0;
    int i4908 = -1;
    while (exitStatusTrue==0)
    {
        // start the loop at 0
        // for vector indexing
        i4908++;
        
        // calculate running total
        cumuFitnessVector += uFitnessVector[i4908];
        
        if (cumuFitnessVector >= normalizedRandom)
        {
            exitStatusTrue=1;
        }
    }
    //cout << "cumuFitnessVector: " << cumuFitnessVector << "\n\n";
    
    // i4908 is the index value of the particle chosen
    return(i4908);
}
#endif

#ifndef FSWAPAROO_H
#define FSWAPAROO_H
vector<double> Fswaparoo (vector<double> invector)
{
    int vecsize = invector.size();
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
    }
    return(outvector);
}
#endif

#ifndef FWRITEMCMC_H
#define FWRITEMCMC_H
// Globally declared constants
void fMakeMCMC ()
{
    string filename;
    for (int g = 0; g < ::NGROUPS; g++)
    {
        for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
        {
            filename = "mcmc_" + std::to_string(g) + "_" + std::to_string(p) + ".csv";
            ofstream output_file;
            output_file.open(filename.c_str());
            output_file << "New, Mu, Sigma, Lambda, PLL \r";
            for (int i = 0; i < ::NITERATIONS; i++)
            {
                output_file << ::HMCMCCHAINS.newYN [g][p][i] << ",";
                output_file << ::HMCMCCHAINS.particle [g][p][0][i] << ",";
                output_file << ::HMCMCCHAINS.particle [g][p][1][i] << ",";
                output_file << ::HMCMCCHAINS.particle [g][p][2][i] << ",";
                output_file << ::HMCMCCHAINS.pll [g][p][i] << "\r";
            }
            output_file.close();
        }
    }
}

#endif

double pdf_normal(double mu, double sigma, double x)
{
    double part1;
    double part2;
    double answer;
    
    part1 = 1 / sqrt(2 * M_PI * sigma * sigma);
    part2 = exp( (-1 * (x - mu) * (x - mu) ) / ( 2 * sigma * sigma ));
    answer = part1 * part2;
    return answer;
}

double cdf_normal(double mu, double sigma, double x)
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

void shuffle_int(vector <int> &x_vector)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    default_random_engine generator(seed);
    shuffle(x_vector.begin(), x_vector.end(), generator);
}

void shuffle_double(vector <double> &x_vector)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    this_thread::sleep_for(std::chrono::nanoseconds(1));
    default_random_engine generator(seed);
    shuffle(x_vector.begin(), x_vector.end(), generator);
}

double getKernelDensityftn (double Yvalue, vector<double> Xvector)
{
    int vectorLength = Xvector.size();
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
    double logKernelDensity;
    if (kernelDensity <= 0)
    {
        logKernelDensity = ::NEGEXTREMELOG;
    }
    else
    {
        logKernelDensity = log(kernelDensity);
        if (logKernelDensity < ::NEGEXTREMELOG)
        {
            logKernelDensity = ::NEGEXTREMELOG;
        }
    }
    return logKernelDensity;
}

double Fmad (vector<double> v)
{
    int L = v.size();
    // compute the mean
    double vecsum=0;
    for (int i=0; i < L; i++)
    {
        vecsum += v[i];
        //cout << "X (now called v) vector element is: " << v[i] << endl;
    }
    double vecmean = vecsum / L;
    double adsum = 0.0;
    for (int j=0; j < L; j++)
    {
        double deviation = v[j] - vecmean;
        if (deviation < 0)
        {
            deviation = deviation * (-1.0);
        }
        adsum += deviation;
    }
    double mad = adsum / L;
    return mad;
}

double getKernelHftn (vector<double> X)
{
    double sd;
    double iqr;
    //double mad;
    //mad = Fmad(X);
    sd  = getSDftn(X);
    //cout << "The sd is: " << sd << endl;
    iqr = getIQRftn(X) / 1.34;
    //cout << "The iqr/1.34 is: " << iqr << endl;
    double minspread;
    //cout << "The minspread is: " << iqr << endl;
    minspread = min(sd,iqr);
    double n;
    n = X.size();
    double fifthRootOfn;
    fifthRootOfn = pow(n,0.2);
    //cout << "The fifth root of n is: " << fifthRootOfn << endl;
    double h;
    h = 0.9 * minspread / fifthRootOfn;
    //h = 0.9 * mad / fifthRootOfn;
    if (h == 0.0)
    {
        //h = 1.0;
        cout << "Warning: h was calcuated as zero !" << endl;
        cout << "Size of vector is: " << n << endl;
        //cout << "MAD is: " << mad << endl;
    }
    return(h);
}

double getSDftn (vector<double> xVector) {
    int vectorLength;
    vectorLength = xVector.size();
    
    double sumX; // sum of x values
    sumX = 0.0;
    
    double sumX2; // sum of x squared values
    sumX2 = 0.0;
    
    double squaredSumX; // squared sum of x
    squaredSumX = 0.0;
    
    double variance; // with n in denominator
    variance = 0.0;
    
    double stdv; // standard deviation
    stdv = 0.0;
    
    switch(vectorLength)
    {
        case 1:
            stdv = 0.0000001;
            break;
        default:
            for (int i=0; i < vectorLength ; i=i+1)
            {
                sumX  = sumX + xVector[i];
                sumX2 = sumX2 + xVector[i]*xVector[i];
            }
            squaredSumX = sumX * sumX;
            variance = (sumX2 / vectorLength) - (squaredSumX / (vectorLength*vectorLength));
            stdv = sqrt(variance);
    }
    return stdv;
}

double getIQRftn (vector<double> v)
{
    sort(v.begin(), v.end());
    //for (const auto &i: v)
    //    cout << i << ' '<<endl;
    int L = v.size();
    //cout << "The size of the vector is: " << L << endl;
    //cout << "Element zero is: " << v[0] << endl;
    //cout << "Element one is: " << v[1] << endl;
    int remainder = (L+1) % 4;
    int iQ1;
    int iQ3;
    double IQR;
    double Q1;
    double Q3;
    switch (L)
    {
        case 1:
            IQR = 0.0000001;
            break;
        case 2:
            IQR = abs(v[1]-v[0]);
            break;
        case 3:
            IQR = abs(v[2]-v[0]);
            break;
        case 4:
            IQR = abs(v[2]-v[1]);
        default:
            //cout << "We are in case: " << remainder << endl;
            switch (remainder) {
                case 0:
                    //cout << "Begin case 0 section" << endl;
                    iQ1 = (L+1)/4;
                    //cout << "iQ1: " << iQ1 << endl;
                    iQ3 = (L+1)-iQ1;
                    //cout << "iQ3: " << iQ3 << endl;
                    IQR = v[iQ3-1] - v[iQ1-1];
                    break;
                case 1:
                    //cout << "Begin case 1 section" << endl;
                    iQ1 = L/4;
                    iQ3 = L - iQ1;
                    //cout << "iQ1: " << iQ1 << endl;
                    //cout << "iQ3: " << iQ3 << endl;
                    Q1 = (v[iQ1-1] + v[iQ1])/2;
                    Q3 = (v[iQ3-1] + v[iQ3])/2;
                    IQR = Q3 - Q1;
                    break;
                case 2:
                    //cout << "Begin case 2 section" << endl;
                    iQ1 = (L+3) / 4;
                    iQ3 = (L+1) - iQ1;
                    IQR = v[iQ3-1] - v[iQ1-1];
                    break;
                case 3:
                    //cout << "Begin case 3 section" << endl;
                    iQ1 = (L+2) / 4;
                    iQ3 = (L+1) - iQ1;
                    IQR = v[iQ3] - v[iQ1];
                    break;
                    
            }
    }
    return(IQR);
}

double getKofUftn (double U){
    double KofU;
    if (U >= -1.0 && U <= 1.0)
    {
        KofU = 3.0*( 1 - U*U )/4.0;
    }
    else
    {
        KofU = 0.0;
    }
    //cout << "K(u) is: " << KofU << endl;
    return KofU;
}

void FupdateMCMC (int group, int particle, int iteration, int stage)
{
    int numequivalents=0; // number of parameters equal in mcmcchains and holdingpen
    for (int w = 0; w < NPARAMETERS; w++)
    {
        switch (stage)
        {
            case 1:
                if (::HMIGRATION.parameter[group][particle][w][iteration] == ::HMCMCCHAINS.parameter[group][particle][w][iteration])
                {
                    numequivalents++;
                }
            case 2:
                if (::HMUTATION.parameter[group][particle][w][iteration] == ::HMCMCCHAINS.parameter[group][particle][w][iteration])
                {
                    numequivalents++;
                }
            case 3:
                if (::HCROSSOVER.parameter[group][particle][w][iteration] == ::HMCMCCHAINS.parameter[group][particle][w][iteration])
                {
                    numequivalents++;
                }
        }
        //if (::HOLDINGPEN[group][particle][w] == ::MCMCCHAINS[group][particle][w][iteration])
        //{
        //    numequivalents++;
        //}
    }
    if (numequivalents == NPARAMETERS)
    {
        cout << "Exiting FupdateMCMC with return 0 - this may not work " << endl;
        return;
    }
    double tempLogQratio;
    tempLogQratio = computeTransProbQs (group, particle, iteration, stage) ;
    // CHECK TO MCMC PROBABILITY TO SEE IF TO KEEP IT
    //
    double logMHprob;
    logMHprob = computeMHprob (group, particle, iteration, 0, 0, stage);
    if (logMHprob > 0)
    { // keep it for sure
        ::HMCMCCHAINS.newYN[group][particle][iteration] = 1;
        switch (stage)
        {
            case 1:
                ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMIGRATION.pll[group][particle][iteration];
            case 2:
                ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMUTATION.pll[group][particle][iteration];
            case 3:
                ::HMCMCCHAINS.pll[group][particle][iteration] = ::HCROSSOVER.pll[group][particle][iteration];
        }
        for (int w = 0; w < ::NPARAMETERS; w++)
        {
            switch (stage)
            {
                case 1:
                    ::HMCMCCHAINS.parameter[group][particle][w][iteration] = HMIGRATION.parameter[group][particle][w][iteration];
                case 2:
                    ::HMCMCCHAINS.parameter[group][particle][w][iteration] = HMUTATION.parameter[group][particle][w][iteration];
                case 3:
                    ::HMCMCCHAINS.parameter[group][particle][w][iteration] = HCROSSOVER.parameter[group][particle][w][iteration];
            }
        }
    }
    else
    {
        double tempRand;
        tempRand = Frandom01();
        double logTempRand;
        logTempRand = log(tempRand);
        if (logTempRand < logMHprob)
        { // keep it
            ::HMCMCCHAINS.newYN[group][particle][iteration] = 1;
            switch (stage)
            {
                case 1:
                    ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMIGRATION.pll[group][particle][iteration];
                case 2:
                    ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMUTATION.pll[group][particle][iteration];
                case 3:
                    ::HMCMCCHAINS.pll[group][particle][iteration] = ::HCROSSOVER.pll[group][particle][iteration];
            }
            for (int w = 0; w < ::NPARAMETERS; w++)
            {
                switch (stage)
                {
                    case 1:
                        ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HMIGRATION.parameter[group][particle][w][iteration];
                    case 2:
                        ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HMUTATION.parameter[group][particle][w][iteration];
                    case 3:
                        ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HCROSSOVER.parameter[group][particle][w][iteration];
                }
            }
        }
    }
}

void Fcrossover (int group, int particle, int iteration, int burninYN)
{
    ::HCROSSOVER.usedYN[group][particle][iteration] = 1;
    
    double gamma1;
    double gamma2;
    vector<double> thetaStarVector;
    vector<double> thetaTvector;
    vector<double> thetaBvector;
    vector<double> thetaMvector;
    vector<double> thetaNvector;
    
    int tminusone;
    cout << "Entering Fcrossover at group=" << group << " particle=" << particle << " iteration=" << iteration << endl;
    // baseparticle is the one that pushes back into good parameter space
    // Theta B in Turner and Sederburg
    vector<double> fitnessvector2;
    int triple[3];
    int thetaBindex;
    int thetaTindex;
    int thetaMindex;
    int thetaNindex;
    cout << "Checkpoint ADAM in crossover " << endl;
    for (int p260 = 0; p260 < ::NPARTICLESPERGROUP; p260++)
    {
        // this loop creates a list of the numbers from 0 to NPARTICLESPERGROUP-1
        fitnessvector2.push_back(::HMCMCCHAINS.pll[group][particle][iteration]);
        cout << "Checkpoint BATHSHEBA in crossover. " << endl;
    }
    int baseparticle = FselectParticleXFitness(fitnessvector2,0,1);
    fitnessvector2.clear();
    vector<int> tempindexvector;
    cout << "Checkpoint CADMAEL in crossover. " << endl;
    for (int i = 0; i < ::NPARTICLESPERGROUP; i++)
    {
        // this loop creates a list of the numbers from 0 to NPARTICLESPERGROUP-1
        tempindexvector.push_back(i);
        cout << "Checkpoint DAVID in crossover. " << endl;
    }
    // shuffling the list of numbers will help us sample from them
    shuffle_int(tempindexvector);
    cout << "Checkpoint EZEKIEL in crossover. " << endl;
    //for (int s = 0; s < ::NPARTICLESPERGROUP; s++)
    //{
    // this is just a cout loop for debugging
    //}
    int donecheck = 0;
    // donecheck tells us whether we have all the particles we need (3)
    int runner = 0;
    // runner tells us whether we have more particles to consider
    cout << "Checkpoint FRANCIS in crossover. " << endl;
    while (donecheck < 3)
    {
        // the purpose of this while loop is to select three distinct particles
        // none of which is the baseparticle
        // selecting each with equal probability
        // the vector triple will contain the index values of the three
        if (tempindexvector[runner] != baseparticle)
            // we cannot use the baseparticle for either of the others
        {
            cout << "Checkpoint GABRIEL in crossover. " << endl;
            triple[donecheck] = tempindexvector[runner];
            donecheck++;
            cout << "Checkpoint HADASSAH in crossover. " << endl;
        }
        runner++;
        cout << "Checkpoint ISAAC in crossover. " << endl;
    }
    tempindexvector.clear();
    thetaStarVector.push_back(0);
    //thetaTvector.push_back(0);
    //thetaBvector.push_back(0);
    //thetaMvector.push_back(0);
    //thetaNvector.push_back(0);
    thetaBindex = baseparticle;
    thetaTindex = triple[0];
    thetaMindex = triple[1];
    thetaNindex = triple[2];
    int loopCounter=0;
    cout << "Checkpoint JEZEBEL in crossover. " << endl;
    int starProblem = 1; // default to problem state
    while (starProblem == 1)
    {
        starProblem = 0;
        cout << "Checkpoint KAAPO in crossover. " << endl;
        thetaStarVector.clear();
        //thetaStarVector.push_back(0);
        for (int T=0; T < ::NPARAMETERS; T++)
        {
            cout << "Checkpoint LACHLAN in crossover. " << endl;
            thetaTvector.push_back(::HMCMCCHAINS.parameter[group][thetaTindex][T][iteration]);
            thetaBvector.push_back(::HMCMCCHAINS.parameter[group][thetaBindex][T][iteration]);
            thetaMvector.push_back(::HMCMCCHAINS.parameter[group][thetaMindex][T][iteration]);
            thetaNvector.push_back(::HMCMCCHAINS.parameter[group][thetaNindex][T][iteration]);
        }
        //thetaStarVector.clear();
        //thetaStarVector.push_back(0);
        loopCounter = (loopCounter + 1) % 100;
        cout << "Checkpoint MOLOCH in crossover. " << endl;
        for (int t=0; t < ::NPARAMETERS; t++)
        {
            cout << "Checkpoint NADIA in crossover. " << endl;
            cout << "We are on parameter " << t << endl;
            if (Frandom01() < ::KAPPA)
            {
                cout << "Checkpoint OLIVIA in crossover. " << endl;
                gamma1 = ((100-loopCounter)/100)*(0.5 + 0.5*(Frandom01())); // if its hard to find a diplacement of the particle that fits
                gamma2 = ((100-loopCounter)/100)*(0.5 + 0.5*(Frandom01())); // then slowly drop the displacement to zero while resampling it
                thetaStarVector.push_back(thetaTvector[t] + gamma1 * (thetaMvector[t] - thetaNvector[t]) + (1-burninYN) * gamma2 * (thetaBvector[t] - thetaTvector[t]) + Frandom01()-0.5);
                if (LOLIMITS[0][t]==1 && HILIMITS[0][t]==1)
                {
                    cout << "Checkpoint PACIFICO in crossover. " << endl;
                    if (thetaStarVector[t] < LOLIMITS[1][t] || thetaStarVector[t] > HILIMITS[1][t])
                    {
                        starProblem = 1;
                    }
                }
                else if (LOLIMITS[0][t]==1)
                {
                    cout << "Checkpoint QUENTIN in crossover. " << endl;
                    if (thetaStarVector[t] < LOLIMITS[1][t])
                    {
                        cout << "Warning: thetaStarVector[" << t << "] too low !!!" << endl;
                        starProblem = 1;
                    }
                }
                else if (HILIMITS[0][t]==1)
                {
                    cout << "Checkpoint RACHEL in crossover. " << endl;
                    if (thetaStarVector[t] > HILIMITS[1][t])
                    {
                        starProblem = 1;
                    }
                }
                ::HCROSSOVER.parameter[group][particle][t][iteration] = thetaStarVector[t];
            }
            else
            {
                cout << "Checkpoint SAMUEL in crossover. " << endl;
                thetaStarVector.push_back(::HMCMCCHAINS.parameter[group][thetaTindex][t][iteration]);
                ::HCROSSOVER.parameter[group][particle][t][iteration] = thetaStarVector[t];
            }
            cout << "Checkpoint THOMAS in crossover. " << endl;
        }
        cout << "Checkpoint UFFE in crossover. " << endl;
    }
    cout << "Checkpoint VALKYRIE in crossover. " << endl;
    //for (int t = 0; t < ::NPARAMETERS; t++)
    //{
    //    ::HOLDINGPEN[group][particle][t] = thetaStarVector[t];
    //    ::HCROSSOVER.parameter[group][particle][tminusone][iteration] = thetaStarVector[t];
    //}
    // Not sure if the next line is needed will comment out...
    // FupdateMCMC(group,particle,iteration,3);
    cout << "Checkpoint WINSTON in crossover. " << endl;
}

void FdecideOnJump (int group, int particle, int iteration, int stage, int longversionYN)
{
    string runLogFile = "mhlog.txt";
    ofstream myLogFile;
    myLogFile.open(runLogFile.c_str(), std::ios_base::app);

    double logMHprob;
    double tempRand;
    double logTempRand;
    double tempLogQratio;
    double MHprob;
    
    tempLogQratio = computeTransProbQs (group, particle, iteration, stage) ;
    // CHECK TO MCMC PROBABILITY TO SEE IF TO KEEP IT
    //
    logMHprob = computeMHprob (group, particle, iteration, longversionYN, tempLogQratio, stage);
    MHprob = exp(logMHprob);
    myLogFile << "G: " << group << " P: " << particle << " I: " << iteration << " MH Probability is: " << MHprob;
    //if (logMHprob > 0)
    if (MHprob >= 1)
    { // keep it for sure
        myLogFile <<" and we are keeping it. " << endl;
        ::HMCMCCHAINS.newYN[group][particle][iteration] = 1;
        switch (stage)
        {
            case 2:
                ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMUTATION.pll[group][particle][iteration] ;
            case 3:
                ::HMCMCCHAINS.pll[group][particle][iteration] = ::HCROSSOVER.pll[group][particle][iteration] ;
        }
        for (int w = 0; w < ::NPARAMETERS; w++)
        {
            switch (stage)
            {
                case 2:
                    ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HMUTATION.parameter[group][particle][w][iteration];
                case 3:
                    ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HCROSSOVER.parameter[group][particle][w][iteration];
            }
        }
    }
    else
    {
        tempRand = Frandom01();
        myLogFile << " runif(0,1) is " << tempRand;
        //logTempRand = log(tempRand);
//        if (logTempRand < logMHprob) // this should be correct. line above is a wild guess
        if (tempRand < MHprob)
        { // keep it
            myLogFile << " so we keep it. " << endl;
            ::HMCMCCHAINS.newYN[group][particle][iteration] = 1;
            switch (stage)
            {
                case 2:
                    ::HMCMCCHAINS.pll[group][particle][iteration] = ::HMUTATION.pll[group][particle][iteration] ;
                case 3:
                    ::HMCMCCHAINS.pll[group][particle][iteration] = ::HCROSSOVER.pll[group][particle][iteration] ;
            }
            for (int w = 0; w < ::NPARAMETERS; w++)
            {
                switch (stage)
                {
                    case 2:
                        ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HMUTATION.parameter[group][particle][w][iteration];
                    case 3:
                        ::HMCMCCHAINS.parameter[group][particle][w][iteration] = ::HCROSSOVER.parameter[group][particle][w][iteration];
                }
            }
        }
        else
        {
            myLogFile << " so we don't keep it. " << endl;
        }
    }
    myLogFile.close();
}


/*
 #ifndef FWRITEMCMC_H
 #define FWRITEMCMC_H
 void fMakeMCMC (double MCMCCHAINS [NGROUPS][NPARTICLESPERGROUP][NMCMCPARAMETERS][NITERATIONS])
 {
 string filename;
 for (int g = 0; g < NGROUPS; g++)
 {
 for (int p = 0; p < NPARTICLESPERGROUP; p++)
 {
 filename = "mcmc_" + std::to_string(g) + "_" + std::to_string(p) + ".csv";
 ofstream output_file;
 output_file.open(filename.c_str());
 output_file << "newparticle, Mu, Sigma, Lambda, PLL \r";
 for (int i = 0; i < NITERATIONS; i++)
 {
 output_file << MCMCCHAINS [g][p][0][i] << ",";
 output_file << MCMCCHAINS [g][p][1][i] << ",";
 output_file << MCMCCHAINS [g][p][2][i] << ",";
 output_file << MCMCCHAINS [g][p][3][i] << ",";
 output_file << MCMCCHAINS [g][p][4][i] << ",";
 output_file << MCMCCHAINS [g][p][5][i] << ",";
 output_file << MCMCCHAINS [g][p][6][i] << ",";
 output_file << MCMCCHAINS [g][p][7][i] << "\r";
 }
 output_file.close();
 }
 }
 }
 #endif
 */
