 // Current problem around VALERIE
// We seem to have particle initialized as random
// In one run, particle = 1107296256
// It should be based on the value of triple[]
// Probably there is a missing computation

// This code is working at level 1
// It completes without errors thrown by c++
// However, some chains are apparently fully missing
// And the model does not recover parameters well at all
// Hypotheses:
//  1. When crossover generates an out of bounds particle,
//      current code moves it just across the bound,
//      but it should probably be discarded as unlikely
//  2. We aren't taking covariance into consideration in q-prob
//  3. It would cut processing time to sort X and Y before
//      computing pseudoLL and run only so long as needed to get
//      a PLL

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

const int NGROUPS = 10;
const int NPARTICLESPERGROUP = 10;
const int NITERATIONS = 10;
const int BURNIN      = NITERATIONS / 2; // at end of burn in crossover is simplified
const int NPARAMETERS = 3;
const int NPARAMETERSANDONE = NPARAMETERS + 1; // adds a slot for the PLL
const int NPARAMETERSLOTS = NPARAMETERS + 2;
const int NMCMCPARAMETERS = 5;
const int PLLSLOT = NPARAMETERS + 1;

// Function protocalls
double cdf_height(double mu, double sigma, double x);
double computeMHprob (int group, int particle, int iteration, int longVersionYN, double logQratio);

double computeTransProbQs (int group, int particle, int iteration);

vector<int> FchooseGroups2Migrate ();
void   Fcrossover (int group, int particle, int iteration, int burninYN);
void   FdefaultHoldingPen (int iteration);
double FexGaussian (double mu, double sigma, double lambda);
void   fFilling (int iteration);
double FholdingPenPriorCalculation (int group, int particle);
double FlogPriorProb (int group, int particle, int iteration);
double Fmad (vector<double> v);
void   fMakeMCMC (double MCMCCHAINS [::NGROUPS][::NPARTICLESPERGROUP][::NMCMCPARAMETERS][::NITERATIONS]);
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
void   FupdateMCMC (int group, int particle, int iteration);
void   FwriteParticlePLL (int group, int particle, int iteration);

double getKernelDensityftn (double Yvalue, vector<double> Xvector);
double getKernelHftn (vector<double> X);
double getKofUftn (double U);
double getIQRftn (vector<double> v);
double getSDftn (vector<double> xVector);

double pdf_height(double mu, double sigma, double x);

void shuffle(vector <double> &x_vector);
void shuffle_int(vector <int> &x_vector);
void shuffle_double(vector <int> &x_vector);

// Overall Control Paramters
const double PMIGRATE = 0.00;
const double PMUTATE = 0.00;
const double KAPPA = 1.0; // prob of a particular particle crossing over in a crossover step

// Define parameters to be recovered
const double MU     = 100.0;
const double SIGMA  = 15.0;
const double LAMBDA = 10.0;

const int NTRIALSPERSTUDY = 1000;
const int NTRIALSPERSIMULATION = 1000;

// Priors parameters
// prior is normal on all variables
const double PRIORMEANOFMU     = 100;
const double PRIORSDOFMU       = 100;
const double PRIORMEANOFSIGMA  = 15;
const double PRIORSDOFSIGMA    = 5;
const double PRIORMEANOFLAMBDA = 10;
const double PRIORSDOFLAMBDA   = 3;
const double PRIORMEANSVEC[4]  = {0, PRIORMEANOFMU, PRIORMEANOFSIGMA, PRIORMEANOFLAMBDA};
const double PRIORSDSVEC[4]    = {0, PRIORSDOFMU  , PRIORSDOFSIGMA,   PRIORSDOFLAMBDA  };

//const int EXTREMELOG     =  250;
const int NEGEXTREMELOG  = -250;
const double EXTREMEZ    =  22.18; // for 250 as extremelog
const double NEGEXTREMEZ = -22.18;

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
double MUTATESDMU;
double MUTATESDSIGMA;
double MUTATESDLAMBDA;
double SDMUTATEVECTOR[4] = {0,MUTATESDMU, MUTATESDSIGMA, MUTATESDLAMBDA};
double LOLIMITS[2][4] = {{1,1,1,1},{0,0,0,0}};
// first we have whether or not a lower limit then the vector of limits
// 999 means no limits but not important
double HILIMITS[2][4] = {{0,0,0,0},{999,999,999,999}};

int main() 
{
    // Main control loop
    //   Executes an exGaussian parameter recovery
    
    
    vector<double> yVector;
    double gaussian;
    double exponential;
    double exgaussian;
    
    // Generate data based on parameters to be recovered
    for (int i = 0; i < ::NTRIALSPERSTUDY; i++)
    {
        gaussian = Frandom_normal(::MU,::SIGMA);
        exponential = FrandomExponential(::LAMBDA) ; // change when matthew gets function done
        exgaussian = exponential + gaussian;
        yVector.push_back(exgaussian);
    }
    sort(yVector.begin(),yVector.end());
    for (int i = 0; i < ::NTRIALSPERSTUDY; i++)
    {
        ::YDATA[i] = yVector[i];
    }
    // Loop through the particles and groups choosing parameters based on priors
    for (int i61 = 0; i61 < ::NGROUPS; i61++)
    {
        for (int j61 = 0; j61 < ::NPARTICLESPERGROUP; j61++)
        {
            ::MCMCCHAINS[i61][j61][0][0] = 1;
            ::MCMCCHAINS[i61][j61][1][0] = FsampleMuFromPrior     ();
            ::MCMCCHAINS[i61][j61][2][0] = FsampleSigmaFromPrior  ();
            ::MCMCCHAINS[i61][j61][3][0] = FsampleLambdaFromPrior ();
        }
    }
    
    fFilling(0); // fills X with data at iteration
    FPLL(0); // puts PLL in MCMC at iteration
    
    int iterationMinusOne;
    cout << "Entering loop through iterations. " << endl;
    for (int i88 = 1; i88 < ::NITERATIONS; i88++)
    {
        cout << "Beginning iteration: " << i88 << endl;
        iterationMinusOne = i88 - 1;
        // initialize the MCMC chains values
        cout << "MCMC. Group: ";
        for (int g88 = 0; g88 < ::NGROUPS; g88++)
        {
            cout << g88;
            for (int p88 = 0; p88 < ::NPARTICLESPERGROUP; p88++)
            {
                cout << "\n Particle: " << p88 << endl;
                ::MCMCCHAINS[g88][p88][0][i88] = 0;
                for (int x88 = 1; x88 < ::NPARAMETERSLOTS; x88++)
                {
                    ::MCMCCHAINS[g88][p88][x88][i88] = ::MCMCCHAINS[g88][p88][x88][iterationMinusOne];
                    cout << "\n Parameter " << x88 << "=" << ::MCMCCHAINS[g88][p88][x88][iterationMinusOne] << endl;
                }
            }
        }
        cout << "MCMC chains have now been initialized." << endl;
        // Initialize the holding pen values to MCMC chains at previous iteration
        FdefaultHoldingPen(i88);
        
        // Loop across groups
        // For each group decide whether or not to MIGRATE
        
        for (int g109 = 0; g109 < ::NGROUPS; g109++)
        {
            double prob01 = Frandom01(); // NOTE THAT THE RANDOM FUNCTION NEEDS TO BE INCLUDED
            // ALSO NOTE THAT THE RANDOM FUNCTION REQUIRES CHRON PACKAGES
            if (prob01 < ::PMIGRATE)
            {
                cout << "A decision was made to migrate. Beginning migration." << endl;
                // EXECUTE THE MIGRATION STEP
                vector<int> groups2migrate;
                groups2migrate = FchooseGroups2Migrate();
                int number2migrate;
                number2migrate = groups2migrate.size();
                vector<double> groupAndParticleVector;
                groupAndParticleVector.clear();
                //double groupAndParticleVector[number2migrate];
                for (int g163 = 0; g163 < number2migrate; g163++)
                {
                    vector<double> fitnessvector;
                    int chosenParticle;
                    for (int p163 = 0; p163 < ::NPARTICLESPERGROUP; p163++)
                    {
                        fitnessvector.push_back(::MCMCCHAINS[g163][p163][4][i88]);
                    }
                    chosenParticle = FselectParticleXFitness(fitnessvector,1,1);
                    double tempvar198;
                    tempvar198 = g163 + (chosenParticle/1000.0);
                    groupAndParticleVector.push_back(tempvar198);
                    fitnessvector.clear();
                }
                vector<double> postSwapVector;
                postSwapVector = Fswaparoo(groupAndParticleVector);
                for (int g177 = 0; g177 < number2migrate; g177++)
                {
                    int group;
                    int particle;
                    int hpgroup;
                    int hpparticle;
                    int mcmcgroup;
                    int mcmcparticle;
                    for (int e177 = 0; e177 < ::NPARAMETERSLOTS; e177++)
                    {
                        group    = trunc(groupAndParticleVector[g177]);
                        particle = 1000 * (groupAndParticleVector[g177] - group);
                        ::HOLDINGPEN[group][particle][e177] = ::MCMCCHAINS[group][particle][e177][i88];
                    }
                    for (int e187 = 1; e187 < ::NPARAMETERSLOTS; e187++)
                    {
                        hpgroup = group;
                        hpparticle = particle;
                        mcmcgroup = trunc(postSwapVector[g177]);
                        mcmcparticle = 1000 * (postSwapVector[g177] - mcmcgroup);
                        ::MCMCCHAINS[mcmcgroup][mcmcparticle][e187][i88] = ::HOLDINGPEN[hpgroup][hpparticle][e187];
                    }
                    ::MCMCCHAINS[mcmcgroup][mcmcparticle][0][i88] = 1; // says this is a new set of values
                }
            }
            cout << "Migration step has ended." << endl;
        }
        int thetaTindex; // this is a repeated declaration
        for (int g122 = 0; g122 < ::NGROUPS; g122++)
        {
            for (int p122 = 0; p122 < ::NPARTICLESPERGROUP; p122++)
            {
                double tempLogQratio;
                // DECIDE WHETHER TO MUTATE
                double prob02 = Frandom01();
                if (prob02 < ::PMUTATE)
                {
                    cout << "A decision has been made to mutate." << endl;
                    cout << "checkpoint andy" << endl;
                    Fmutation (g122, p122, i88);
                    cout << "checkpoint betty" << endl;
                    //double tempLogQratio;
                    cout << "checkpoint charlie" << endl;
                    tempLogQratio = computeTransProbQs (g122, p122, i88) ;
                    cout << "checkpoint dave";
                    // CHECK TO MCMC PROBABILITY TO SEE IF TO KEEP IT
                    //
                    double logMHprob;
                    cout << "checkpoint eddie";
                    logMHprob = computeMHprob (g122, p122, i88, 1, tempLogQratio);
                    cout << "checkpoint fred";
                    if (logMHprob > 0)
                    { // keep it for sure
                        cout << "checkpoint gracie";
                        ::MCMCCHAINS[g122][p122][0][i88] = 1.0;
                        cout << "checkpoint harold";
                        for (int w = 1; w <= ::NPARAMETERSANDONE; w++)
                        {
                            ::MCMCCHAINS[g122][p122][w][i88] = ::HOLDINGPEN[g122][p122][w];
                            cout << "checkpoint isaac " << w << endl;
                        }
                    }
                    else
                    {
                        double tempRand;
                        cout << "checkpoint jamie";
                        tempRand = Frandom01();
                        cout << "checkpoint karl";
                        double logTempRand;
                        cout << "checkpoint lewis";
                        logTempRand = log(tempRand);
                        cout << "checkpoint mark";
                        if (logTempRand < logMHprob)
                        { // keep it
                            ::MCMCCHAINS[g122][p122][0][i88] = 1.0;
                            cout << "checkpoint nancy";
                            for (int w = 1; w <= ::NPARAMETERSANDONE; w++)
                            {
                                
                                ::MCMCCHAINS[g122][p122][w][i88] = ::HOLDINGPEN[g122][p122][w];
                                cout << "checkpoint owen " << w << endl;
                            }
                        }
                    }
                    cout << "The mutation step has ended." << endl;
                }
                // IF NOT MUTATE THEN CROSSOVER
                else
                {
                    cout << "A decision has been made to crossover." << endl;
                    // g122 is group
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
                    int triple[3];
                    int baseparticle;
                    int burninYN;
                    if (i88 < ::BURNIN)
                    {
                        Fcrossover (g122,p122,i88,1);
                        FwriteParticlePLL(g122,p122,i88);
                        FupdateMCMC(g122,p122,i88);
//                        cout << "Checkpoint ALEX, g122="<< g122 <<", i88=" << i88 << endl;
//                        // baseparticle is the one that pushes back into good parameter space
//                        // Theta B in Turner and Sederburg
//                        vector<double> fitnessvector2;
//                        for (int p260 = 0; p260 < ::NPARTICLESPERGROUP; p260++)
//                        {
//                            // this loop creates a list of the numbers from 0 to NPARTICLESPERGROUP-1
//                            cout << "Checkpoint BARRY, p260=" << p260 << endl;
//                            fitnessvector2.push_back(MCMCCHAINS[g122][p260][4][i88]);
//                        }
//                        baseparticle = FselectParticleXFitness(fitnessvector2,0,1);
//                        cout << "base particle (Theta b) is " << baseparticle << endl;
//                        fitnessvector2.clear();
//                        //int triple [3];
//                        vector<int> tempindexvector;
//                        for (int i = 0; i < ::NPARTICLESPERGROUP; i++)
//                        {
//                            // this loop creates a list of the numbers from 0 to NPARTICLESPERGROUP-1
//                            cout << "Checkpoint CAROL, i=" << i << endl;
//                            tempindexvector.push_back(i);
//                        }
//                        // shuffling the list of numbers will help us sample from them
//                        shuffle_int(tempindexvector);
//                        for (int s = 0; s < ::NPARTICLESPERGROUP; s++)
//                        {
//                            // this is just a cout loop for debugging
//                            cout << "tempindexvector is: " << tempindexvector[s] << endl;
//                        }
//                        int donecheck = 0;
//                        // donecheck tells us whether we have all the particles we need (3)
//                        int runner = 0;
//                        // runner tells us whether we have more particles to consider
//                        cout << "Checkpoint DAVID, i88=" << i88 << endl;
//                        while (donecheck < 3)
//                        {
//                            // the purpose of this while loop is to select three distinct particles
//                            // none of which is the baseparticle
//                            // selecting each with equal probability
//                            // the vector triple will contain the index values of the three
//                            if (tempindexvector[runner] != baseparticle)
//                            // we cannot use the baseparticle for either of the others
//                            {
//                                triple[donecheck] = tempindexvector[runner];
//                                cout << "Checkpoint EDDIE, donecheck=" << donecheck << endl;
//                                cout << "baseparticle: " << baseparticle << endl;
//                                cout << "runner: " << runner << endl;
//                                cout << "donecheck: " << donecheck << endl;
//                                cout << "triple[donecheck]: " << triple[donecheck] << endl;
//                                cout << "tempindexvector[runner]: " << tempindexvector[runner] << endl;
//                                donecheck++;
//                            }
//                            runner++;
//                        }
//                        cout << "triple[0]: " << triple[0] << endl;
//                        cout << "triple[1]: " << triple[1] << endl;
//                        cout << "triple[2]: " << triple[2] << endl;
//                        tempindexvector.clear();
//                        cout << "Checkpoint FRANK, i88=" << i88 << endl;
//                        cout << "::NPARAMETERS: " << ::NPARAMETERS << endl;
//                        for (int t = 1; t <= ::NPARAMETERS; t++)
//                        {
//                            // loop across parameters doing crossover
//                            double temprand284;
//                            temprand284 = Frandom01();
//                            cout << "Checkpoint GARY, t=" << t << endl;
//                            // but only crossover with probability KAPPA (e.g., .90)
//                            //if (temprand284 < ::KAPPA)
//                            //{
//                            cout << "Checkpoint HANK: temprand284 < KAPPA." << endl;
//                            double tempb;
//                            tempb = Frandom01()/1000; // small random noise
//                            // this should probably depend on the parameter being displaced
//                            double thetaStar; // proposed parameter value
//                            double thetaB; // value of base particle
//                            thetaB = MCMCCHAINS[g122][baseparticle][t][i88];
//                            cout << "Checkpoint ISAAC" << endl;
//                            // int thetaTindex; // this is a duplicated declaration
//                            thetaTindex = triple[0]; // index of target particle to be crossed over
//                            double thetaT;
//                            thetaT = MCMCCHAINS[g122][thetaTindex][t][i88]; // value of target parameter
//                            cout << "thetaTindex: " << thetaTindex << endl;
//                            cout << "Checkpoint JONES" << endl;
//                            int thetaMindex;
//                            thetaMindex = triple[1];
//                            double thetaM;
//                            cout << "g122 (group): " << g122 << endl;
//                            cout << "thetaMindex (particle): " << thetaMindex << endl;
//                            cout << "t (parameter): " << t << endl;
//                            cout << "i88 (iteration): " << i88 << endl;
//                            thetaM = MCMCCHAINS[g122][thetaMindex][t][i88];
//                            cout << "Checkpoint KINGSTON" << endl;
//                            int thetaNindex;
//                            thetaNindex = triple[2];
//                            double thetaN;
//                            cout << "g122 (group): " << g122 << endl;
//                            cout << "thetaNindex (particle): " << thetaNindex << endl;
//                            cout << "t (parameter): " << t << endl;
//                            cout << "i88 (iteration): " << i88 << endl;
//                            thetaN = MCMCCHAINS[g122][thetaNindex][t][i88];
//                            cout << "Checkpoint LORENZO" << endl;
//                            double gamma1;
//                            double gamma2;
//                            gamma1 = 0.5 + 0.5*(Frandom01());
//                            gamma2 = 0.5 + 0.5*(Frandom01());
//                            cout << "gamma1 is: " << gamma1 << endl;
//                            cout << "gamma2 is: " << gamma2 << endl;
//                            cout << "Checkpoint MELANIA" << endl;
//                            thetaStar = thetaT + gamma1 * (thetaM - thetaN) + gamma2 * (thetaB - thetaT) + tempb;
//                            cout << "Checkpoint MELVIN" << endl;
//                            // make sure thetaStar has not gone outside of its support
//                            // start with the knowledge that it is only bounded from below
//                            cout << "t = " << t << endl;
//                            cout << "thetaStar = " << thetaStar << endl;
//                            cout << "LOLIMITS[1][3] = " << LOLIMITS[1][3] << endl;
//                            while (thetaStar < LOLIMITS[1][t] || thetaStar > HILIMITS[1][t])
//                            {
//                                gamma1 = Frandom01();
//                                gamma2 = Frandom01();
//                                thetaStar = thetaT + gamma1 * (thetaM - thetaN) + gamma2 * (thetaB - thetaT) + tempb;
//                            }
//                            ::HOLDINGPEN[g122][thetaTindex][t] = thetaStar;
//                            if (temprand284 > ::KAPPA)
//                            {
//                                ::HOLDINGPEN[g122][thetaTindex][t] = ::MCMCCHAINS[g122][thetaTindex][t][i88];
//                            }
//                        } // dont forget we need new PLL
//                        FupdateMCMC(g122,p122,i88);
                    }
                    else // what to do if not in burnin
                    {
                        Fcrossover (g122,p122,i88,0);
                        FwriteParticlePLL(g122,p122,i88);
                        FupdateMCMC(g122,p122,i88);
//                        cout << "Checkpoint NANCY. " << endl;
//                        vector<int> tempindexvector;
//                        for (int i = 0; i < ::NPARTICLESPERGROUP; i++)
//                        {
//                            tempindexvector.push_back(i);
//                        }
//                        cout << "Checkpoint OVERLORD. " << endl;
//                        shuffle_int(tempindexvector);
//                        triple[0] = tempindexvector[0];
//                        cout << "triple[0] = " << triple[0] << endl;
//                        triple[1] = tempindexvector[1];
//                        cout << "triple[1] = " << triple[1] << endl;
//                        triple[2] = tempindexvector[2];
//                        cout << "triple[2] = " << triple[2] << endl;
//                        tempindexvector.clear();
//                        cout << "Checkpoint PUSSYCAT. " << endl;
//                        for (int t = 1; t <= ::NPARAMETERS; t++)
//                        {
//                            double temprand284;
//                            temprand284 = Frandom01();
//                            //if (temprand284 < ::KAPPA)
//                            //{
//                            cout << "Checkpoint QUIET. random variable < Kappa, not in burnin. " << endl;
//                            double tempb;
//                            tempb = Frandom01()/1000;
//                            double thetaStar;
//                            cout << "triple[0]: " << triple[0] << endl;
//                            thetaTindex = triple[0];
//                            double thetaT;
//                            thetaT = ::MCMCCHAINS[g122][thetaTindex][t][i88];
//                            cout << "Checkpoint RANDOLF. " << endl;
//                            int thetaMindex;
//                            thetaMindex = triple[1];
//                            double thetaM;
//                            thetaM = ::MCMCCHAINS[g122][thetaMindex][t][i88];
//                            cout << "Checkpoint SAMMY. " << endl;
//                            int thetaNindex;
//                            thetaNindex = triple[2];
//                            double thetaN;
//                            thetaN = ::MCMCCHAINS[g122][thetaNindex][t][i88];
//                            cout << "Checkpoint THOMAS. " << endl;
//                            double gamma1;
//                            gamma1 = 0.5 + 0.5*(Frandom01());
//                            thetaStar = thetaT + gamma1 * (thetaM - thetaN) + tempb;
//                            if (thetaStar < LOLIMITS[1][t] || thetaStar > HILIMITS[1][t] || temprand284 > ::KAPPA)
//                            {
//                                ::HOLDINGPEN[g122][p122][t] = ::MCMCCHAINS[g122][p122][t][i88];
//                            }
//                            else
//                            {
//                                ::HOLDINGPEN[g122][p122][t] = thetaStar;
//                            }
//                            // make sure thetaStar has not gone outside of its support
//                            // start with the knowledge that it is only bounded from below
//                            //while (thetaStar < LOLIMITS[1][t282])
//                            //{
//                            // tempb = Frandom01()/1000;
//                            // thetaStar = thetaT + gamma1 * (thetaM - thetaN) + tempb;
//                            // cout << "Checkpoint UVALDE. " << endl;
//                            //    thetaStar = LOLIMITS[1][t282] + 1/1000;
//                            //}
//                            //} end kappa decision loop
//                        }
//                        FupdateMCMC(g122,p122,i88);
                    }
//--not in burnin   }
                    // new PLL needs to be calculated based on new values
                    cout << "Checkpoint VALERIE. " << endl;
                    cout << "g122 (group): " << g122 << endl;
// why                   cout << "thetaTindex (particle): " << thetaTindex << endl;
                    cout << "i88 (iteration): " << i88 << endl;
// why                   FwriteParticlePLL(g122,thetaTindex,i88);
                    cout << "Checkpoint WILLIAM. " << endl;
// why                   ::MCMCCHAINS[g122][thetaTindex][0][i88]=1; // update new particle indicator
                }
            }
        }
        
    }
    // WRITE MCMC FILES WITH RESULTS
    cout << "Checkpoint XAVIER. " << endl;
    fMakeMCMC(::MCMCCHAINS);
}

void FdefaultHoldingPen (int iteration)
{
    for (int g = 0; g < NGROUPS; g++)
    {
        for (int p = 0; p < NPARTICLESPERGROUP; p++)
        {
            for (int t = 0; t < NPARAMETERSLOTS; t++)
            {
                ::HOLDINGPEN[g][p][t] = ::MCMCCHAINS[g][p][t][iteration-0];
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
                gaussian = Frandom_normal(::MCMCCHAINS[i][j][1][iteration],::MCMCCHAINS[i][j][2][iteration]);
                while (gaussian <= 0.0)
                {
                    gaussian = Frandom_normal(::MCMCCHAINS[i][j][1][iteration],::MCMCCHAINS[i][j][2][iteration]);
                }
                exponential = FrandomExponential(::MCMCCHAINS[i][j][3][iteration]);
                ::XDATA [i][j][k] = gaussian + exponential;
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
    double exponential = FrandomExponential (lambda);
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
            ::MCMCCHAINS[i][j][::PLLSLOT][iterationi] = particlePLL;
        }
    }
}

#ifndef FPLLPARTICLE_H
#define FPLLPARTICLE_H
double FPLLparticle (vector<double> xdata)
{
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
void FwriteParticlePLL (int group, int particle, int iteration)
{
    cout << "Checkpoint: entering FwriteParticlePLL. " << endl;
    cout << "Group: "     << group     << endl;
    cout << "Particle: "  << particle  << endl;
    cout << "Iteration: " << iteration << endl;
    for (int i450 = 0; i450 < NTRIALSPERSIMULATION; i450++)
    {
        double    tempvar450;
        double    mu450;
        double    sigma450;
        double    lambda450;
        mu450     = ::MCMCCHAINS[group][particle][1][iteration];
        sigma450  = ::MCMCCHAINS[group][particle][2][iteration];
        lambda450 = ::MCMCCHAINS[group][particle][3][iteration];
        tempvar450 = FexGaussian (mu450, sigma450, lambda450);
        ::XPARTICLE[i450] = tempvar450;
    }
    double pllOfParticle;
    vector<double> Txparticle;
    int lengthTxparticle;
    lengthTxparticle = sizeof(::XPARTICLE)/sizeof(::XPARTICLE[0]);
    for (int j450 = 0; j450 < lengthTxparticle; j450++)
    {
        Txparticle.push_back(::XPARTICLE[j450]);
    }
    pllOfParticle = FPLLparticle(Txparticle);
//    ::MCMCCHAINS[group][particle][4][iteration]=pllOfParticle;
    ::HOLDINGPEN[group][particle][4] = pllOfParticle;
}
#endif

void Fmutation (int groupNumber, int particleNumber, int iterationID)
{
    for (int q205 = 1; q205 <= ::NPARAMETERS; q205++)
    {
        cout << "Checkpoint Fmutation, group=" << groupNumber << ", particle=" << particleNumber << ", iteration=" << iterationID << endl;
        double tempvar207 = ::MCMCCHAINS[groupNumber][particleNumber][q205][iterationID] + Frandom_normal(0,.001);
        if (::LOLIMITS[0][q205] == 0)
        {
            if (::HILIMITS[0][q205] == 0)
            {
                ::HOLDINGPEN[groupNumber][particleNumber][q205] = tempvar207;
                cout << "Checkpoint Alonzo in Fmutation reached. " << endl;
            }
            else // upper limit but no lower one
            {
                while (tempvar207 > ::HILIMITS[1][q205])
                {
                    tempvar207 = ::MCMCCHAINS[groupNumber][particleNumber][q205][iterationID] + Frandom_normal(0,.001);
                }
                ::HOLDINGPEN[groupNumber][particleNumber][q205] = tempvar207;
                cout << "Checkpoint Bernardo in Fmutation reached. " << endl;
            }
        }
        else // there is a lower limit
        {
            while (tempvar207 < ::LOLIMITS[1][q205])
            {
                tempvar207 = ::MCMCCHAINS[groupNumber][particleNumber][q205][iterationID] + Frandom_normal(0,.001);
                cout << "Checkpoint Carlo in Fmutation reached. " << endl;
            }
            if (::HILIMITS[0][q205] == 1)
            {
                while (tempvar207 > ::HILIMITS[1][q205])
                {
                    tempvar207 = ::MCMCCHAINS[groupNumber][particleNumber][q205][iterationID] + Frandom_normal(0,.001);
                }
            }
            ::HOLDINGPEN[groupNumber][particleNumber][q205] = tempvar207;
            cout << "Checkpoint Daniella in Fmutation reached. " << endl;
        }
    }
    for (int i230 = 0; i230 < NTRIALSPERSIMULATION; i230++)
    {
        double    tempvar230;
        double    mu230;
        double    sigma230;
        double    lambda230;
        cout << "Checkpoint Eduardo in Fmutation reached. " << endl;
        mu230     = ::HOLDINGPEN[groupNumber][particleNumber][1];
        sigma230  = ::HOLDINGPEN[groupNumber][particleNumber][2];
        lambda230 = ::HOLDINGPEN[groupNumber][particleNumber][3];
        tempvar230 = FexGaussian (mu230, sigma230, lambda230);
        cout << "Checkpoint Fernando in Fmutation reached. " << endl;
        ::XPARTICLE[i230] = tempvar230;
    }
    double pllOfParticle;
    vector<double> Txparticle2;
    int lengthTxparticle2 = sizeof(::XPARTICLE)/sizeof(::XPARTICLE[0]);
    for (int i548 = 0; i548 < lengthTxparticle2; i548++)
    {
        Txparticle2.push_back(::XPARTICLE[i548]);
        cout << "Checkpoint Gandolph in Fmutation reached. " << endl;
    }
    pllOfParticle = FPLLparticle(Txparticle2);
    ::HOLDINGPEN[groupNumber][particleNumber][4]=pllOfParticle;
    // decide whether to make the jump
    
    if (::HOLDINGPEN[groupNumber][particleNumber][4] > ::MCMCCHAINS[groupNumber][particleNumber][4][iterationID])
    {
        for (int i240 = 0; i240 < ::NPARAMETERSLOTS; i240++)
        {
            ::MCMCCHAINS[groupNumber][particleNumber][i240][iterationID] = ::HOLDINGPEN[groupNumber][particleNumber][i240];
            cout << "Checkpoint Harold in Fmutation reached. " << endl;
        }
        ::MCMCCHAINS[groupNumber][particleNumber][0][iterationID] = 1;
        cout << "Checkpoint Ismereldo in Fmutation reached. " << endl;
    }
    else
    { // here we jump with an MH probability
        double testprob;
        testprob = Frandom01();
        double Ltestprob;
        Ltestprob = log(testprob);
        double LogDifference = ::HOLDINGPEN[groupNumber][particleNumber][4] - ::MCMCCHAINS[groupNumber][particleNumber][4][iterationID];
        cout << "Checkpoint Jannine in Fmutation reached. " << endl;
        // SEGFAULT AFTER JANNINE
        cout << "MCMCCHAINS 0 - 0 - 0 - 1 " <<  ::MCMCCHAINS[0][0][0][1] << endl;
        cout << "Holding pen " << ::HOLDINGPEN[0][0][0] << endl;
        cout << "Ltestprob " << Ltestprob << endl;
        cout << "LogDifference " << LogDifference << endl;
        if (Ltestprob < LogDifference)
        {
            cout << "Checkpoint JUMP " << endl;
            for (int i = 0; i < ::NPARAMETERSLOTS; i++)
            {
                ::MCMCCHAINS[groupNumber][particleNumber][i][iterationID] = ::HOLDINGPEN[groupNumber][particleNumber][i];
                cout << "Checkpoint Klondike in Fmutation reached. " << endl;
            }
            cout << "Checkpoint Kali reached. " << endl;
            ::MCMCCHAINS[groupNumber][particleNumber][0][iterationID] = 1.0;
            cout << "Checkpoint Lyle in Fmutation reached. " << endl;
        }
    }
}


double FlogPriorProb (int group, int particle, int iteration)
{
    double logmupriorprob     ;
    double logsigmapriorprob  ;
    double loglambdapriorprob ;
    double mupriorprob     = pdf_height (::PRIORMEANOFMU, ::PRIORSDOFMU ,::MCMCCHAINS[group][particle][1][iteration]);
    if (mupriorprob == 0)
    {logmupriorprob = ::NEGEXTREMELOG;}
    else
    {
        logmupriorprob = log(mupriorprob);
        if (logmupriorprob < ::NEGEXTREMELOG)
        {logmupriorprob = ::NEGEXTREMELOG;}
    }
    double sigmapriorprob  = pdf_height (::PRIORMEANOFSIGMA, ::PRIORSDOFSIGMA , ::MCMCCHAINS[group][particle][2][iteration]);
    if (sigmapriorprob == 0)
    {logsigmapriorprob = ::NEGEXTREMELOG;}
    else
    {
        logsigmapriorprob = log(sigmapriorprob);
        if (logsigmapriorprob < ::NEGEXTREMELOG)
        {logsigmapriorprob = ::NEGEXTREMELOG;}
    }
    double lambdapriorprob = pdf_height (::PRIORMEANOFLAMBDA,::PRIORSDOFLAMBDA,::MCMCCHAINS[group][particle][3][iteration]);
    if (lambdapriorprob == 0)
    {loglambdapriorprob = ::NEGEXTREMELOG;}
    else
    {
        loglambdapriorprob = log(lambdapriorprob);
        if (loglambdapriorprob < ::NEGEXTREMELOG)
        {loglambdapriorprob = ::NEGEXTREMELOG;}
    }
    double logprior;
    logprior = logmupriorprob + logsigmapriorprob + loglambdapriorprob;
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

double computeTransProbQs (int group, int particle, int iteration)
{
    //return 0;
    vector<double> zOldMutate;
    vector<double> zNewMutate;
    double numerSum=0;
    double denomSum=0;
    
    // Changed to int i = 1, to int i = 0 in the for loop
    for (int i = 1; i <= ::NPARAMETERS; i++)
    {
        cout << "Test 1" << endl;
        int lohiBoundyn = ::LOLIMITS[0][i]*10 + HILIMITS[0][i];
        cout << "Test 2" << endl;
        double oldMutateElement;
        oldMutateElement = ::MCMCCHAINS[group][particle][i][iteration];
        double newMutateElement;
        newMutateElement = ::HOLDINGPEN[group][particle][i];
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
                    numerSum = numerSum + log(cdf_height (oldMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]));
                case 10: // bounded only from below
                    numerSum = numerSum + log(cdf_height (::LOLIMITS[1][i], ::SDMUTATEVECTOR[i],oldMutateElement));
                case 11: // bounded on both sides
                    numerSum = numerSum + log (cdf_height(oldMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]) - cdf_height(oldMutateElement,::SDMUTATEVECTOR[i],::LOLIMITS[1][i]));
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
                    denomSum = denomSum + log(cdf_height (newMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]));
                case 10: // bounded only from below
                    denomSum = denomSum + log(cdf_height (::LOLIMITS[1][i],::SDMUTATEVECTOR[i],newMutateElement));
                case 11: // bounded on both sides
                    denomSum = denomSum + log (cdf_height(newMutateElement,::SDMUTATEVECTOR[i],::HILIMITS[1][i]) - cdf_height(newMutateElement,::SDMUTATEVECTOR[i],::LOLIMITS[1][i]));
            }
        }
    }
    return numerSum - denomSum; // returns the log of (q(thetaT|thetaStar)/q(thetaStar|thetaT))
}




#ifndef FWRITEMCMC_H
#define FWRITEMCMC_H
// Globally declared constants
void fMakeMCMC (double MCMCCHAINS[::NGROUPS][::NPARTICLESPERGROUP][::NMCMCPARAMETERS][NITERATIONS])
{
    string filename;
    for (int g = 0; g < ::NGROUPS; g++)
    {
        for (int p = 0; p < ::NPARTICLESPERGROUP; p++)
        {
            filename = "mcmc_" + std::to_string(g) + "_" + std::to_string(p) + ".csv";
            ofstream output_file;
            output_file.open(filename.c_str());
            output_file << "newParameters, Mu, Sigma, Lambda, PLL \r";
            for (int i = 0; i < ::NITERATIONS; i++)
            {
                output_file << ::MCMCCHAINS [g][p][0][i] << ",";
                output_file << ::MCMCCHAINS [g][p][1][i] << ",";
                output_file << ::MCMCCHAINS [g][p][2][i] << ",";
                output_file << ::MCMCCHAINS [g][p][3][i] << ",";
                output_file << ::MCMCCHAINS [g][p][4][i] << "\r";
            }
            output_file.close();
        }
    }
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
double computeMHprob (int group, int particle, int iteration, int longVersionYN, double logQratio)
{
    
    ::HOLDINGPENPRIORS[group][particle]      = FholdingPenPriorCalculation(group, particle);
    ::MCMCPRIORS[group][particle][iteration] = FMCMCpriorCalculation(group, particle, iteration);
    
    double logMHprob;
    double logPriorRatio;
    logPriorRatio = ::HOLDINGPENPRIORS[group][particle] - ::MCMCPRIORS[group][particle][iteration];
    double logPLLratio;
    logPLLratio = ::HOLDINGPEN[group][particle][4] - MCMCCHAINS[group][particle][4][iteration];
    
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
double FholdingPenPriorCalculation (int group, int particle)
{
    double logtempprob;
    double sumlogtempprob = 0.0;
    for (int i500 = 1; i500 <= 3; i500++)
    {
        double tempprob = 0.0;
        double supportArea = 1.0;
        tempprob = pdf_height (::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HOLDINGPEN[group][particle][i500]);
        if (LOLIMITS[0][i500] == 1)
        {
            supportArea = supportArea - cdf_height(::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::LOLIMITS[1][i500]);
        }
        if (HILIMITS[0][i500] == 1)
        {
            supportArea = supportArea - (1.0 - cdf_height(::PRIORMEANSVEC[i500],::PRIORSDSVEC[i500],::HILIMITS[1][i500]));
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
    for (int i501 = 1; i501 <= 3; i501++)
    {
        double tempprob = 0.0;
        double logtempprob;
        double supportArea = 1.0;
        tempprob = pdf_height (::PRIORMEANSVEC[i501],::PRIORSDSVEC[i501],::MCMCCHAINS[group][particle][i501][iteration]);
        if (LOLIMITS[0][i501] == 1)
        {
            supportArea = supportArea - cdf_height(::PRIORMEANSVEC[i501],::PRIORSDSVEC[i501],::LOLIMITS[1][i501]);
        }
        if (HILIMITS[0][i501] == 1)
        {
            supportArea = supportArea - (1.0 - cdf_height(::PRIORMEANSVEC[i501],::PRIORSDSVEC[i501],::HILIMITS[1][i501]));
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
    vector<double> groupIDvector;
    int numberOfGroups2Migrate = FrandDiscreteUniform (3, ::NGROUPS);
    for (int i640 = 0; i640 < ::NGROUPS; i640++)
    {
        baseGroupVector.push_back(i640);
    }
    shuffle(baseGroupVector);
    for (int i656 = 0; i656 < numberOfGroups2Migrate; i656++)
    {
        groupIDvector.push_back(baseGroupVector[i656]);
    }
    vector<int> INTgroupIDvector;
    for (int i = 0; i < numberOfGroups2Migrate; i++)
    {
        INTgroupIDvector.push_back(groupIDvector[i]);
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
            if (FitnessVector[i] > -700)
            { // covers the case where there is a non-inf value
                uFitnessVector[i] = exp(FitnessVector[i]);
            }
            else
            { // covers the inf value of -700
                
                uFitnessVector[i] = 0;
                
                // in this context 0 is fine
            } // now FitnessVector has been converted to uFitnessVector
            // in the log=TRUE case
        }
    }
    else
    {
        for (int i3204=0; i3204 < FitnessVector.size(); ++i3204)
        {
            uFitnessVector[i3204] = FitnessVector[i3204];
        }
    }
    // choose on inverse scale if needed
    if (inverseYN==1)
    {
        for (int i534 = 0; i534 < uFitnessVector.size(); ++i534)
        {
            uFitnessVector[i534] = 1 / (uFitnessVector[i534]+exp(-500));
        }
    }
    // compute cumulative sum
    for (int i5640 = 0; i5640 < uFitnessVector.size(); ++i5640)
    {
        uFitnessVectorSum = uFitnessVectorSum + uFitnessVector[i5640];
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
    int exitStatusTrue;
    exitStatusTrue=0;
    int i4908;
    i4908 = -1;
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
                output_file << ::MCMCCHAINS [g][p][0][i] << ",";
                output_file << ::MCMCCHAINS [g][p][1][i] << ",";
                output_file << ::MCMCCHAINS [g][p][2][i] << ",";
                output_file << ::MCMCCHAINS [g][p][3][i] << ",";
                output_file << ::MCMCCHAINS [g][p][4][i] << "\r";
            }
            output_file.close();
        }
    }
}

#endif

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
  double adsum = 0;
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
    //double sd;
    //double iqr;
    double mad;
    mad = Fmad(X);
    //sd  = getSDftn(X);
    //cout << "The sd is: " << sd << endl;
    //iqr = getIQRftn(X);
    //cout << "The iqr is: " << iqr << endl;
    //double minspread;
    //cout << "The minspread is: " << iqr << endl;
    //minspread = min(sd,iqr);
    double n;
    n = X.size();
    double fifthRootOfn;
    fifthRootOfn = pow(n,0.2);
    //cout << "The fifth root of n is: " << fifthRootOfn << endl;
    double h;
    //h = 0.9*minspread/fifthRootOfn;
    h = 0.9 * mad / fifthRootOfn;
    if (h == 0.0)
    {
        h = 1.0;
        cout << "Warning: h was calcuated as zero !" << endl;
        cout << "Size of vector is: " << n << endl;
        cout << "MAD is: " << mad << endl;
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
        KofU = 3.0*(1-U*U)/4.0;
    }
    else
    {
        KofU = 0.0;
    }
    //cout << "K(u) is: " << KofU << endl;
    return KofU;
}

void FupdateMCMC (int group, int particle, int iteration)
{
    int numequivalents=0; // number of parameters equal in mcmcchains and holdingpen
    for (int w = 1; w <= NPARAMETERS; w++)
    {
        if (::HOLDINGPEN[group][particle][w] == ::MCMCCHAINS[group][particle][w][iteration])
        {
            numequivalents++;
        }
    }
    if (numequivalents == NPARAMETERS)
    {
        cout << "Exiting FupdateMCMC with return 0 - this may not work " << endl;
        return;
    }
    double tempLogQratio;
    tempLogQratio = computeTransProbQs (group, particle, iteration) ;
    // CHECK TO MCMC PROBABILITY TO SEE IF TO KEEP IT
    //
    double logMHprob;
    logMHprob = computeMHprob (group, particle, iteration, 1, tempLogQratio);
    if (logMHprob > 0)
    { // keep it for sure
        ::MCMCCHAINS[group][particle][0][iteration] = 1.0;
        for (int w = 1; w <= ::NPARAMETERSANDONE; w++)
        {
            ::MCMCCHAINS[group][particle][w][iteration] = ::HOLDINGPEN[group][particle][w];
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
            ::MCMCCHAINS[group][particle][0][iteration] = 1.0;
            for (int w = 1; w <= ::NPARAMETERSANDONE; w++)
            {
                ::MCMCCHAINS[group][particle][w][iteration] = ::HOLDINGPEN[group][particle][w];
            }
        }
    }
}

void Fcrossover (int group, int particle, int iteration, int burninYN)
{
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
        fitnessvector2.push_back(MCMCCHAINS[group][particle][4][iteration]);
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
    double gamma1;
    double gamma2;
    vector<double> thetaStarVector;
    vector<double> thetaTvector;
    vector<double> thetaBvector;
    vector<double> thetaMvector;
    vector<double> thetaNvector;
    thetaStarVector.push_back(0);
    thetaTvector.push_back(0);
    thetaBvector.push_back(0);
    thetaMvector.push_back(0);
    thetaNvector.push_back(0);
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
        thetaStarVector.push_back(0);
        for (int T=1; T <= ::NPARAMETERS; T++)
        {
            cout << "Checkpoint LACHLAN in crossover. " << endl;
            thetaTvector.push_back(::MCMCCHAINS[group][thetaTindex][T][iteration]);
            thetaBvector.push_back(::MCMCCHAINS[group][thetaBindex][T][iteration]);
            thetaMvector.push_back(::MCMCCHAINS[group][thetaMindex][T][iteration]);
            thetaNvector.push_back(::MCMCCHAINS[group][thetaNindex][T][iteration]);
        }
        thetaStarVector.clear();
        thetaStarVector.push_back(0);
        loopCounter = (loopCounter + 1) % 100;
        cout << "Checkpoint MOLOCH in crossover. " << endl;
        for (int t=1; t <= ::NPARAMETERS; t++)
        {
            cout << "Checkpoint NADIA in crossover. " << endl;
            cout << "We are on parameter " << t << endl;
            if (Frandom01() < ::KAPPA)
            {
                cout << "Checkpoint OLIVIA in crossover. " << endl;
                gamma1 = ((100-loopCounter)/100)*(0.5 + 0.5*(Frandom01())); // if its hard to find a diplacement of the particle that fits
                gamma2 = ((100-loopCounter)/100)*(0.5 + 0.5*(Frandom01())); // then slowly drop the displacement to zero while resampling it
                thetaStarVector.push_back(thetaTvector[t] + gamma1 * (thetaMvector[t] - thetaNvector[t]) + burninYN * gamma2 * (thetaBvector[t] - thetaTvector[t]) + Frandom01()-0.5);
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
            }
            else
            {
                cout << "Checkpoint SAMUEL in crossover. " << endl;
                thetaStarVector.push_back(::MCMCCHAINS[group][thetaTindex][t][iteration]);
            }
            cout << "Checkpoint THOMAS in crossover. " << endl;
        }
        cout << "Checkpoint UFFE in crossover. " << endl;
    }
    cout << "Checkpoint VALKYRIE in crossover. " << endl;
    for (int t = 1; t <= ::NPARAMETERS; t++)
    {
        ::HOLDINGPEN[group][particle][t] = thetaStarVector[t];
    }
    
    FupdateMCMC(group,particle,iteration);
    cout << "Checkpoint WINSTON in crossover. " << endl;
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
