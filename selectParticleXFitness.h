#include<iostream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <vector>


// Function to return a random number
// from 0 to 1
// uniformly distributed

#ifndef SELECTPARTICLEXFITNESS_H
#define SELECTPARTICLEXFITNESS_H
double random01()
 {
  int elapsed_seconds = time(nullptr);
  srand(elapsed_seconds);
  int randomnum = rand();
  cout << "randomnum: " << randomnum << "\n\n";
  int remainder = randomnum % 10000;
  cout << "remainder: " << remainder << "\n\n";
  double outnum;
  outnum = remainder/10000.0;
  return(outnum);
 }

// Function to select a value from a vector
// The vector has only fitness values
// usually on a log scale
// The function returns the index of a chosen value
// inverseYN 0: select in direct proportion to fitness
//           1: select in inverse proportion to fitness
// logYN 0: not on a log scale
// logYN 1: on a log scale
// return is only the index of the chosen value

int selectParticleXFitness (vector<double> FitnessVector, int inverseYN, int logYN) {
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

  cout << "uFitnessVectorSum: " << uFitnessVectorSum << "\n\n";

  // normalizedRandom is between 0 and uFitnessVectorSum;
  // normalizedRandom = uFitnessVectorSum * rand();
  double randomvar = random01();
  cout << "randomvar: " << randomvar << "\n\n";
  normalizedRandom = randomvar * uFitnessVectorSum;
  cout << "normalizedRandom: " << normalizedRandom << "\n\n";

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
  cout << "cumuFitnessVector: " << cumuFitnessVector << "\n\n";

  // i4908 is the index value of the particle chosen
  return(i4908);
 }
#endif

//int main()
//  {
//    // testing code
//    vector<double> likelihood(4);

//    cout << "Enter the first likelihood: \n ";
//    cin >> likelihood[0];

//    cout << "Enter the second likelihood: \n";
//    cin >> likelihood[1];

//    cout << "Enter the third likelihood: \n";
//    cin >> likelihood[2];

//    cout << "Enter the fourth likelihood: \n";
//    cin >> likelihood[3];

//    cout << "Are these on a log scale? 1,0 \n";
//    int logscaleYN;

//    cin >> logscaleYN;
//    cout << "Select on inverse basis? 1,0 \n";

//    int inversepropYN;
//    cin >> inversepropYN;
    
//    int particleSelected;
//    particleSelected = selectParticleXFitness (likelihood,inversepropYN,logscaleYN);
//    cout << "The particle selected is: " << particleSelected << " \n\n\n";
//  }
