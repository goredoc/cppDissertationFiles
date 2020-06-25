// Matthew Lang - Modified Code
// 6/24/2020

//g++ -Wall TESTcFclearHoldingPen.cpp

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <chrono>
#include <sstream>
#include <string>
//#include <boost/random.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <vector>

#include "cFclearHoldingPen.h"

using namespace std;



int main() 
{
    double holdingPen[NGROUPS][NPARTICLESPERGROUP][NRECORDEDVARIABLES];

    for (int i = 0; i < NGROUPS; i++)
    {
        for (int j = 0; j < NPARTICLESPERGROUP; j++)
        {
            for (int k = 0; k < NRECORDEDVARIABLES; k++)
            {
                holdingPen[i][j][k] = rand() % 10;
                cout << "\t" << holdingPen[i][j][k];
            }
            cout << "\t";
        }
    cout << endl;
    }
    
    // Now call the function to test it
    FclearHoldingPen(holdingPen); // notice we did not put any of the bracketed variables
      // including those index variables would create an error
    // Verify that it is clear (filled with zeros)
    for (int i=0; i < NGROUPS; i++)
    {
        for (int j=0; j < NPARTICLESPERGROUP; j++)
        {
            for (int k=0; k < NRECORDEDVARIABLES; k++)
            {
                cout << "\t" << holdingPen[i][j][k];
            }
            cout << "\t";
        }
    cout << endl;
    }

}
