// Matthew Lang - Modified Code
// 6/24/2020

#include <iostream>

#ifndef CFCLEARHOLDINGPEN_H
#define CFCLEARHOLDINGPEN_H
// FclearHoldingPen fills the holding pen with zeroes

// GLOBALS
// vector<double> holdingPen[group][particle][variable or parameter]
// int nGroups
// int nParticlesPerGroup
// int nRecordedVariables

// GLOBAL DECLARED CONSTANTS
    const int NGROUPS = 6;
    const int NPARTICLESPERGROUP = 3;
    const int NRECORDEDVARIABLES = 2;

void FclearHoldingPen(double holdingPen[NGROUPS][NPARTICLESPERGROUP][NRECORDEDVARIABLES])
// function header main proto call
 {
    for (int i5416 = 0; i5416 < NGROUPS; i5416++)
        {
            for (int j5094=0; j5094 < NPARTICLESPERGROUP; j5094++)
            {
                for (int k2380=0; k2380 < NRECORDEDVARIABLES; k2380++)
                {
                    holdingPen[i5416][j5094][k2380]=0.0;
                }
            }
        }
 }

#endif 
