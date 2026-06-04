#ifndef HEAT_EQ_H
#define HEAT_EQ_H

#include <hip/hip_runtime.h>
#include <utility>
#include <vector>

using namespace std;

vector<float> heat_equation_2d(vector<float> d_u, int rows, int cols, float alpha,
int iter, int gridsize = -1, int blocksize = 256, int stride = 1); //Gridsize is -1 because by default it will be determined by
                                                                   //The blocksize and stride


#endif