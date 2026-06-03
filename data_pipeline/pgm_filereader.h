#ifndef PGM_H
#define PGM_H

#include <utility>
#include <iostream>
#include <algorithm>
#include <numeric> 
#include <vector>
#include <string>

using namespace std;

//Returns pair of rows,cols
pair<int,int> rows_and_cols(string filename);

vector<float> read_pgm_file(string filename, bool row_major);

#endif