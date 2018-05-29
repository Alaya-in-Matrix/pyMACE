// CEC14_EOTP_C.cpp : Defines the entry point for the console application.
#include "cec14_eotp.h"
#include <iostream>
#include <iomanip>      // std::setprecision
#include <cstdlib>
#include <fstream>
#include <vector>
int test();
using namespace std;
double* read_config(size_t dim);
int main(int arg_num, char* args[])
{
    if(arg_num < 2)
    {
        cerr << "Need problem_no!" << endl;
    }
    bool is_good;
    size_t problem_no = atoi(args[1]);
    double* x  = read_config(10);
    double ret = cec14_eotp_problems(x, 10, problem_no, &is_good);
    delete[] x;
    if(not is_good)
    {
        cerr << "Not good!" << endl;
        for(size_t i = 0; i < 10; ++i)
            cerr << x[i] << endl;
        cerr << "Ret: " << ret << endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        cout << setprecision(16) << ret;
    }
	// double *x;
	// int dim, problem_no;
	// double ret;
	// bool isGood;

	// // test();
	// // Read std::cin
	// std::cin >> dim >> problem_no;
	// x = (double *)malloc(dim*sizeof(double));
	// for (int i = 0; i < dim; i++)
	// 	std::cin >> std::setprecision(16)>> x[i];

	// // calculated the problem
	// ret = cec14_eotp_problems(x, dim, problem_no, &isGood);
	// // output to std::cout
	// if (isGood)
	// 	std::cout << std::setprecision(16) << ret;
	// else
	// 	std::cout << "NaN";

	// free(x);
    return EXIT_SUCCESS;
}

double* read_config(size_t dim)
{
    ifstream ifile;
    ifile.open("./param");
    if((!ifile.is_open()) || ifile.fail())
    {
        cerr << "param file can not open" << endl;
        exit(EXIT_FAILURE);
    }

    double param;
    vector<double> tmp_params;
    while(ifile >> setprecision(16) >> param)
    {
        tmp_params.push_back(param);
    }
    if(tmp_params.size() != dim)
    {
        cerr << "Invalid dimension" << endl;
        exit(EXIT_FAILURE);
    }
    
    double* xs = new double[dim];
    for(size_t i = 0; i < dim; ++i)
        xs[i] = tmp_params[i];
    return xs;
}


// int test()
// {

// 	double x10[10];
// 	double x20[20];
// 	double x30[30];
// 	double *x;
// 	double ret;
// 	int problem_count;
// 	bool isGood;

// 	int function_count, dim_count;


// 	memset(x10, 1, 10 * sizeof(double));
// 	memset(x20, 1, 20 * sizeof(double));
// 	memset(x30, 1, 30 * sizeof(double));


// 	for (function_count = 1; function_count <= 8; function_count++)
// 	{
// 		for (dim_count = 1; dim_count <= 3; dim_count++)
// 		{
// 			if (dim_count == 1)
// 				x = x10;
// 			if (dim_count == 2)
// 				x = x20;
// 			if (dim_count == 3)
// 				x = x30;

// 			problem_count = 3 * (function_count - 1) + dim_count;

// 			ret = cec14_eotp_problems(x, dim_count*10, problem_count,&isGood);
	
// 			if (!isGood)
// 				printf("Error from cec14_eotp_problems.\n");
// 			else
// 				printf("Problem f%d(0) = %f \n", problem_count, ret);
// 		}
// 	}
// 	return 0;
// }
