/*
%CEC14_EOTP_PROBLEMS Implement CEC14 expensive optimization test problems.
%   F = CEC14_EOTP_PROBLEMS(X, DIM, NO) with X as the design
%   points and NO as the problem' number (in [1,2,...,24]). F
%   should be the evaluation of F_no(x). X is a vector coresponds to
%   one design point with a dimension dim.
%
%   Example
%         f=cec14_eotp_problems({0,0,0,0,0,0,0,0,0,0},10, 1);
%       calculate the 1st problem at (0;0;0;0;0;0;0;0;0;0)
%
%	This function will verify the dims is consistent with the definitions.
%
%   Details are to be found in B. Liu, Q. Chen and Q. Zhang, J. J. Liang,
%   P. N. Suganthan, B. Y. Qu, "Problem Definitions and Evaluation Criteria
%   for Computationally Expensive Single Objective Numerical Optimization",
%   Technical Report.
%
%   If you have any question, please contact Qin Chen (cheqin1980@gmail.com)
%
%   $Autor   : Qin Chen$
%   $E-mail  : cheqin1980@gmail.com$
%   $Revision: 1.0 $  $Date: 2013/12/21 20:14:00 $




Function name                  Function No.		Problem No.    Dimension
'Sphere function',				1				[1,  2, 3]     10,20,30
'Ellipsoid function',			2				[4,  5, 6]     10,20,30
'Rotated Ellipsoid function',	3				[7,  8, 9]     10,20,30
'Step function',				4				[10,11,12]     10,20,30
'Ackley function',				5				[13,14,15]     10,20,30
'Griewank function',			6				[16,17,18]     10,20,30
'Rosenbrock function',			7				[19,20,21]     10,20,30
'Rastrigin function',			8				[22,23,24]     10,20,30
*/


double cec14_eotp_problems(const double * x, int dim, int no, bool * isGood);