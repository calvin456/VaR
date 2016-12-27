//test_copula.cpp

/* Compute VaR using Monte Carlo sim and copula

1. copula DJIA vs 6 indexes

	- Gaussian (Normal), t, Clayton, Gumbel (Logistic) copulas

2. Gaussian copula DJIA over the whole path

*/

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include<memory>

#include <Eigen/Dense>

#include "compute_returns_eigen.h"
#include "compute_var.h"
#include "path.h"
#include "mc_engine.h"
#include "rng.h"
#include "pca.h"

using namespace Eigen;
using namespace std;

void readCSV(std::istream &input, std::vector< std::vector<std::string> > &output)
//https://www.gamedev.net/topic/444193-c-how-to-load-in-a-csv-file/
{
	std::string csvLine;
	// read every line from the stream
	while( std::getline(input, csvLine) )
	{
		std::istringstream csvStream(csvLine);
		std::vector<std::string> csvColumn;
		std::string csvElement;
		// read every element from the line that is seperated by commas
		// and put it into the vector or strings
		while( std::getline(csvStream, csvElement, ',') )
		{
			csvColumn.push_back(csvElement);
		}
		output.push_back(csvColumn);
	}
}


int main()
{

    try{

    // Read daily index level for major and emerging markets
    // daily series obtained from Yahoo! Finance through
	//https://www.quandl.com

	std::fstream file("/home/mrnoname/Documents/VaR/data/StockIndexData.csv", ios::in);
	if(!file.is_open())
	{
		std::cout << "File not found!\n";
		return 1;
	}
	// typedef to save typing for the following object
	typedef std::vector< std::vector<std::string> > csvVector;
	csvVector csvData;

	readCSV(file, csvData);

    //test
    for(size_t i = 0;i < 5; ++i){
        for(size_t j = 0;j < csvData[i].size();++j){
            cout << csvData[i][j] << '\t';

        }

        cout << endl;
    }
    cout << endl;

    // Remove lines with missing values

    size_t n(csvData.size() - 1);
    size_t m(csvData[0].size() - 1);

    Mat _prices;
    _prices.resize(m,Vec(n-1062));

    for(size_t i = 1062;i < n;++i){
        for(size_t j = 1;j < csvData[i].size();++j){
            std::string tmp = csvData[i][j];
            if(tmp.empty()){
                _prices[j-1][i-1062] = 99999.;
            }
            else{
                _prices[j-1][i-1062] = std::stod(tmp);
            }
        }
    }

    std::vector<std::string> indexNames(csvData[0].size() - 1);

    for(size_t i = 1;i < csvData[0].size();++i){
        indexNames[i-1] = csvData[0][i];
    }

	//Remove missing values to compute trailling returns
    //Asynchornous time series. Shift to the next value
    Mat prices;
    prices.resize(m,Vec(0));

	for(size_t i = 0;i < _prices.size();++i){
        for(size_t j = 0;j < _prices[i].size();++j){
            if(!((_prices[i][j] == 99999) || (_prices[i][j] == 0)))
                prices[i].push_back(_prices[i][j]);
        }
	}

    std::shared_ptr<ComputeReturn> cr(new ComputeReturn(prices,1,252,true));
	// 252 / 4 = 63 - 3 months
    // 4 * 252 = 1008 use 4 years of data to compute mean, and std dev

    /*

	1. copula DJIA vs 6 indexes

		- Gaussian (Normal), t, Clayton, Gumbel (Logistic) copulas

	*/

	std::vector<AR1xGARCH11> processes(7);

    processes[0] = AR1xGARCH11(-0.0003114, -0.0693, 0.01854, 0.10150, 0.88374); // DJIA
    processes[1] = AR1xGARCH11(-0.0003515,-0.0729,0.01979, 0.09502, 0.89028); // GSPC
    processes[2] = AR1xGARCH11(-0.0004741,-0.1041,0.02788, 0.08605, 0.89754); // NDX
    processes[3] = AR1xGARCH11(-1.37e-17, 0.,0.02614, 0.09003, 0.89950); // GDAXI
    processes[4] = AR1xGARCH11(2.348e-17,0.,0.02476, 0.08731, 0.90240); // FCHI
    processes[5] = AR1xGARCH11(2.468e-17,0.,0.02987, 0.07847, 0.91352); // SSEC
    processes[6] = AR1xGARCH11(5.382e-18,0.,2.166e+00, 4.902e-01, 4.990e-15); // SENSEX

	Eigen::MatrixXd C(7,7);

/*
rho.1   9.714e-01  6.833e-04 1421.637  < 2e-16 ***
## rho.2   4.661e-01  1.195e-02   38.994  < 2e-16 ***
## rho.3  -3.060e-02  1.702e-02   -1.798  0.07225 .
## rho.4  -3.929e-02  1.701e-02   -2.310  0.02092 *
## rho.5  -1.696e-02  1.705e-02   -0.995  0.31989
## rho.6  -1.430e-02  1.705e-02   -0.839  0.40166

## rho.7   4.915e-01  1.157e-02   42.464  < 2e-16 ***
## rho.8  -2.463e-02  1.703e-02   -1.447  0.14801
## rho.9  -4.021e-02  1.701e-02   -2.364  0.01809 *
## rho.10 -1.877e-02  1.705e-02   -1.101  0.27075
## rho.11 -1.387e-02  1.705e-02   -0.814  0.41593

## rho.12 -4.980e-02  1.699e-02   -2.931  0.00338 **
## rho.13  1.646e-06  1.705e-02    0.000  0.99992
## rho.14  1.580e-03  1.705e-02    0.093  0.92620
## rho.15 -5.006e-03  1.706e-02   -0.293  0.76914

## rho.16 -1.738e-02  1.705e-02   -1.019  0.30805
## rho.17  1.913e-02  1.705e-02    1.122  0.26171
## rho.18  8.482e-03  1.705e-02    0.497  0.61893

## rho.19  9.321e-03  1.705e-02    0.547  0.58467
## rho.20 -1.856e-02  1.705e-02   -1.089  0.27628

## rho.21  1.395e-02  1.705e-02    0.818  0.41315
*/

	C(0,0) = 1.;
	C(1,1) = 1.;
	C(2,2) = 1.;
	C(3,3) = 1.;
	C(4,4) = 1.;
	C(5,5) = 1.;
	C(6,6) = 1.;


	C(0,1) = 9.714e-01; C(1,0) = 9.714e-01;
	C(0,2) = 4.661e-01; C(2,0) = 4.661e-01;
	C(0,3) = -3.060e-02; C(3,0) = -3.060e-02;
	C(0,4) = -3.929e-02; C(4,0) = -3.929e-02;
	C(0,5) = -1.696e-02; C(5,0) = -1.696e-02;
	C(0,6) = -1.430e-02; C(6,0) = -1.430e-02;

	C(1,2) = 4.915e-01; C(2,1) = 4.915e-01;
	C(1,3) = -2.463e-02; C(3,1) = -2.463e-02;
	C(1,4) = -4.021e-02; C(4,1) = -4.021e-02;
	C(1,5) = -1.877e-02; C(5,1) = -1.877e-02;
	C(1,6) = -1.387e-02; C(6,1) = -1.387e-02;

	C(2,3) = -4.980e-02; C(3,2) = -4.980e-02;
	C(2,4) = 1.646e-06; C(4,2) = 1.646e-06;
	C(2,5) = 1.580e-03; C(5,2) = 1.580e-03;
	C(2,6) = -5.006e-03; C(6,3) = -5.006e-03;

	C(3,4) = -1.738e-02; C(4,3) = -1.738e-02;
	C(3,5) = 1.913e-02; C(5,3) = 1.913e-02;
	C(3,6) = 8.482e-03; C(6,3) = 8.482e-03;

	C(4,5) = 9.321e-03; C(5,4) = 9.321e-03;
	C(4,6) = -1.856e-02; C(6,4) = -1.856e-02;

	C(5,6) = 1.395e-02; C(6,5) = 1.395e-02;

	HistoricalVaR var1;

	rng _rng;

    VaRCopulaCompute<ComputeReturn, HistoricalVaR, AR1xGARCH11> VaRCopula(cr, var1, processes, _rng, C);

    size_t w(cr->getReturns().size() -1);

	cout << endl << "Gaussian copula VaR: " << endl;
	Vec _VaRs = VaRCopula._computeVaR(1., 1., Gauss, Gaussian,w - 1,w);
	cout << "DJIA: " << _VaRs[0] << endl;
	cout << "GSPC: " << _VaRs[1] << endl;
	cout << "NDX: " << _VaRs[2] << endl;
	cout << "GDAXI: " << _VaRs[3] << endl;
	cout << "FCHI: " << _VaRs[4] << endl;
	cout << "SSEC: " << _VaRs[5] << endl;
	cout << "SENSEX: " << _VaRs[6] << endl;

	double dof = 10.5921609;

/*
## rho.1   0.9713667  0.0007552 1286.156  < 2e-16 ***
## rho.2   0.5145676  0.0124411   41.360  < 2e-16 ***
## rho.3  -0.0293183  0.0177551   -1.651  0.09869 .
## rho.4  -0.0316248  0.0178154   -1.775  0.07588 .
## rho.5  -0.0133095  0.0177248   -0.751  0.45272
## rho.6  -0.0034288  0.0176745   -0.194  0.84618

## rho.7   0.5414400  0.0120706   44.856  < 2e-16 ***
## rho.8  -0.0242672  0.0177692   -1.366  0.17204
## rho.9  -0.0333458  0.0178330   -1.870  0.06150 .
## rho.10 -0.0159670  0.0177309   -0.901  0.36785
## rho.11 -0.0038308  0.0176985   -0.216  0.82864

## rho.12 -0.0473888  0.0177987   -2.662  0.00776 **
## rho.13 -0.0045290  0.0179107   -0.253  0.80037
## rho.14 -0.0062200  0.0177792   -0.350  0.72645
## rho.15  0.0038222  0.0177604    0.215  0.82960

## rho.16 -0.0198613  0.0175692   -1.130  0.25828
## rho.17  0.0191351  0.0175063    1.093  0.27438
## rho.18  0.0013937  0.0175002    0.080  0.93653

## rho.19  0.0018595  0.0175989    0.106  0.91585
## rho.20 -0.0199084  0.0174932   -1.138  0.25509

## rho.21  0.0196046  0.0174675    1.122  0.26172
*/

	C(0,0) = 1.;
	C(1,1) = 1.;
	C(2,2) = 1.;
	C(3,3) = 1.;
	C(4,4) = 1.;
	C(5,5) = 1.;
	C(6,6) = 1.;

	C(0,1) = 0.9713667; C(1,0) = 0.9713667;
	C(0,2) = 0.5145676; C(2,0) = 0.5145676;
	C(0,3) = -0.0293183; C(3,0) = -0.0293183;
	C(0,4) = -0.0316248; C(4,0) = -0.0316248;
	C(0,5) = -0.0133095; C(5,0) = -0.0133095;
	C(0,6) = -0.0034288; C(6,0) = -0.0034288;

	C(1,2) = 0.5414400; C(2,2) = 0.5414400;
	C(1,3) = -0.0242672; C(3,2) = -0.0242672;
	C(1,4) = -0.0333458; C(4,2) = -0.0333458;
	C(1,5) = -0.0159670; C(5,2) = -0.0159670;
	C(1,6) = -0.0038308; C(6,2) = -0.0038308;

	C(2,3) = -0.0473888; C(3,2) = -0.0473888;
	C(2,4) = -0.0045290; C(4,2) = -0.0045290;
	C(2,5) = -0.0062200; C(5,2) = -0.0062200;
	C(2,6) = 0.0038222; C(6,2) = 0.0038222;

	C(3,4) = -0.0198613; C(4,3) = -0.0198613;
	C(3,5) = 0.0191351; C(5,3) = 0.0191351;
	C(3,6) = 0.0013937; C(6,3) = 0.0013937;

	C(4,5) = 0.0018595; C(5,4) = 0.0018595;
	C(4,6) = -0.0199084; C(6,4) = -0.0199084;

	C(5,6) = 0.0196046; C(6,5) = 0.0196046;

	VaRCopula.setCorrelMat(C);

	cout << endl << "t copula VaR: " << endl;
	Vec _VaRs1 = VaRCopula._computeVaR(dof, 1., t, Student,w - 1,w);
	cout << "DJIA: " << _VaRs1[0] << endl;
	cout << "GSPC: " << _VaRs1[1] << endl;
	cout << "NDX: " << _VaRs1[2] << endl;
	cout << "GDAXI: " << _VaRs1[3] << endl;
	cout << "FCHI: " << _VaRs1[4] << endl;
	cout << "SSEC: " << _VaRs1[5] << endl;
	cout << "SENSEX: " << _VaRs1[6] << endl;

	cout << endl << "t copula VaR: " << endl;

	for(double i = 1;i <= 10;++i)
        cout << "GSPC - dof " << i << ": " <<  VaRCopula._computeVaRSingle(1, i, 1., t, Student,w - 1,w) << endl;

	cout << endl << "Clayton copula VaR: " << endl;
	Vec _VaRs2 = VaRCopula._computeVaR(2., 0.095633, Clayton, Gaussian,w - 1,w);
	cout << "DJIA: " << _VaRs2[0] << endl;
	cout << "GSPC: " << _VaRs2[1] << endl;
	cout << "NDX: " << _VaRs2[2] << endl;
	cout << "GDAXI: " << _VaRs2[3] << endl;
	cout << "FCHI: " << _VaRs2[4] << endl;
	cout << "SSEC: " << _VaRs2[5] << endl;
	cout << "SENSEX: " << _VaRs2[6] << endl;

	cout << endl << "Gumbel copula VaR: " << endl;
	Vec _VaRs3 = VaRCopula._computeVaR(2., .99, Gumbel, Gaussian,w - 1,w); //1.043675
	cout << "DJIA: " << _VaRs3[0] << endl;
	cout << "GSPC: " << _VaRs3[1] << endl;
	cout << "NDX: " << _VaRs3[2] << endl;
	cout << "GDAXI: " << _VaRs3[3] << endl;
	cout << "FCHI: " << _VaRs3[4] << endl;
	cout << "SSEC: " << _VaRs3[5] << endl;
	cout << "SENSEX: " << _VaRs3[6] << endl;

	// 2. Gaussian copula GSPC over the whole path

	cout << "start computing whole path ..." << endl;

	std::vector<double> VaRWholePath(VaRCopula.computeVaRWholePath(1));

    std::string fileName = "/home/mrnoname/Documents/VaR/data/output/var_copula_var_SNP_252.csv";

    ofstream myfile;
    myfile.open (fileName);
    myfile << "Copula" << endl;
    for(size_t i = 0;i < VaRWholePath.size();++i){
    	myfile << VaRWholePath[i]  << endl;
    }
    myfile.close();

    return 0;

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}


