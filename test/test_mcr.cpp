//test_mcr.cpp

//compute Monte Carlo sim

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

    //-------------------------------------------------------------------------
    // single case

	// Simulate stock rtn using AR(1)xGARCH(1,1) through brute force Monte-Carlo
    //

	AR1xGARCH11 process (-0.0003114, -0.0693, 0.01854, 0.10150, 0.88374); // DJIA

	rng _rng;

	MCEngine<rng,AR1xGARCH11> engine(_rng, process);

	engine.setValues(cr->getReturns(1),cr->getStdDev(1));

	Vec sim(10);

	sim = engine.DoSimulation(10,Gaussian);

    for(auto& i : sim)   cout << i << endl;

    engine.setValues(cr->getReturns(0),cr->getRollingStdDev(0).back());

    Vec sim1(10);

    sim1 = engine.DoSimulation(10,Gaussian);

    cout << endl; for(auto& j : sim1)   cout << j << endl;

    // --------------------------------------------------------------
    // Multiple stock returns

    Eigen::MatrixXd C = cr->getVarCov();

    Eigen::MatrixXd A( C.llt().matrixL() );

    std::vector<AR1xGARCH11> processes(7);

    processes[0] = AR1xGARCH11(-0.0003114, -0.0693, 0.01854, 0.10150, 0.88374); // DJIA
    processes[1] = AR1xGARCH11(-0.0003515,-0.0729,0.01979, 0.09502, 0.89028); // GSPC
    processes[2] = AR1xGARCH11(-0.0004741,-0.1041,0.02788, 0.08605, 0.89754); // NDX
    processes[3] = AR1xGARCH11(-1.37e-17, 0.,0.02614, 0.09003, 0.89950); // GDAXI
    processes[4] = AR1xGARCH11(2.348e-17,0.,0.02476, 0.08731, 0.90240); // FCHI
    processes[5] = AR1xGARCH11(2.468e-17,0.,0.02987, 0.07847, 0.91352); // SSEC
    processes[6] = AR1xGARCH11(5.382e-18,0.,2.166e+00, 4.902e-01, 4.990e-15); // SENSEX

    MCEngine<rng,AR1xGARCH11> engine1(_rng, processes, A, Cholesky);

    Mat d = cr->getReturns();

    Vec b;

	for(size_t i = 0;i < m;++i) b.push_back(cr->getRollingStdDev(i).back());

    engine1.setValues(d,b);

    Mat sim2(7,Vec(10));

    sim2 = engine1.DoMultiSimulation(10,Gaussian);

    cout << endl;
    for(size_t i =0;i < sim2.size();++i){
        for(size_t j =0;j < sim2[i].size();++j)
            cout << sim2[i][j] << '\t';
        cout << endl;
    }

    //-----------------------------------------------------------------
    // test PCA case

    // convert to req format
    vector<float> vec;

	n = C.rows();
	m = C.cols();

    for(size_t i = 0;i < n;++i)
        for(size_t j = 0;j < m;++j)
            vec.push_back(C(i,j));

	std::shared_ptr<Pca> pca(new Pca());

  	int init_result = pca->Calculate(vec, n, m);

  	if(init_result == 1) cout << "correl mat positive semi-definite " << endl;

    vector<float> scores = pca->scores(); //Rotated data

  	unsigned int kaiser = pca->kaiser(); //Kaiser criterion 99%

    unsigned int nrows = pca->nrows();

	Eigen::MatrixXd PC(nrows, kaiser);

	for(size_t i = 0;i < nrows;++i)
		for(size_t j = 0;j < kaiser;++j)
			PC(i,j) = scores[j + kaiser*i];

    MCEngine<rng,AR1xGARCH11> engine2(_rng, processes, PC, pc);

    engine2.setValues(d,b);

    Mat sim3(7,Vec(10));

    sim3 = engine2.DoMultiSimulation(10,Gaussian);

    cout << endl;
    for(size_t i =0;i < sim2.size();++i){
        for(size_t j =0;j < sim2[i].size();++j)
            cout << sim3[i][j] << '\t';
        cout << endl;
    }

    return 0;

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}


