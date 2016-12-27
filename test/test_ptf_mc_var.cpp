//test_ptf_mc_var.cpp

//test portfolio VaR through MC simulation

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
#include "portfolio.h"
#include "ptf_var.h"

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
    // Compute Monte Carlo VaR


	// Simulate stock rtn using AR(1)xGARCH(1,1) through brute force Monte-Carlo

	std::vector<AR1xGARCH11> processes(7);

    processes[0] = AR1xGARCH11(-0.0003114, -0.0693, 0.01854, 0.10150, 0.88374); // DJIA
    processes[1] = AR1xGARCH11(-0.0003515,-0.0729,0.01979, 0.09502, 0.89028); // GSPC
    processes[2] = AR1xGARCH11(-0.0004741,-0.1041,0.02788, 0.08605, 0.89754); // NDX
    processes[3] = AR1xGARCH11(-1.37e-17, 0.,0.02614, 0.09003, 0.89950); // GDAXI
    processes[4] = AR1xGARCH11(2.348e-17,0.,0.02476, 0.08731, 0.90240); // FCHI
    processes[5] = AR1xGARCH11(2.468e-17,0.,0.02987, 0.07847, 0.91352); // SSEC
    processes[6] = AR1xGARCH11(5.382e-18,0.,2.166e+00, 4.902e-01, 4.990e-15); // SENSEX

    Path1x1 process;

	HistoricalVaR var1;

    // Case of full replication of index - DJIA,GSPC,NDX,GDAXI,FCHI,SSEC,SENSEX : 7 indices

	double a = double(1./7.); 

	std::vector<double> weights{a,a,a,a,a,a,a}; //initialization. Equi-weighted asset for mere convenience

	Ptf _ptf;

	for(unsigned int i = 0;i < 7;++i){
        shared_ptr<Instrument> instrument(new DeltaOne());
        auto p = std::make_pair(i,instrument);
		_ptf.push_back(p);
	}

	shared_ptr<Portfolio> ptf(new Portfolio(_ptf, weights, cr, false, 1.e+07));

	rng _rng;

    VaRPtfMCCompute<HistoricalVaR, AR1xGARCH11> VaRMonteCarlo(ptf,var1, processes, _rng);

	cout << "Monte Carlo VaR: " << VaRMonteCarlo.computeVaR() << endl;

	//simulate portfolio rtn instead instead of component

	VaRMonteCarloCompute<Portfolio, HistoricalVaR, Path1x1> VaRMonteCarlo1(ptf,var1, process, _rng);

	cout << "Monte Carlo VaR - ptf rtn: " << VaRMonteCarlo1.computeVaR() << endl;

	// Compute whole path

	//Vec myPath = VaRMonteCarlo.computeVaRWholePath();

    return 0;

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}


