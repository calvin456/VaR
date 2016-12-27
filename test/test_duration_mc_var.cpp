//test_duration_mc_var.cpp

//test Treasury portfolio VaR through MC simulation

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include<memory>

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

#include "compute_returns_eigen.h"
#include "compute_var.h"
#include "path.h"
#include "ptf_var.h"
#include "rng.h"
#include "portfolio.h"


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

    // Read US Treasury Zero-Coupon Yield Curve
    // daily series obtained from US Federal Reserve Data Releases
	//https://www.quandl.com

	std::fstream file("/home/mrnoname/Documents/VaR/data/TermStructureData.csv", ios::in);
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
                _prices[j-1][i-1062] = 1. - std::stod(tmp)/100.;              
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

	// Simulate yield chge through brute force Monte-Carlo

	Path1x1 process;

	std::vector<Path1x1> processes(7);

	for(size_t i = 0;i < 7;++i) processes[i] = Path1x1();

	HistoricalVaR var1;

    // Portfolio with 7 durations across the yield curve

	double a = double(1./7.);

	std::vector<double> weights{a,a,a,a,a,a,a}; //initialization. Equi-weighted asset for mere convenience

	Ptf _ptf;

	_ptf.push_back(std::make_pair(0,shared_ptr<Instrument> (new FI(.960/100.)))); //1 yr
	_ptf.push_back(std::make_pair(1,shared_ptr<Instrument> (new FI(1.918/100.)))); //2 yrs
	_ptf.push_back(std::make_pair(2,shared_ptr<Instrument> (new FI(2.913/100.)))); //3 yrs
	_ptf.push_back(std::make_pair(4,shared_ptr<Instrument> (new FI(4.704/100.)))); //5 yrs
	_ptf.push_back(std::make_pair(6,shared_ptr<Instrument> (new FI(6.406/100.)))); // 7 yrs
	_ptf.push_back(std::make_pair(9,shared_ptr<Instrument> (new FI(8.874/100.)))); // 10  yrs
	_ptf.push_back(std::make_pair(29,shared_ptr<Instrument> (new FI(19.592/100.)))); // 30 yrs

    /*
    http://online.wsj.com/mdc/public/page/2_3022-bondmkt.html

    The Bond Market: Ryan Indexes
    Tuesday, December 20, 2016

    1 yr Treasury .960
    2 yr Treasury 1.918
    3 yr Treasury 2.913
    5 yr Treasury 4.704
    7 yr Treasury 6.406
    10 yr Treasury 8.874
    30 yr Treasury 19.592
    */

	shared_ptr<Portfolio> ptf(new Portfolio(_ptf, weights, cr, true, 1.e+07));

	rng _rng;

    //ptf value at risk

	VaRPtfMCCompute<HistoricalVaR, Path1x1> VaRMonteCarlo(ptf,var1, processes, _rng);

	cout << "Monte Carlo VaR: " << VaRMonteCarlo.computeVaR() << endl;

	//simulate portfolio rtn instead instead of component

	VaRMonteCarloCompute<Portfolio, HistoricalVaR, Path1x1> VaRMonteCarlo1(ptf,var1, process, _rng);

	cout << "Monte Carlo VaR - ptf rtn: " << VaRMonteCarlo1.computeVaR() << endl;


    return 0;

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}

