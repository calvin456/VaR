//test_mcr_var.cpp

//compute Monte Carlo VaR

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
    // Compute Monte Carlo VaR

	// Simulate stock rtn using AR(1)xGARCH(1,1) through brute force Monte-Carlo

	AR1xGARCH11 process(-0.0003515,-0.0729,0.01979, 0.09502, 0.89028); // ^GSPC

	HistoricalVaR var1;

	rng _rng;

    VaRMonteCarloCompute<ComputeReturn, HistoricalVaR, AR1xGARCH11> VaRMonteCarlo(cr, var1, process, _rng);

	cout << "Monte Carlo VaR: " << VaRMonteCarlo.computeVaR() << endl;

    //-----------------------------------------------------------------
	// Compute Monte Carlo VaR through whole path

    std::vector<double> VaRWholePath(VaRMonteCarlo.computeVaRWholePath());

    std::string fileName = "/home/mrnoname/Documents/VaR/data/output/var_mc_var_SNP_252.csv";

    ofstream myfile;
    myfile.open (fileName);
    myfile << "Monte Carlo" << endl;
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

