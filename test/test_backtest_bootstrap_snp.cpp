//test_backtest_bootstrap_snp.cpp

// Test computation of VaR using different methods

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include<algorithm> 
#include<memory>

#include "compute_returns_eigen.h"
#include "compute_var.h"

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


int main(){


try{
	// Read S&P 500 index data from csv file (GSPC)
	// daily close from 1950-01-03 to 2016-10-27
	//https://www.quandl.com/data/YAHOO/INDEX_GSPC-S-P-500-Index

	std::fstream file("/home/mrnoname/Documents/VaR/data/table.csv", ios::in);
	if(!file.is_open())
	{
		std::cout << "File not found!\n";
		return 1;
	}
	// typedef to save typing for the following object
	typedef std::vector< std::vector<std::string> > csvVector;
	csvVector csvData;

	readCSV(file, csvData);

	std::vector<double> prices;

	unsigned int n(csvData.size());

	for(size_t i =0 ;i < n-1;++i)
		prices.push_back(std::stod(csvData[n - 1 - i][6])); //adj close column

    std::shared_ptr<ComputeReturn> cr(new ComputeReturn(1,1008));

	cr->geometricReturns(prices);

	Vec v = cr->getReturns(0);

    // Compute point estimate VaR

	cout << endl <<  "Compute daily VaR using different methods - alpha .05" << endl;

	// 1. Riskmetrics

	RiskMetricsVaR var1(.05,.94,false);

	VaRParamCompute<ComputeReturn, RiskMetricsVaR> VaRRiskMetrics(cr, var1);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR() << endl;

	std::vector<double> VaRWholePath(VaRRiskMetrics.computeVaRWholePath());

	// 2. GARCH

	//## Coefficient(s):
    //##     Estimate  Std. Error  t value Pr(>|t|)
    //## a0  0.031941    0.003913    8.162 2.22e-16 ***
    //## a1  0.123563    0.011042   11.191  < 2e-16 ***
    //## b1  0.853231    0.012166   70.132  < 2e-16 ***

	GarchVaR var2(.05, 0.031941, 0.123563, 0.853231, false);

    VaRParamCompute<ComputeReturn, GarchVaR> VaRGarch(cr, var2);

	cout << "GARCH VaR: " << VaRGarch.computeVaR() << endl;

	std::vector<double> VaRWholePath1(VaRGarch.computeVaRWholePath());

	// ------------------------------------------------------------

	// 3. Historical method

	HistoricalVaR var3;

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical(cr, var3);

	cout << "Historical VaR: " << VaRHistorical.computeVaR() << endl;

	std::vector<double> VaRWholePath2(VaRHistorical.computeVaRWholePath());

	// 4. Historical method - weighting scheme

	HistoricalVaR var4(.05, .98, hybrid);

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical1(cr, var4);

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR() << endl;

	std::vector<double> VaRWholePath3(VaRHistorical1.computeVaRWholePath());

	// 5. Historical method - HW method

	HistoricalVaR var5(.05, .94, hw);

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical2(cr, var5);

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR() << endl;

	std::vector<double> VaRWholePath4(VaRHistorical2.computeVaRWholePath());

    // ------------------------------------------------------------

	// 8. Peak over threshold - Generalized Pareto dist

	PoTVaR var8(2.30966,1.0468998 , 0.2125231);

	VaRExtremeValueCompute<ComputeReturn, PoTVaR> VaRPoT(cr, var8);

	cout << "Peak over threshold - gen Pareto VaR: " << VaRPoT.computeVaR() << endl;

	std::vector<double> VaRWholePath5(VaRPoT.computeVaRWholePath());

    // ------------------------------------------------------------

    std::string fileName = "/home/mrnoname/Documents/VaR/data/output/var_risk_metrics_SNP_1.csv";

    ofstream myfile;
    myfile.open (fileName);
    myfile << "Riskmetrics" << "," << "GARCH"
    << "," << "HistSimul"
    << "," << "HistSimulWght"
    << "," << "HistSimulVolWght"
    << "," << "POTPareto"<< endl;
    for(size_t i = 0;i < VaRWholePath.size();++i){
    	myfile << VaRWholePath[i] << "," << VaRWholePath1[i]
        << "," << VaRWholePath2[i] << "," << VaRWholePath3[i] << "," << VaRWholePath4[i]
        << "," << VaRWholePath5[i] <<endl;
   	}
    myfile.close();


	return 0;
} catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}
