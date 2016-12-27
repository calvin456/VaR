//test_cvar_var.cpp

// Test computation of VaR using different methods

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
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

	for(size_t i = 0;i < n-1;++i)
		prices.push_back(std::stod(csvData[n - 1 - i][6])); //adj close column

    std::shared_ptr<ComputeReturn> cr(new ComputeReturn);

	cr->geometricReturns(prices);

    // Compute point estimate VaR

	cout << endl <<  "Compute daily VaR using different methods - alpha .05" << endl;

	// 1. Riskmetrics

	RiskMetricsVaR var1(.05,.94,true);

	VaRParamCompute<ComputeReturn, RiskMetricsVaR> VaRRiskMetrics(cr, var1);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR() << endl;

	// 2. GARCH

	GarchVaR var2(.05, 0.009076, 0.082916, 0.909160, true);

    VaRParamCompute<ComputeReturn, GarchVaR> VaRGarch(cr, var2);

	cout << "GARCH VaR: " << VaRGarch.computeVaR() << endl;

	// ------------------------------------------------------------

    cr->arithmetricReturns(prices); //switch to arithmetic return

	// 3. Historical method

	HistoricalVaR var3;

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical(cr, var3);

	cout << "Historical VaR: " << VaRHistorical.computeVaR() << endl;

	// 4. Historical method - weighting scheme

	HistoricalVaR var4(.05, .98, hybrid);

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical1(cr, var4);

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR() << endl;

	// 5. Historical method - HW method

	HistoricalVaR var5(.05, .94, hw);

	VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical2(cr, var5);

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR() << endl;

    // ------------------------------------------------------------

	// 8. Peak over threshold - Generalized Pareto dist

	PoTVaR var8(2.30966,1.0468998 , 0.2125231);

	VaRExtremeValueCompute<ComputeReturn, PoTVaR> VaRPoT(cr, var8);

	cout << "Peak over threshold - gen Pareto VaR: " << VaRPoT.computeVaR() << endl;

    // ------------------------------------------------------------

    cr->geometricReturns(prices);

    cout << endl << "Compute daily ES using different methods" << endl;

	cout << "Riskmetrics VaR - ES: " << ExpectedShortfall<VaRParamCompute<ComputeReturn, RiskMetricsVaR>>(VaRRiskMetrics) << endl;

	cout << "Garch VaR - ES: " << ExpectedShortfall<VaRParamCompute<ComputeReturn, GarchVaR>>(VaRGarch) << endl;

	cr->arithmetricReturns(prices); //switch to arithmetic return

	cout << "Historical VaR - ES: " << ExpectedShortfall<VaRnoneParamCompute<ComputeReturn, HistoricalVaR>>(VaRHistorical) << endl;

	cout << "Historical VaR w/ weighting scheme - ES: " << ExpectedShortfall<VaRnoneParamCompute<ComputeReturn, HistoricalVaR>>(VaRHistorical1) << endl;

	cout << "Historical VaR w/ HW weighting scheme - ES: " << ExpectedShortfall<VaRnoneParamCompute<ComputeReturn, HistoricalVaR>>(VaRHistorical2) << endl;

    //cout << "Peak over threshold - gen Pareto VaR - ES: " << ExpectedShortfall<VaRExtremeValueCompute<ComputeReturn, PoTVaR>>(VaRPoT) << endl;
    cout << "Peak over threshold - gen Pareto VaR - ES: " <<  VaRPoT.computeES() << endl;

    // ------------------------------------------------------------

    // Extend rolling period 1 week, 1 month

    cout << endl << "Compute weekly VaR using different methods - alpha .05" << endl;

    VaRRiskMetrics.setPeriod(5);
    cr->geometricReturns(prices);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR() << endl;

	cout << "GARCH VaR: " << VaRGarch.computeVaR() << endl;

	// ------------------------------------------------------------

    cr->arithmetricReturns(prices); //switch to arithmetic return

	cout << "Historical VaR: " << VaRHistorical.computeVaR() << endl;

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR() << endl;

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR() << endl;

	cout << "Peak over threshold - gen Pareto VaR: " << VaRPoT.computeVaR() << endl;

	// ------------------------------------------------------------

    cout << endl << "Compute monthly VaR using different methods - alpha .05" << endl;

    VaRRiskMetrics.setPeriod(21);
    cr->geometricReturns(prices);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR() << endl;

	cout << "GARCH VaR: " << VaRGarch.computeVaR() << endl;

	// ------------------------------------------------------------

    cr->arithmetricReturns(prices); //switch to arithmetic return

	cout << "Historical VaR: " << VaRHistorical.computeVaR() << endl;

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR() << endl;

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR() << endl;

	cout << "Peak over threshold - gen Pareto VaR: " << VaRPoT.computeVaR() << endl;

	return 0;
} catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }

}


