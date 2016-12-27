//test_ptf_var.cpp

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include<memory>

#include <Eigen/Dense>

#include "compute_returns_eigen.h"
#include "portfolio.h"
#include "instrument.h"
#include "ptf_var.h"
#include "var_model.h"
#include "compute_var.h"

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

    // Read mid FX fix for currency pairs majors and exotics
    // daily series obtained for Bank of England through
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

    Mat _rtns = cr->getReturns();

    // ------------------------------------------------

    // Case of full replication of index - DJIA,GSPC,NDX,GDAXI,FCHI,SSEC,SENSEX : 7 indices

	double a = double(1./7.); //cout << a << endl;

	std::vector<double> weights{a,a,a,a,a,a,a}; //initialization. Equi-weighted asset for mere convenience

	Ptf _ptf;

	for(unsigned int i = 0;i < 7;++i){
        shared_ptr<Instrument> instrument(new DeltaOne());
        auto p = std::make_pair(i,instrument);
		_ptf.push_back(p);
	}

	shared_ptr<Portfolio> ptf(new Portfolio(_ptf, weights, cr, false, 1.e+07));

	cout << "ptf's avg rtn: " << ptf->getMeanPtfRn() << endl;
	cout << "ptf's vol: " << ptf->getPtfSdev() << endl<< endl;

	double alpha = .05;

	VaRPtfCompute model(ptf, alpha);

    // Compute ptf VaR of index

	cout << endl << "Portfolio VaR - equi index " << alpha << " : " << model.getPtfVaR() << endl;

	//-----------------------------------------------------------------
	cout << endl <<  "Compute daily VaR using different methods - alpha .05" << endl;

	// 1. Riskmetrics

	RiskMetricsVaR var1; //(.05,.94,false);

	VaRParamCompute<Portfolio, RiskMetricsVaR> VaRRiskMetrics(ptf, var1);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR() << endl;

	// 2. GARCH

	GarchVaR var2; //(.05, 0., .25, .75, false);

    VaRParamCompute<Portfolio, GarchVaR> VaRGarch(ptf, var2);
    
	cout << "GARCH VaR: " << VaRGarch.computeVaR() << endl;

	// ------------------------------------------------------------

	// 3. Historical method

	HistoricalVaR var3;

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical(ptf, var3);

	cout << "Historical VaR: " << VaRHistorical.computeVaR() << endl;

	// 4. Historical method - weighting scheme

	HistoricalVaR var4(.05, .98, hybrid);

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical1(ptf, var4);

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR() << endl;

	// 5. Historical method - HW method

	HistoricalVaR var5(.05, .94, hw);

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical2(ptf, var5);

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR() << endl;

	//-----------------------------------------------------------------

	// Compute conponent VaR

	Vec compVaR = model.computeComponentVaR();

	cout << endl << "component VaR: " << endl;

	for(size_t i = 0;i < 7;++i)
		cout << indexNames[i] << ": " << compVaR[i] << endl;

    double sumCompVaR(0.);
    for(auto& i : compVaR) sumCompVaR += i;
    cout << "sum component VaR: " << sumCompVaR << endl;

	// Compute marginal VaR

	Vec marVaR = model.computeMarginalVaR();

	cout << endl << "marginal VaR: " << endl;

	for(size_t i = 0;i < 7;++i)
		cout << indexNames[i] << ": " << marVaR[i] << endl;

	// Compute incremental VaR

	double amount = 1.e+06;

	cout << endl << "incremental VaR - add " << amount << " : " << model.computeIncrementalVaR(amount) << endl;

	//------------------------------------------------------------------------------------------------

    weights = {.1,.15,.2,.1,.1,.2,.15};

	shared_ptr<Portfolio> ptf1(new Portfolio(_ptf, weights, cr, false, 1.e+07));

	cout << "ptf's avg rtn: " << ptf1->getMeanPtfRn() << endl;
	cout << "ptf's vol: " << ptf1->getPtfSdev() << endl<< endl;

	VaRPtfCompute model1(ptf1, alpha);

    // Compute ptf VaR of index

	cout << endl << "Portfolio VaR - active index " << alpha << " : " << model1.getPtfVaR() << endl;

	//-----------------------------------------------------------------
	cout << endl <<  "Compute daily VaR using different methods - alpha .05" << endl;

	// 1. Riskmetrics

	VaRParamCompute<Portfolio, RiskMetricsVaR> VaRRiskMetrics1(ptf1, var1);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics1.computeVaR() << endl;

	// 2. GARCH

    VaRParamCompute<Portfolio, GarchVaR> VaRGarch1(ptf1, var2);

	cout << "GARCH VaR: " << VaRGarch1.computeVaR() << endl;

	// ------------------------------------------------------------

	// 3. Historical method

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical11(ptf1, var3);

	cout << "Historical VaR: " << VaRHistorical11.computeVaR() << endl;

	// 4. Historical method - weighting scheme

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical12(ptf1, var4);

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical12.computeVaR() << endl;

	// 5. Historical method - HW method

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical13(ptf1, var5);

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical13.computeVaR() << endl;

	//-----------------------------------------------------------------

	// Compute conponent VaR

	compVaR = model1.computeComponentVaR();

	cout << endl << "component VaR: " << endl;

	for(size_t i = 0;i < 7;++i)
		cout << indexNames[i] << ": " << compVaR[i] << endl;

    sumCompVaR = 0.;
    for(auto& i : compVaR) sumCompVaR += i;
    cout << "sum component VaR: " << sumCompVaR << endl;

	// Compute marginal VaR

	marVaR = model1.computeMarginalVaR();

	cout << endl << "marginal VaR: " << endl;

	for(size_t i = 0;i < 7;++i)
		cout << indexNames[i] << ": " << marVaR[i] << endl;

	// Compute incremental VaR

	cout << endl << "incremental VaR - add " << amount << " : " << model1.computeIncrementalVaR(amount) << endl;

	//-----------------------------------------------------------------

	// Buy 1 call and 1 put on S&P 500. Sell .5 unit of index

	// Case of derivatives, full replication of index

	Vec weights1 =  {1.,1.,-.5};

	Ptf _ptf1;

	shared_ptr<Instrument> instrument(new Derivatives(0.46118, 0.01013));
    _ptf1.push_back(std::make_pair(1,instrument));
    shared_ptr<Instrument> instrument1(new Derivatives(-0.52983, 0.00658));
    _ptf1.push_back(std::make_pair(1,instrument1));
    shared_ptr<Instrument> instrument2(new DeltaOne());
	_ptf1.push_back(std::make_pair(1,instrument2));

	/*
    Call @SPX 161216C02200000 Delta0.46118 Gamma0.01013 Rho0.38430 Theta-0.79242 Vega1.69184 Impvol0.11648

    Put @SPX 161216P02200000 Delta-0.52983 Gamma0.00658 Rho-0.34403 Theta-0.93300 Vega1.67799 Impvol0.11835

	*/

	shared_ptr<Portfolio> ptf2(new Portfolio(_ptf1, weights1, cr, false, 1.e+07));

    VaRPtfCompute model2(ptf2, alpha);

    // Compute ptf VaR of index

	cout << endl << "Portfolio VaR - equity derivatives " << alpha << " : " << model2.getPtfVaR() << endl;

	//-----------------------------------------------------------------
	cout << endl <<  "Compute daily VaR using different methods - alpha .05" << endl;

	// 1. Riskmetrics

	VaRParamCompute<Portfolio, RiskMetricsVaR> VaRRiskMetrics21(ptf2, var1);

	cout << "Riskmetrics VaR: " << VaRRiskMetrics21.computeVaR() << endl;

	// 2. GARCH

    VaRParamCompute<Portfolio, GarchVaR> VaRGarch22(ptf1, var2);

	cout << "GARCH VaR: " << VaRGarch22.computeVaR() << endl;

	// ------------------------------------------------------------

	// 3. Historical method

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical21(ptf1, var3);

	cout << "Historical VaR: " << VaRHistorical21.computeVaR() << endl;

	// 4. Historical method - weighting scheme

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical22(ptf1, var4);

	cout << "Historical VaR w/ weighting scheme: " << VaRHistorical22.computeVaR() << endl;

	// 5. Historical method - HW method

	VaRnoneParamCompute<Portfolio, HistoricalVaR> VaRHistorical23(ptf1, var5);

	cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical23.computeVaR() << endl;

	// Compute conponent VaR

	Vec compVaR2 = model2.computeComponentVaR();

	cout << endl << "component VaR: " << endl;

	for(size_t i = 0;i < 3;++i)
		cout << i << ": " << compVaR2[i] << endl;

    sumCompVaR = 0.;
    for(auto& i : compVaR2) sumCompVaR += i;
    cout << "sum component VaR: " << sumCompVaR << endl;

	// Compute marginal VaR

	Vec marVaR2 = model2.computeMarginalVaR();

	cout << endl << "marginal VaR: " << endl;

	for(size_t i = 0;i < 3;++i)
		cout << i << ": " << marVaR2[i] << endl;

	// Compute incremental VaR

	cout << endl << "incremental VaR - add " << amount << " : " << model2.computeIncrementalVaR(amount) << endl;

    // ------------------------------------------------------------

    return 0;

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }


}


