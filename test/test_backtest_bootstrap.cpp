//test_backtest_bootstrap.cpp

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


typedef std::vector<double> Vec;

 // 2. GARCH

	// CHFEUR a0 : 0.08898 		a1 : 0.16399 	b1 : 0.57424
	// DKKEUR a0 : 2.504e-06 	a1 : 8.882e-02 	b1 : 9.070e-01
	// CZKEUR a0 : 1.336e-14 	a1 : 3.473e-01 	b1 : 7.957e-01
	// BRLUSD a0 : 0.2025 		a1 : 0.1120 	b1 : 0.7054

	Vec v1{0.08898,0.16399,0.57424};
	Vec v2{2.504e-06,8.882e-02,9.070e-01};
	Vec v3{1.336e-14,3.473e-01,7.957e-01};
	Vec v4{0.2025,0.1120,0.7054};

	static std::map<std::string, Vec> GARCHDict = {
                                  {"CHFEUR", v1},
                                  {"DKKEUR", v2},
                                  {"CZKEUR", v3},
                                  {"BRLUSD", v4}
                                  };

// 6. Peak over threshold - Generalized Pareto dist

    // CHFEUR u =.4786,     sigma/beta = 0.1787757,     xi = 0.7724900
    // DKKEUR u = .0188,    sigma/beta = 0.01109081,   xi = 0.11716942
    // CKZEUR u = .0668,     sigma/beta = 0.20086837,    xi = 0.01995657
    // BRLUSD u = 1.9838,    sigma/beta = 0.5069357,   xi = 0.2589703

    Vec v11{.4786,0.1787757,0.7724900};
    Vec v21{.0188,0.01109081,0.11716942};
    Vec v31{.0668,0.20086837,0.01995657};
    Vec v41{1.9838,0.5069357,0.2589703};

    static std::map<std::string, Vec> GPDDict = {
                                  {"CHFEUR", v11},
                                  {"DKKEUR", v21},
                                  {"CZKEUR", v31},
                                  {"BRLUSD", v41}
                                  };


int main()
{

    try{

    // Read mid FX fix for currency pairs majors and exotics
    // daily series obtained for Bank of England through
	//https://www.quandl.com

	std::fstream file("/home/mrnoname/Documents/VaR/data/FXData.csv", ios::in);
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
        for(size_t j = 0;j < 13;++j){
            cout << csvData[i][j] << '\t';
        }

        cout << endl;
    }
    cout << endl;

    // Remove lines with missing values

    size_t n(csvData.size() - 1);
    size_t m(csvData[0].size() - 1);

    Mat _prices;
    _prices.resize(m,Vec(n));

    for(size_t j = 0;j < n;++j){
        //cout << j << '\t';
        for(size_t i = 0;i < m;++i){
            //cout << csvData[j+1][i + 1] << endl;
            std::string tmp = csvData[j+1][i + 1];
            if(tmp.empty()) _prices[i][j] = 99999.;
            else    _prices[i][j] = std::stod(tmp);
        }
    }

    std::vector<std::string> pairNames;

    for(size_t i = 0;i < m;++i)
        pairNames.push_back(csvData[0][i + 1]);


	//Remove missing values to compute trailling returns
    //Asynchornous time series. Shift to the next value
    Mat prices;
    prices.resize(m,Vec(0));

	for(size_t i = 0;i < m;++i){

        for(size_t j = 0;j < n;++j){
            if(_prices[i][j] == 99999.){
                if(j + 1 < n)   prices[i].push_back(_prices[i][j+1]);
                else break;
            }
            else prices[i].push_back(_prices[i][j]);
        }
	}

	std::vector<unsigned int> w{63,126,189,252};
	// 252 1 year
	// 252 / 4 = 63 - 3 months
    // 4 * 252 = 1008 use 4 years of data to compute mean, and std dev

	for(unsigned int q = 0;q < w.size();++q){

        cout << endl << "window : " << w[q] << endl;

        std::shared_ptr<ComputeReturn> cr(new ComputeReturn(prices,1,w[q]));

        cr->geometricReturns(prices);
 
        for(size_t p = 0;p < 4;++p){

            cout << endl << pairNames[p] << " VaR" << endl;

            // 1. Riskmetrics
            RiskMetricsVaR var1(.05,.94,false);

            VaRParamCompute<ComputeReturn, RiskMetricsVaR> VaRRiskMetrics(cr, var1);

            cout << "Riskmetrics VaR: " << VaRRiskMetrics.computeVaR(p) << endl;

            std::vector<double> VaRWholePath(VaRRiskMetrics.computeVaRWholePath(p));
        
            // 2. GARCH

            // CHFEUR a0 : 0.08898 		a1 : 0.16399 	b1 : 0.57424
            // DKKEUR a0 : 2.504e-06 	a1 : 8.882e-02 	b1 : 9.070e-01
            // CZKEUR a0 : 1.336e-14 	a1 : 3.473e-01 	b1 : 7.957e-01
            // BRLUSD a0 : 0.2025 		a1 : 0.1120 	b1 : 0.7054

            Vec param1;

            // searching
            std::map<std::string, Vec>::iterator it = GARCHDict.find(pairNames[p]);
            if(it != GARCHDict.end())
                param1 = it->second;
            else
                cout << "GARCH param not found" << endl;

            GarchVaR var2(.05,param1[0],param1[1],param1[2],false);

            VaRParamCompute<ComputeReturn, GarchVaR> VaRGarch(cr, var2);

            cout << "GARCH VaR: " << VaRGarch.computeVaR(p) << endl;

            std::vector<double> VaRWholePath1(VaRGarch.computeVaRWholePath(p));

            // ------------------------------------------------------------

            cr->arithmetricReturns(prices); //switch to arithmetic return

            // 3. Historical method

            HistoricalVaR var3;

            VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical(cr, var3);

            cout << "Historical VaR: " << VaRHistorical.computeVaR(p) << endl;

            std::vector<double> VaRWholePath2(VaRHistorical.computeVaRWholePath(p));

            // 4. Historical method - weighting scheme

            HistoricalVaR var4(.05, .98, hybrid);

            VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical1(cr, var4);

            cout << "Historical VaR w/ weighting scheme: " << VaRHistorical1.computeVaR(p) << endl;

            std::vector<double> VaRWholePath3(VaRHistorical1.computeVaRWholePath(p));

            // 5. Historical method - HW method

            HistoricalVaR var5(.05, .94, hw);

            VaRnoneParamCompute<ComputeReturn, HistoricalVaR> VaRHistorical2(cr, var5);

            cout << "Historical VaR w/ HW weighting scheme: " << VaRHistorical2.computeVaR(p) << endl;

            std::vector<double> VaRWholePath4(VaRHistorical2.computeVaRWholePath(p));

            // 6. Peak over threshold - Generalized Pareto dist

            // CHFEUR u =.145,      sigma/beta = 0.1609310,     xi = 0.4220372
            // DKKEUR u = .0053,    sigma/beta = 0.007703182,   xi = 0.280205110
            // CKZEUR u =.0547,     sigma/beta = 0.19736405,    xi = 0.02903438
            // BRLUSD u = .6705,    sigma/beta = 0.692224368,   xi = 0.006628222

            Vec param2;

            // searching
            std::map<std::string, Vec>::iterator it1 = GPDDict.find(pairNames[p]);
            if(it != GPDDict.end())
                param2 = it1->second;
            else
                cout << "GPD param not found" << endl;

            PoTVaR var6(param2[0],param2[1], param2[2]);

            VaRExtremeValueCompute<ComputeReturn, PoTVaR> VaRPoT(cr, var6);

            cout << "Peak over threshold - gen Pareto VaR: " << VaRPoT.computeVaR(p) << endl;

            std::vector<double> VaRWholePath5(VaRPoT.computeVaRWholePath(p));

            std::string fileName = "/home/mrnoname/Documents/VaR/data/output/var_risk_metrics_" + pairNames[p] + "_" + std::to_string(w[q]) + ".csv";

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

            // ------------------------------------------------------------

            cr->arithmetricReturns(prices); //switch to arithmetic return

            // 3. Historical method

            std::pair<double,double> estimate = VaRHistorical.computeBootstrapVaR(p);

            cout << "Historical VaR - bootstrap avg: " << estimate.first << '\t' << "std dev: "<< estimate.second << endl;

            // 4. Historical method - weighting scheme

            estimate = VaRHistorical1.computeBootstrapVaR(p);

            cout << "Historical VaR w/ weighting scheme - bootstrap avg: " << estimate.first << '\t' << "std dev: "<< estimate.second << endl;

            // 5. Historical method - HW method

            estimate = VaRHistorical2.computeBootstrapVaR(p);

            cout << "Historical VaR w/ HW weighting scheme - bootstrap avg: " << estimate.first << '\t' << "std dev: "<< estimate.second << endl;

            // ------------------------------------------------------------

            cout << endl;

        }

    }

    } catch (const std::exception& e) { // caught by reference to base
        std::cout << " a standard exception was caught, with message '"
                  << e.what() << "'\n";
    }


}
