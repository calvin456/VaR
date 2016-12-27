//portfolio.h

#include <tuple>
#include<memory>
#include<vector>
#include <algorithm>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "compute_returns_eigen.h"
#include "instrument.h"

using namespace boost::accumulators;
using namespace std;

#ifndef PORTFOLIO_H
#define PORTFOLIO_H

typedef std::vector<pair<unsigned int, shared_ptr<Instrument>>> Ptf;

class Portfolio{
    friend class VaRPtfCompute;

public:
	Portfolio(const Ptf& _ptf,
		  	  const Vec& _weights,
		  	  shared_ptr<ComputeReturn>& _cr,
		  	  bool _isFI = false,
		  	  double _grossNotional = 1.);

	inline Vec getReturns(size_t p = 0) const;
	Vec getComponentReturns(size_t p) const;

	inline Eigen::MatrixXd getVarCov() const;

	inline double getMeanPtfRn() const;
	inline double getPtfSdev() const;
	inline double getnbAssets() const;
	inline double getGrossAmt() const;
	inline unsigned int getPeriod() const;
	inline unsigned int getWindow() const;

	inline bool isFixedIncome() const;

	inline Ptf getPositions() const;

	void computeRtn(unsigned int nbAssets);

	//! Compute rolling std dev for component return p
	Vec getComputeReturnStdDev(size_t p = 0) const;

	//! Compute rolling mean rtnaccording to period ie day, week, month and window
	Vec getRollingMean(size_t p = 0) const;
	//! // Compute rolling std dev according to period ie day, week, month and window
	Vec getRollingStdDev(size_t p = 0) const;

	Vec getWeight() const;
	void setWeight(const Vec& _weights);

	void setReturns(const Mat& _mAssetReturns);

protected:

	//! w' * R - Need to account for approx of asset return : delta, gamma, etc
	void computeMeanPtfRtn(){computeRiskReturn(ptfRtns)[0];};

	//! w' * sigma * w - Need to account for approx of asset return : delta, gamma, etc
	void computerPtfSdev(){sqrt(computeRiskReturn(ptfRtns)[1]);};

	//! \result component ptf return
	void ComputeComponentPtfRtn(){

		Mat _rtns = cr->getReturns();

		ComponentPtfRtns.resize(nbAssets,Vec(0));

		for(unsigned int i = 0;i < nbAssets;++i){
			for(unsigned int j = 0;j < _rtns[i].size();++j){
                double tmp;
				tmp = _rtns[i][j];
				if(!isFI) {
                    tmp = ptf[i].second->operator()(_rtns[i][j]);
				}
				else {
                    tmp = ptf[i].second->operator()(_rtns[ptf[i].first][j]);
				}

				ComponentPtfRtns[i].push_back(tmp);
			}
		}

		ComponentPtfRtns.resize(nbAssets,Vec(ComponentPtfRtns[0].size()));

	};

	void computeVarCov(){

		Vec mMeanReturns;

		//1. average rtn
		for(size_t i = 0;i < nbAssets;++i){

			accumulator_set<double, features<tag::mean> > acc;
			acc = std::for_each(ComponentPtfRtns[i].begin(), ComponentPtfRtns[i].end(), acc);

			mMeanReturns.push_back(boost::accumulators::mean(acc)); //boost mean
		}

		//2. compute covariance

		VarCov.resize(nbAssets, nbAssets);

		for(size_t i = 0;i <nbAssets;++i){
			for(size_t j = 0;j <nbAssets;++j){

				Vec _tmp;

				for(size_t p = 0;p < ComponentPtfRtns[0].size();++p)
					_tmp.push_back((ComponentPtfRtns[i][p] - mMeanReturns[i]) *
										(ComponentPtfRtns[j][p] - mMeanReturns[j]));

				accumulator_set<double, features<tag::mean> > acc;
				acc = std::for_each(_tmp.begin(), _tmp.end(), acc);

				VarCov(i,j) = boost::accumulators::mean(acc); //boost mean

			}
		}

	};


private:
	Ptf ptf;
	Vec weights;
	shared_ptr<ComputeReturn> cr;
	double ptfMeanRtn;
	double ptfSdev;
	double nbAssets;

	Vec ptfRtns;
	Mat ComponentPtfRtns;
	Eigen::MatrixXd VarCov;

	bool isFI; //use if FI ptf and key rate duration
	double grossNotional;
};

inline Vec Portfolio::getReturns(size_t p) const{return ptfRtns;}

inline Eigen::MatrixXd Portfolio::getVarCov() const{return VarCov;}

inline double Portfolio::getMeanPtfRn() const{return ptfMeanRtn;}
inline double Portfolio::getPtfSdev() const{return ptfSdev;}
inline double Portfolio::getnbAssets() const{return nbAssets;}

inline Ptf Portfolio::getPositions() const{return ptf;}

inline Vec Portfolio::getWeight() const{return weights;}

inline double Portfolio::getGrossAmt() const {return grossNotional;}

inline unsigned int Portfolio::getPeriod() const {return cr->getPeriod();}
inline unsigned int Portfolio::getWindow() const {return cr->getWindow();}

inline bool Portfolio::isFixedIncome() const {return isFI;}

#endif //PORTFOLIO_H




