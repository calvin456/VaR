//ptf_var.cpp

#include <algorithm>
#include <numeric>
#include <math.h>

#include "ptf_var.h"
#include "var_model.h"
#include "mc_engine.h"
#include "pca.h"

using namespace std;

//!slpe of a linear regression
double slope(const std::vector<double>& x, const std::vector<double>& y) {
    const auto n    = x.size();
    const auto s_x  = std::accumulate(x.begin(), x.end(), 0.0);
    const auto s_y  = std::accumulate(y.begin(), y.end(), 0.0);
    const auto s_xx = std::inner_product(x.begin(), x.end(), x.begin(), 0.0);
    const auto s_xy = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
    const auto a    = (n * s_xy - s_x * s_y) / (n * s_xx - s_x * s_x);
    return a;
}

VaRPtfCompute::VaRPtfCompute(const shared_ptr<Portfolio>& _ptf, double _alpha):ptf(_ptf),alpha(_alpha){

	Vec ptfReturn = ptf->getReturns();

	beta.clear();

	for(size_t i =0;i < ptf->getnbAssets();++i)
		beta.push_back(slope(ptfReturn,ptf->getComponentReturns(i)));

	nbAssets = ptf->getnbAssets();
}

VaRPtfCompute::VaRPtfCompute(const VaRPtfCompute& other):
    ptf(other.ptf),
	alpha(other.alpha),
	beta(other.beta),
	nbAssets(other.nbAssets)
{}


double VaRPtfCompute::getPtfVaR() const{

	double grossAmount = ptf->getGrossAmt();

	Vec weights = ptf->getWeight();

	Vec investAmount(nbAssets);

	transform(weights.begin(), weights.end(), investAmount.begin(), std::bind1st(std::multiplies<double>(),grossAmount));

	Eigen::MatrixXd Cov = ptf->getVarCov();

	double dollarPtfSdev(0.);

	for(size_t i =0;i < nbAssets;++i)
		for(size_t j =0;j < nbAssets;++j)
			dollarPtfSdev += investAmount[i] * investAmount[j] * Cov(i,j);

	RiskMetricsVaR _VaR(alpha);

	return _VaR.NormalVaR(0.,sqrt(dollarPtfSdev)/100.);
}

Vec VaRPtfCompute::computeMarginalVaR() const{

	Vec margVaR(nbAssets);

	double percPtfVar = getPtfVaR()/ptf->getGrossAmt();

	transform(beta.begin(), beta.end(), margVaR.begin(), std::bind1st(std::multiplies<double>(), percPtfVar/100.));

	return margVaR;
}

Vec VaRPtfCompute::computeComponentVaR() const{

	Vec componentVaR(nbAssets);
	Vec weights = ptf->getWeight();

	double PtfVar = this->getPtfVaR();

	for(unsigned int i = 0;i < weights.size();++i)
		componentVaR[i] = PtfVar * beta[i]  * weights[i];

	return componentVaR;
}


double VaRPtfCompute::computeIncrementalVaR(double amount) const{

	Vec marginalVaR = computeMarginalVaR();

	double incrementalVaR(0.);

	for(unsigned int i = 0;i < static_cast<unsigned int>(nbAssets);++i)
        incrementalVaR += marginalVaR[i];

	return incrementalVaR * amount;
}

Vec VaRPtfCompute::computeIndividualVaR() const{

	double grossAmount = ptf->getGrossAmt();

	Vec weights = ptf->getWeight();

	Vec investAmount(nbAssets);

	transform(weights.begin(), weights.end(), investAmount.begin(), std::bind1st(std::multiplies<double>(),grossAmount));

	Eigen::MatrixXd Cov = ptf->getVarCov();

	RiskMetricsVaR _VaR(alpha);

	Vec VaRs;

	for(size_t i =0;i < nbAssets;++i)
		 VaRs.push_back(investAmount[i] * _VaR.NormalVaR(0.,sqrt(Cov(i,i))/100.));

	return VaRs;
}
