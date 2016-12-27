//portfolio.cpp

#include<algorithm>
#include<utility>

#include "portfolio.h"

Portfolio::Portfolio(const Ptf& _ptf,
                     const Vec& _weights,
                     shared_ptr<ComputeReturn>& _cr,
                     bool _isFI,
                     double _grossNotional)
:ptf(_ptf),weights(_weights),cr(_cr),isFI(_isFI), grossNotional(_grossNotional)
{
	ptfRtns.clear(); ComponentPtfRtns.clear();

	nbAssets = _ptf.size();

	ComputeComponentPtfRtn();
	computeRtn(nbAssets);
	computeMeanPtfRtn();
	computerPtfSdev();
	computeVarCov();
}

Vec Portfolio::getComponentReturns(size_t p) const {return ComponentPtfRtns[p];}

void Portfolio::computeRtn(unsigned int nbAssets){

	//rescale
	Vec _weights; for(size_t i=0;i < nbAssets;++i) _weights.push_back(weights[i]);

	double sumWeight(0);for (auto& n : _weights)  sumWeight += n;

	transform(_weights.begin(),_weights.end(),_weights.begin(), std::bind1st(std::multiplies<double>(),1./sumWeight));

	for(unsigned int i = 0;i < ComponentPtfRtns[0].size();++i){

		double tmp(0);

		for(unsigned int j = 0;j < nbAssets;++j)
			tmp += _weights[j] * ComponentPtfRtns[j][i];

		ptfRtns.push_back(tmp);
	}
}


Vec Portfolio::getRollingMean(size_t p) const{

	Vec RollingMeanReturns;

	unsigned int k = cr->getWindow();

	unsigned int n = ptfRtns.size();

	for(unsigned int i = 0;i < n - k;++i){

		Vec _assetReturns;
		for(size_t j =0;j < i + k;++j)  _assetReturns.push_back(ptfRtns[j]);

		RollingMeanReturns.push_back(computeRiskReturn(_assetReturns)[0]);
	}
	return RollingMeanReturns;
}

Vec Portfolio::getRollingStdDev(size_t p) const{

	Vec RollingSigmastminus1;

	unsigned int k = cr->getWindow();

	unsigned int n = ptfRtns.size();

	for(unsigned int i = 0;i < n - k;++i){

		Vec _assetReturns;
		for(size_t j =0;j < i + k;++j)  _assetReturns.push_back(ptfRtns[j]);

		RollingSigmastminus1.push_back(sqrt(computeRiskReturn(_assetReturns)[1]));
	}
	return RollingSigmastminus1;
}

Vec Portfolio::getComputeReturnStdDev(size_t p) const{
    Vec RollingSigmastminus1;

	unsigned int k = cr->getWindow();

	Vec rtns(this->getComponentReturns(p));

	unsigned int n = rtns.size();

	for(unsigned int i = 0;i < n - k;++i){

		Vec _assetReturns;
		for(size_t j =i;j < i + k;++j)  _assetReturns.push_back(rtns[j]);

		RollingSigmastminus1.push_back(sqrt(computeRiskReturn(_assetReturns)[1]));
	}
	return RollingSigmastminus1;

}

void Portfolio::setWeight(const Vec& _weights){

	weights = _weights;
	computeRtn(nbAssets);
	computeMeanPtfRtn();
	computerPtfSdev();
	computeVarCov();
}

void Portfolio::setReturns(const Mat& _mAssetReturns){

	cr->setReturns(_mAssetReturns);
}


