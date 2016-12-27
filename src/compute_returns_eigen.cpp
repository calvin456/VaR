//compute_returns.cpp

#include<string>

#include <algorithm>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "pca.h"
#include "compute_returns_eigen.h"

using namespace boost::accumulators;

Vec computeRiskReturn(const Vec& assetReturns) {
	accumulator_set<double, features<tag::mean, tag::variance> > acc;
	acc = std::for_each(assetReturns.begin(), assetReturns.end(), acc);

	Vec tmp;
	tmp.push_back(boost::accumulators::mean(acc)); //boost mean
	tmp.push_back(boost::accumulators::variance(acc)); //boost sigma

	return tmp;
}

ComputeReturn::ComputeReturn(unsigned int _period, unsigned int _k)
:period(_period),k(_k),logRtns(false)
{
   	mPrices.clear(), mAssetReturns.clear(), mMeanReturns.clear();
   	mPrices.resize(1,Vec(1)), mAssetReturns.resize(1,Vec(1)), VarCov.resize(0,0);
}

ComputeReturn::ComputeReturn(const Vec& _prices,
				  	  	  	 unsigned int _period,
				  	  	  	 unsigned int _k,
							 bool _logRtns)
:period(_period),k(_k),logRtns(_logRtns)
{
	mPrices.clear(), mAssetReturns.clear(), mMeanReturns.clear();

	size_t n(_prices.size());

	mPrices.resize(1, Vec(n));
	mPrices[0] = _prices;

	//Compute rtns
	mAssetReturns.resize(1, Vec(n));
	logRtns ? this->_geometricReturns() : this->_arithmetricReturns();

	//Compute average return and volatility
	Vec tmp = computeRiskReturn(mAssetReturns[0]);

	mMeanReturns.push_back(tmp[0]);

	VarCov.resize(1,1);

    VarCov(0,0) = tmp[1];
}

ComputeReturn::ComputeReturn(const Mat& _prices,
				  			 unsigned int _period,
				  			 unsigned int _k ,
                  			 bool _logRtns)
:period(_period),k(_k),logRtns(_logRtns)
{
	mPrices.clear(), mAssetReturns.clear(), mMeanReturns.clear();

	size_t m(_prices.size());
	size_t n(_prices[0].size());

	mPrices.resize(m, Vec(n));
	mPrices = _prices;

    mAssetReturns.resize(m, Vec(n));
	//Compute rtns
	for(size_t i = 0;i < m;++i)
		logRtns ? _geometricReturns(i) : _arithmetricReturns(i);

	//1. average rtn
	for(size_t i = 0;i < m;++i){

		accumulator_set<double, features<tag::mean> > acc;
		acc = std::for_each(mAssetReturns[i].begin(), mAssetReturns[i].end(), acc);

		mMeanReturns.push_back(boost::accumulators::mean(acc)); //boost mean
	}

	//Compute Var-Cov matrix

	VarCov.resize(m,m);

	for(size_t i = 0;i <m;++i){
		for(size_t j = 0;j <m;++j){

			//2. compute covariance
			Vec _tmp;

			double q = mAssetReturns[i].size() < mAssetReturns[j].size() ? mAssetReturns[i].size() : mAssetReturns[j].size();

			for(size_t p = 0;p < q;++p)
				_tmp.push_back((mAssetReturns[i][p] - mMeanReturns[i]) * (mAssetReturns[j][p] - mMeanReturns[j]));

			accumulator_set<double, features<tag::mean> > acc;
			acc = std::for_each(_tmp.begin(), _tmp.end(), acc);

			VarCov(i,j) = boost::accumulators::mean(acc); //boost mean

		}
	}
}


ComputeReturn::ComputeReturn(const ComputeReturn& other):

    period(other.period), k(other.k), logRtns(other.logRtns),
	mPrices(other.mPrices), mAssetReturns(other.mAssetReturns),
	mMeanReturns(other.mMeanReturns), VarCov(other.VarCov)
{}

void ComputeReturn::arithmetricReturns(const Vec& _prices)
{
    mPrices.resize(1, Vec(_prices.size()));

	mPrices[0] = _prices;
	_arithmetricReturns();
}

void ComputeReturn::geometricReturns(const Vec& _prices)
{
    mPrices.resize(1, Vec(_prices.size()));

	mPrices[0] = _prices;
	_geometricReturns();
}

void ComputeReturn::arithmetricReturns(const Mat& _prices){

    mPrices = _prices;

    for(size_t i = 0;i < mPrices.size();++i)
        _arithmetricReturns(i);
}

void ComputeReturn::geometricReturns(const Mat& _prices){

    mPrices = _prices;

    for(size_t i = 0;i < mPrices.size();++i)
		 _geometricReturns(i);
}

void ComputeReturn::setPeriod(unsigned int _period){

	period = _period;

	for(size_t i = 0;i < mPrices.size();++i){
		logRtns ? _geometricReturns(i) : _arithmetricReturns(i);
	}
}

void ComputeReturn::setWindow(unsigned int _k){

	k = _k;

	for(size_t i = 0;i < mPrices.size();++i){
		logRtns ? _geometricReturns(i) : _arithmetricReturns(i);
	}
}

Vec ComputeReturn::getReturns(size_t p) const {return mAssetReturns[p];}

double ComputeReturn::getMeanReturn(size_t p) const {return mMeanReturns[p];}

double ComputeReturn::getStdDev(size_t p) const {return sqrt(VarCov(p,p));}


Vec ComputeReturn::getRollingMean(size_t p) const {

	Vec RollingMeanReturns;

	for(unsigned int i = 0;i < mAssetReturns[p].size() - k;++i){

        Vec _assetReturns;

        for(unsigned int j = i; j < i + k; ++j)
            _assetReturns.push_back(mAssetReturns[p][j]);

		RollingMeanReturns.push_back(computeRiskReturn(_assetReturns)[0]);
	}

	return RollingMeanReturns;
}

Vec ComputeReturn::getRollingStdDev(size_t p) const {

	Vec RollingSigmastminus1;

	for(unsigned int i = 0;i < mAssetReturns[p].size() - k;++i){

    	Vec _assetReturns;

        for(unsigned int j = i; j < i + k; ++j)
        	_assetReturns.push_back(mAssetReturns[p][j]);

		RollingSigmastminus1.push_back(sqrt(computeRiskReturn(_assetReturns)[1]));
	}

	return RollingSigmastminus1;
}


Eigen::MatrixXd ComputeReturn::getCorrelMat(){

	size_t m = VarCov.cols();

	Eigen::MatrixXd CorrelMat(m,m);

    for(size_t i = 0;i < m;++i)
        for(size_t j = 0;j < m;++j)
            CorrelMat(i,j) = VarCov(i,j)/sqrt(VarCov(i,i) * VarCov(j,j));

    return CorrelMat;
}

void ComputeReturn::setReturns(const Mat& _mAssetReturns){

	mAssetReturns = _mAssetReturns;
}

void ComputeReturn::correlReweightedRtns(const Eigen::MatrixXd& Chat){

	Eigen::MatrixXd C = this->getCorrelMat();

    Eigen::MatrixXd A( C.llt().matrixL() );

    Eigen::MatrixXd Ahat( Chat.llt().matrixL() );

	size_t m = mAssetReturns.size();
	size_t n = mAssetReturns[0].size();

    //Fill correl matrix to Eigen matrix
    MatrixXd R(m,n);
    for(size_t i = 0;i <m;++i)
        for(size_t j = 0;j <n;++j)
            R(i,j) = mAssetReturns[i][j];

    MatrixXd Rhat(n,m);
    Rhat = Ahat * A.inverse() * R;

	//Back to std::vector
    for(size_t i = 0;i <m;++i)
        for(size_t j = 0;j <n;++j)
             mAssetReturns[i][j] = Rhat(i,j);

}

Eigen::MatrixXd ComputeReturn::computePC(){

    // convert to req format
    vector<float> vec;

	size_t n = VarCov.rows();
	size_t m = VarCov.cols();

    for(size_t i = 0;i < n;++i)
        for(size_t j = 0;j < m;++j)
            vec.push_back(VarCov(i,j));

	std::shared_ptr<Pca> pca(new Pca());

  	int init_result = pca->Calculate(vec, n, m);//, true, true, false);
  	if(init_result == 1) cout << "correl matrix not positive definite" << endl;

    vector<float> scores = pca->scores(); //Rotated data

  	unsigned int kaiser = pca->kaiser(); //Kaiser criterion 99%

    unsigned int nrows = pca->nrows();

	Eigen::MatrixXd PC(nrows, kaiser);

	for(size_t i = 0;i < nrows;++i)
		for(size_t j = 0;j < kaiser;++j)
			PC(i,j) = scores[j + kaiser*i];

	return PC;
}









