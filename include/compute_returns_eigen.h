//compute_returns_eigen.h

#include <vector>
#include <math.h>
#include <memory>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#ifndef COMPUTE_RTNS_H
#define COMPUTE_RTNS_H

typedef std::vector<double> Vec;
typedef std::vector<Vec> Mat;

Vec computeRiskReturn(const Vec& assetReturns); // func declaration

/*!Compute returns, mean, std dev

   Assume daily return	
*/ 
class ComputeReturn{

public:
	// ComputeReturn();

	/*! constructor 2X overloads
		\param period one day. 252 trading days (US trading calendar)
		\param k rolling period. 1 year
	*/
	ComputeReturn(unsigned int _period = 1,
				  unsigned int _k = 252);

	ComputeReturn(const Vec& _prices,
				  unsigned int _period = 1,
				  unsigned int _k = 252,
                  bool _logRtns = false);

	ComputeReturn(const Mat& _prices,
				  unsigned int _period = 1,
				  unsigned int _k = 252,
                  bool _logRtns = false);
    //! copy constructor implementation
	ComputeReturn(const ComputeReturn& other); //= default; //copy constructor
	~ComputeReturn(){}

	//! compute arithmetic rtns
	void arithmetricReturns(const Vec& _prices);
	//! compute geometric rtns ie. log return
	void geometricReturns(const Vec& _prices);

	void arithmetricReturns(const Mat& _prices);
	void geometricReturns(const Mat& _prices);

	/*! \brief Compute adjusted return \f$ \hat{R}= \hat{A} A^{-1} R\f$

		\param Chat = Bumped original correl matrix for several asset pairs

     	C, Chat correlation matrices

     	Use Cholesky decomposition \f$X = LL^T\f$

     	ie. 

		\f$C = AA^T\f$

     	\f$ \hat{C} = \hat{A} \hat{A} ^{T}\f$
    */
	void correlReweightedRtns(const Eigen::MatrixXd& Chat);

	/*! \brief Compute PCA Principal Component Analysis
		
	Principal Component up to 99%

	    
  	*/
	Eigen::MatrixXd computePC();

	//! Compute rolling mean rtn, and std dev according to period ie day, week, month and window
	Vec getRollingMean(size_t p) const;
	//! Compute rolling mean rtn, and std dev according to period ie day, week, month and window
	Vec getRollingStdDev(size_t p) const;

	inline unsigned int getPeriod() const;
	inline unsigned int getWindow() const;

	Vec getReturns(size_t p) const;
	inline Mat getReturns() const;

	double getMeanReturn(size_t p = 0) const;
	inline Vec getMeanReturns() const;

	double getStdDev(size_t p = 0) const;

	inline Eigen::MatrixXd getVarCov() const;
	Eigen::MatrixXd getCorrelMat();

	void setPeriod(unsigned int _period);
	void setWindow(unsigned int _k);

	void setReturns(const Mat& _mAssetReturns);

protected:
	//! compute arithmetic rtns
	void _arithmetricReturns(size_t j = 0){


        mAssetReturns[j].clear();

        size_t n(mPrices[j].size());


		for(size_t i = 0;i < n - period;++i)
			mAssetReturns[j].push_back((mPrices[j][i + period]/mPrices[j][i] - 1.)*100.);

	};

	//! compute geometric rtns ie. log return
	void _geometricReturns(size_t j = 0){


        mAssetReturns[j].clear();

        size_t n(mPrices[j].size());


		for(size_t i = 0;i < n - period;++i)
			mAssetReturns[j].push_back(log(mPrices[j][i + period]/mPrices[j][i]) * 100.);

	};

private:
	unsigned int period;
	unsigned int k;
	bool logRtns;
    Mat mPrices;
    Mat mAssetReturns;
    Vec mMeanReturns;
    Eigen::MatrixXd VarCov;

};

inline unsigned int ComputeReturn::getPeriod() const {return period;}
inline unsigned int ComputeReturn::getWindow() const {return k;}

inline Mat ComputeReturn::getReturns() const {return mAssetReturns;}
inline Eigen::MatrixXd ComputeReturn::getVarCov() const {return VarCov;}

#endif //COMPUTE_RTNS_H
