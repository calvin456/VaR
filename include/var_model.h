//var_model.h

#include <vector>
#include <map>
#include <iostream>

#include <boost/math/distributions/normal.hpp>

#ifndef VAR_MODELS_H
#define VAR_MODELS_H

using namespace boost::math;
using namespace std;

static boost::math::normal_distribution<> snd(0., 1.);

typedef std::vector<double> Vec;

/*! Abstract base class

 Define different Value at Risk (VaR) models
*/
class VaR{

public:


	VaR(double _alpha = .05);
	virtual ~VaR(){}
	VaR(const VaR& other); //copy constructor
	double getAlpha() const;
	void setAlpha(double _alpha);

	virtual double operator()(double _meanReturn,
							  double _sigmatminus1, double _returnt) const = 0;

	virtual double operator()(double _meanReturn,
							  const Vec& returns) const = 0;

	virtual double operator()(const Vec& returns) const = 0;

protected:

private:
    double alpha;
};

class ParametricVaR : public VaR{

public:
	ParametricVaR(double _alpha = .05, bool _logNormal = false);
	virtual ~ParametricVaR(){}
	ParametricVaR(const ParametricVaR& other);

	virtual double operator()(double _meanReturn,
                              double _sigmatminus1, double _returnt) const = 0;

    virtual double operator()(double _meanReturn,
							  const Vec& returns) const = 0;

	virtual double operator()(const Vec& returns) const = 0;

	double NormalVaR(double u, double sigma) const;
	double LogNormalVaR(double u, double sigma) const;

protected:
	bool logNormal;
private:

};

//! Compute Risk metrics VaR aka Exponential Weight Moving Average (EWMA)
class RiskMetricsVaR : public ParametricVaR {

public:
	RiskMetricsVaR(double _alpha = .05, double _lambda = .94,
                   bool _logNormal = false);
	virtual ~RiskMetricsVaR(){}
	RiskMetricsVaR(const RiskMetricsVaR& other);

	double operator()(double _meanReturn,
                      double _sigmatminus1, double _returnt) const ;

	double operator()(double _meanReturn,
                      const Vec& returns) const;

	double operator()(const Vec& returns) const;


private:
	double lambda;

};

//! Compute GARCH(1,1) VaR
class GarchVaR : public ParametricVaR {

public:
	GarchVaR(double _alpha = .05,
			 double _a = 0., double _b = .74, double _c = .26,
             bool _logNormal = false);
	virtual ~GarchVaR(){}
	GarchVaR(const GarchVaR& other);

	double operator()(double _meanReturn,
                      double _sigmatminus1, double _returntminus1) const ;

	double operator()(double _meanReturn,
                      const Vec& returns) const;

	double operator()(const Vec& returns) const;

private:
	double a;
	double b;
	double c;
};

class NoneParametricVaR: public VaR{

public:
	NoneParametricVaR(double _alpha = .05);
	virtual ~NoneParametricVaR(){}
	NoneParametricVaR(const NoneParametricVaR& other);

	virtual double operator()(const Vec& returns) const {
		cout << "not to be used w/ NoneParametricVaR class" << endl;
		return 1;
	}

	double operator()(double _meanReturn,
                      double _sigmatminus1, double _returnt) const {
		cout << "not to be used w/ NoneParametricVaR class" << endl;
		return 1;
	}

    virtual double operator()(double  _sigmatminus1,
                      		  const Vec& returns) const = 0;
private:

};

enum WeightingScheme {none, hybrid, hw}; //hw Hull-White

//! Compute VaR using historical method (histogram method)
class HistoricalVaR : public NoneParametricVaR {

public:
	HistoricalVaR(double _alpha = .05, double _lambda = .98, WeightingScheme _ws = none);
	virtual ~HistoricalVaR(){}
	HistoricalVaR(const HistoricalVaR& other);

    double operator()(double _sigmatminus1,
                      const Vec& returns) const;

private:
	double lambda;
	double a;
	double b;
	WeightingScheme ws;
};

class ExtremeValueVaR: public VaR{
public:
	ExtremeValueVaR(double _alpha = .05);
	virtual ~ExtremeValueVaR(){}
	ExtremeValueVaR(const ExtremeValueVaR& other);

	virtual double operator()(double _meanReturn,
                      		  double _sigmatminus1, double _returnt) const {
		cout << "not to be used w/ ExtremeValueVaR class" << endl;
		return 1;
	}
	double operator()(double _meanReturn,
                      const Vec& returns) const {
		cout << "not to be used w/ ExtremeValueVaR class" << endl;
		return 1;
	}

	virtual double operator()(const Vec& returns) const = 0;

private:

};

/*! Peak over Threshold (POT)

  Use generalized Pareto distribution
*/
class PoTVaR: public ExtremeValueVaR{

public:
	/*!
	\param u  threshold
	\param beta  scale parameter
	\param xi  \f$\xi\f$ should be > 0
	*/
	PoTVaR(double _u,double _beta, double _xi,double _alpha = .05);
	virtual ~PoTVaR(){}
	PoTVaR(const PoTVaR& other);

	/*! Compute VaR
    \param ratio  # of obs in excess over threshold
	*/
	double operator()(const double& ratio) const;
    //! Compute ratio of exceedance and VaR
	double operator()(const Vec& returns) const;

	double expectedShortfall(const double& ratio) const;
	//! Compute ratio of exceedance and ES
    double expectedShortfall(const Vec& returns) const;

private:
	double xi;
	double beta;
	double u;
};

#endif //VAR_MODELS_H

