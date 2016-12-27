//path.h

#include<iostream>
#include<vector>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

#ifndef PATH_H
#define PATH_H

typedef std::vector<double> Vec;
typedef std::vector<std::vector<double>> Mat;

//! Path fait tail, mean-reverting, commodity process ...
class Path{

public:
	Path(){}
	//Path(const Path& other);
	virtual ~Path(){}

	virtual double doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1) const =0;

	virtual Eigen::VectorXd doSim(const Vec& u,
                                    const Vec&  Sigmat,const Vec& Returnt) const=0;

private:
};

class Path1x1: public Path{

public:
	Path1x1();
	virtual ~Path1x1(){}

	virtual double doOneSim(const double& u,
                                const double& Sigmat=0., const double& Returnt=0., const double& Returntminus1=0.) const;

	virtual Eigen::VectorXd doSim(const Vec& u,
                                    const Vec&  Sigmat= Vec(), const Vec& Returnt = Vec()) const;

private:
};

//! model AR(1) x GARCH(1,1) - Fat tails
class AR1xGARCH11 : public Path{


public:
    /*!
        \param _mu average rtn
        \param _correl  \f$\phi\f$ acf previous period
        \param _alpha \f$\alpha\f$ GARCH process
        \param _beta \f$\beta\f$ GARCH process
        \param _gamma \f$\gamma\f$ GARCH process
	*/
	AR1xGARCH11( double _mu=0., double _correl = 0.5,
				double _alpha=0., double _beta=.24, double _gamma=.76);

	virtual ~AR1xGARCH11(){}

	/*! Simulate return for next period under review
        \param Returntminus1 \f$r_{t-1}\f$ return previous period
        \param Sigmatminus1 \f$\sigma_{t-1}\f$ vol previous period
        \param u innovation drawn sample distr.
        \param v innovation drawn sample distr.
	*/

    virtual double doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1) const;

	virtual Eigen::VectorXd doSim(const Vec& u,
                                    const Vec&  Sigmat,const Vec& Returnt) const;

private:
	double mu;
	double correl;
	double alpha;
	double beta;
	double gamma;
};

//! model GARCH(1,1)
class GARCH11 : public Path{


public:
    /*!
        \param _alpha \f$\alpha\f$ alpha GARCH process
        \param _beta \f$\beta\f$ beta GARCH process
        \param _gamma \f$\gamma\f$ gamma GARCH process
	*/
	GARCH11(double _alpha=0., double _beta=.24, double _gamma=.76);

	/*! Simulate return for next period under review
        \param Returntminus1 \f$r_{t-1}\f$ return previous period
        \param Sigmatminus1 \f$\sigma_{t-1}\f$ vol previous period
        \param u innovation drawn sample distr.
        \param v innovation drawn sample distr.
	*/
	virtual double doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1=0.) const;

    //! Simultate return over whole path
	virtual Eigen::VectorXd doSim(const Vec& u,
                                    const Vec&  Sigmat, const Vec& Returnt) const;

private:
	double mu;
	double correl;
	double alpha;
	double beta;
	double gamma;
};


#endif //PATH_H
