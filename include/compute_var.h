//compute_var.h

#include<iostream>

#include <boost/math/distributions/students_t.hpp>

#include"var_bridge.h"
#include"compute_returns_eigen.h"
#include"bootstrap.h"
#include"mc_engine.h"

using namespace boost::math;
using namespace std;

#ifndef COMPUTE_VARS_H
#define COMPUTE_VARS_H

//! Compute VaR

template<class T, class U>
class VaRCompute{

public:
	VaRCompute(shared_ptr<T>& _cr, const U& _model)
	:computeReturn(_cr), model(_model){}
	virtual ~VaRCompute(){}

	void setAlpha(double alpha){model.setAlpha(alpha);}
	void setPeriod(unsigned int period){computeReturn->setPeriod(period);}
	
    //!compute VaR using whole path 1x overload
	virtual double computeVaR(size_t p = 0) = 0;

	double computeVaR(double _alpha ,unsigned int _period, unsigned _k, size_t p = 0){

        if(_alpha != model.getAlpha()) setAlpha(_alpha);
        if(_period != computeReturn->getPeriod()) setPeriod(_period);
        if(_k != computeReturn->getWindow()) setAlpha(_k);

        return computeVaR(p);
	}

    virtual std::vector<double> computeVaRWholePath(size_t p = 0) = 0;

protected:
	shared_ptr<T> computeReturn;
	U model;
};

//! Compute VaR using parametric mode

template<class T, class U>
class VaRParamCompute: public VaRCompute<T, U>{

public:
	VaRParamCompute(shared_ptr<T>& _cr, const U& _model):VaRCompute<T, U>(_cr, _model){}
	virtual ~VaRParamCompute(){}

	double computeVaR(size_t p = 0){
      
        std::vector<double> _rtns = VaRCompute<T, U>::computeReturn->getReturns(p);

        unsigned int m(VaRCompute<T, U>::computeReturn->getRollingMean(p).size());

        double _meanReturn = VaRCompute<T, U>::computeReturn->getRollingMean(p)[m -2];

        unsigned int n(_rtns.size());
        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        std::vector<double> rtns;

        for(unsigned int i = 0;i < window;++i)
            rtns.push_back(_rtns[n - window + i]);

        return VaRCompute<T, U>::model(_meanReturn,rtns);

	}

	//! compute VaR over whole path
    std::vector<double> computeVaRWholePath(size_t p = 0){

        std::vector<double> _VaRs;

        std::vector<double> _rtns = VaRCompute<T, U>::computeReturn->getReturns(p);

        std::vector<double> _meanReturn = VaRCompute<T, U>::computeReturn->getRollingMean(p);

        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        std::vector<double> rtns(window);

        unsigned int n(_rtns.size());

        for(unsigned int i = 0;i < n - window;++i){

            for(unsigned int j = 0 ;j < window;++j)
                rtns[j] = _rtns[i  + j]; //i + window + j

                double tmp = VaRCompute<T, U>::model(_meanReturn[i], rtns);

                if(tmp == 0){tmp = _VaRs[i-1];}

            _VaRs.push_back(tmp);
        }

        return _VaRs;

    }


private:

};


//! Compute VaR using none-parametric model

template<class T, class U>
class VaRnoneParamCompute: public VaRCompute<T, U>{

public:
	VaRnoneParamCompute(shared_ptr<T>& _cr, const U& _model):VaRCompute<T, U>(_cr, _model),bs(new Bootstrap<T, U>){}
	virtual ~VaRnoneParamCompute(){}

    double computeVaR(size_t p = 0){

        std::vector<double> _rtns = VaRCompute<T, U>::computeReturn->getReturns(p);

        unsigned int m(VaRCompute<T, U>::computeReturn->getRollingStdDev(p).size());

        double _sigmatminus1 = VaRCompute<T, U>::computeReturn->getRollingStdDev(p)[m -2];

        unsigned int n(_rtns.size());
        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        std::vector<double> rtns;

        for(unsigned int i = 0;i < window;++i)
            rtns.push_back(_rtns[n - window + i]);

        return VaRCompute<T, U>::model(_sigmatminus1,rtns);
    }


    std::vector<double> computeVaRWholePath(size_t p = 0){

        std::vector<double> _VaRs;

        std::vector<double> _rtns = VaRCompute<T, U>::computeReturn->getReturns(p);

        std::vector<double> _sigmatminus1 = VaRCompute<T, U>::computeReturn->getRollingStdDev(p);

        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        std::vector<double> rtns(window);

        unsigned int n(_rtns.size());

        for(unsigned int i = 0;i < n - window;++i){

            for(unsigned int j = 0 ;j < window;++j)
                rtns[j] = _rtns[i  + j]; //i + window + j

            _VaRs.push_back(VaRCompute<T, U>::model(_sigmatminus1[i], rtns));
        }

        return _VaRs;

    }
	//! Compute mean VaR with its standard deviation
	std::pair<double, double> computeBootstrapVaR(size_t p = 0){
        return bs->compute(VaRCompute<T, U>::computeReturn, VaRCompute<T, U>::model,p);

	}

private:
    unique_ptr<Bootstrap<T, U>> bs;
};


//! Compute VaR using extreme value - Generalized Pareto
template<class T, class U>
class VaRExtremeValueCompute: public VaRCompute<T, U>{

public:
	VaRExtremeValueCompute(shared_ptr<T>& _cr, const U& _model):VaRCompute<T, U>(_cr, _model){}
	virtual ~VaRExtremeValueCompute(){}

	double computeVaR(size_t p = 0){
        return VaRCompute<T, U>::model(VaRCompute<T, U>::computeReturn->getReturns(p));
	}

	double computeES(size_t p = 0){
        return VaRCompute<T, U>::model.expectedShortfall(VaRCompute<T, U>::computeReturn->getReturns(p));

	}
	//!compute VaR over whole path
    std::vector<double> computeVaRWholePath(size_t p = 0){

        std::vector<double> _VaRs;

        std::vector<double> _rtns = VaRCompute<T, U>::computeReturn->getReturns(p);

        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        std::vector<double> rtns(window);

        unsigned int n(_rtns.size());

        for(unsigned int i = 0;i < n - window;++i){
            for(unsigned int j = 0 ;j < window;++j)
                rtns[j] = _rtns[i + j];

            _VaRs.push_back(VaRCompute<T, U>::model(rtns));

        }

        return _VaRs;
    }

private:

};

/*! \brief Compute VaR using Monte Carlo

	compute VaR using sim returns and histogram method
*/
template<class T, class U, class V> // T: compute rtn, U: model, V: path
class VaRMonteCarloCompute: public VaRCompute<T, U>{

public:

    typedef MCEngine<rng,V> engine;

	VaRMonteCarloCompute(shared_ptr<T>& _cr, const U& _model, const V& _process, rng& _rng, const double& _n=1.e+05)
	:VaRCompute<T, U>(_cr, _model),n(_n), _rng(rng())
	{
		processes.clear();processes.push_back(_process);
	}

	virtual ~VaRMonteCarloCompute(){}

	void setNumberofSim(double _n){n = _n;}

	//! Compute VaR using Monte Carlo simulation from Gaussian distr.
	double computeVaR(size_t p = 0){

		engine _engine(_rng, processes[p]);

		size_t m(VaRCompute<T, U>::computeReturn->getReturns(p).size() -1);

		Vec rtns{VaRCompute<T, U>::computeReturn->getReturns(p).at(m-1),
					VaRCompute<T, U>::computeReturn->getReturns(p).at(m)};

    	_engine.setValues(rtns, VaRCompute<T, U>::computeReturn->getRollingStdDev(p).back());

		return VaRCompute<T, U>::model(0.,_engine.DoSimulation(p,n,Gaussian));
	}

	//!compute VaR over whole path
    std::vector<double> computeVaRWholePath(size_t p = 0){

		this->setNumberofSim(1.e+04);

		Vec _VaRs;

        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

		Vec rtns = VaRCompute<T, U>::computeReturn->getReturns(p);
		Vec sdevs = VaRCompute<T, U>::computeReturn->getRollingStdDev(p);

        unsigned int q(rtns.size());

		engine _engine(_rng, processes[p]);

        for(unsigned int i = 0;i < q - window;++i){
			Vec _rtns({rtns.at(window + i -1), rtns.at(window + i)});

			_engine.setValues(_rtns,sdevs.at(i));

			_VaRs.push_back(VaRCompute<T, U>::model(0,_engine.DoSimulation(p,n,Gaussian)));
        }
        return _VaRs;
   }

private:

	double n;
	rng _rng;
	std::vector<V> processes;
};

/*! \brief Compute VaR using Monte Carlo & Copula

	compute VaR using sim returns and histogram method
*/
template<class T, class U, class V> // T: compute rtn, U: model, V: path
class VaRCopulaCompute: public VaRCompute<T, U>{

public:

	typedef CopulaEngine<rng,V> engine;

	/*!
		\param _cr compute return class
		\param _model VaR model
		\param _process underlying process
		\param C correlation matrix of standardized residuals
		\param n # of simulation draws
	*/

	VaRCopulaCompute(shared_ptr<T>& _cr, const U& _model, const std::vector<V>& _processes,rng& _rng_,
						const Eigen::MatrixXd& _C, const double& _n=1.e+05)
	:VaRCompute<T, U>(_cr, _model), n(_n) , processes(_processes), C(_C), _rng(_rng_)
	{}

	virtual ~VaRCopulaCompute(){}

	void setCorrelMat(const Eigen::MatrixXd& _C){C = _C;}

	//! Compute VaR using Monte Carlo simulation from either Gaussian or Student's t distr.
	/*!
		\param dof degree of freedom
		\param copulaType Gaussian, Student's t, Clayton, Gumbel
		\param underlyingMar Gaussian, Student's t
		\param a param time t-1. Default 0
		\param b param time t. Default 1

		\result Return vectors of VaRs of dimension d
	*/
	Vec _computeVaR(double dof = 3., double gamma = 1.,
	                copulaType ct = Gauss, underlyingProcess ud = Gaussian,
                    size_t a = 0, size_t b = 1){

		engine _engine(_rng, processes, C);

		Mat Returnt = VaRCompute<T, U>::computeReturn->getReturns();

		size_t m(Returnt.size());

		Mat rtns(m,Vec(2));
		Vec Sigmat;

		size_t _a; size_t _b;

		for(size_t j = 0;j < m;++j){

            size_t w = VaRCompute<T, U>::computeReturn->getRollingStdDev(j).size();

            // Check vect size to avoid alloc error
            a >= w ? _a = w - 2, _b = w - 1 : _a = a, _b = b;

			rtns[j][0] = Returnt[j].at(_a);
			rtns[j][1] = Returnt[j].at(_b);

            Sigmat.push_back(VaRCompute<T, U>::computeReturn->getRollingStdDev(j).at(_a));
		}

		Returnt.clear();

		_engine.setValues(rtns, Sigmat);

		Mat sim = _engine.DoMultiSimulation(n, dof, gamma, ct, ud);

		Vec _VaRs;

		for(size_t j = 0;j < m;j++)
			_VaRs.push_back(VaRCompute<T, U>::model(0.,sim[j]));

		return _VaRs;
	}

	//! Compute VaR using Monte Carlo simulation from either Gaussian or Student's t distr.
	/*!

		\param p procees p. Default 0
		\param dof degree of freedom
		\param copulaType Gaussian, Student's t, Clayton, Gumbel
		\param underlyingMar Gaussian, Student's t
		\param a param time t-1. Default 0
		\param b param time t. Default 1

		\result VaRs for process p
	*/
	double _computeVaRSingle(size_t p =0,
                             double dof = 3., double gamma = 1.,
                             copulaType ct = Gauss, underlyingProcess ud = Gaussian,
                             size_t a = 0, size_t b = 1){

		engine _engine(_rng, processes, C);

		size_t _a; size_t _b;

		size_t w = VaRCompute<T, U>::computeReturn->getRollingStdDev(p).size();

        // Check vect size to avoid alloc error
        a >= w ? _a = w - 2, _b = w - 1 : _a = a, _b = b;

        Vec rtns{VaRCompute<T, U>::computeReturn->getReturns(p).at(_a),
					VaRCompute<T, U>::computeReturn->getReturns(p).at(_b)};

		_engine.setValues(rtns, VaRCompute<T, U>::computeReturn->getRollingStdDev(p).at(_a));

		return (VaRCompute<T, U>::model(0.,_engine.DoSimulation(p, n, dof, gamma, ct, ud)));
	}

	/*! \brief Compute for security, index, etc p

		Assume t-copula with 3 degrees of freedom
	*/
	double computeVaR(size_t p = 0){

        size_t w = VaRCompute<T, U>::computeReturn->getRollingStdDev(p).size();

        return this->_computeVaRSingle(p, 3., 1., t, Student, w-2, w-1);
	}

	/*! \brief Compute VaR over whole path

		Assume t-copula with 3 degrees of freedom. Reduce n to 1e+04
	*/
    std::vector<double> computeVaRWholePath(size_t p = 0){

		n = 1.e+04;

		Vec _VaRs;

        unsigned int window(VaRCompute<T, U>::computeReturn->getWindow());

        unsigned int q(VaRCompute<T, U>::computeReturn->getReturns(p).size());

        for(unsigned int i = 0;i < q - window;++i){
            //cout << i << '\t';
			_VaRs.push_back(this->_computeVaRSingle(p, 3., 1., t, Student, i, i + 1));

        }

        return _VaRs;
   }

private:
    unsigned int n;
    std::vector<V> processes;
    Eigen::MatrixXd C;

	rng _rng;
};

//------------------------------------------------------------------------------

/*! Compute Expected Shortfall aka Conditional VaR (CVaR)

  Compute mean of VaRs starting from alpha
*/
template<class T>
double ExpectedShortfall(T& _var, double alpha = .05, size_t computationPoint = 10, size_t p = 0)
{
	double a(alpha);

	double es(0);

	for(size_t i = 0; i < computationPoint; i++){

		es += _var.computeVaR(p);

		a -= alpha/computationPoint * (i + 1);

		_var.setAlpha(alpha);
	}

	return es/ alpha/ 100.;
};

 #endif //COMPUTE_VARS_H
