//mc_engine.h

#include<vector>
#include<memory>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <Eigen/Dense>

#include "path.h"
#include "generate_rm_from_distr.h"

using namespace boost::math;
using namespace Eigen;
using namespace std;

#ifndef MC_ENGINE_H
#define MC_ENGINE_H

enum matType{Cholesky, pc}; //!pc : principal component
enum underlyingProcess{Gaussian, Student}; //!Student's t-distr

//! Simulate return through Monte-Carlo

template<class T, class U> //!T rng, U path
class MCEngine{

public:

    typedef  std::vector<U> paths;

    //! Constructor 2x overloads
	MCEngine(T& _rng, const U& _path)
	:innerGen(_rng)
	{
        Returnstminus1.clear();sigmastminus1.clear();

        paths_.clear();
		paths_.push_back(_path);

		A.resize(0,0);
		m = 1;
	};
	//! multiple assets
	MCEngine(T& _rng, const paths& _paths, const Eigen::MatrixXd& _A, matType _mt)
	:mt(_mt), innerGen(_rng)
	{
		Returnstminus1.clear();sigmastminus1.clear();

		paths_.clear();
		for(size_t i=0;i < _paths.size();++i)
            paths_.push_back(_paths[i]);

		mt == Cholesky ? m = _A.rows() : m = _A.cols(); 

		A.resize(_A.rows(),_A.cols());
		A = _A;
	};


	//! multiple assets w/o correl matrix
	MCEngine(T& _rng, const paths& _paths)
	:innerGen(_rng)
	{
		Returnstminus1.clear();sigmastminus1.clear();

		paths_.clear();

        m = _paths.size();

		for(auto &i : _paths)	paths_.push_back(i);

		A.resize(0,0);
	};


	MCEngine(const MCEngine& other):
        mt(other.mt),
        innerGen(other.innerGen),
        A(other.A),
        paths_(other.paths_),
        m(other.m),
        Returnstminus1(other.Returnstminus1),
        sigmastminus1(other.sigmastminus1)
	{};

	virtual ~MCEngine(){};

	void setValues(const Vec& rtns, const double vol){
        sigmastminus1.clear();sigmastminus1.push_back(vol);

		Returnstminus1.resize(1,Vec(rtns.size()));
		for(size_t i=0;i < rtns.size();++i)
            Returnstminus1[0][i] = rtns[i];

    };

    void setValues(const Mat& rtns, const Vec& vol){

        Returnstminus1.resize(rtns.size(),Vec()); Returnstminus1 = rtns;
        sigmastminus1.resize(vol.size()); sigmastminus1 = vol;

    };

	Vec DoSimulation(size_t p = 0,double n = 1.e+05,underlyingProcess up = Gaussian){

		std::shared_ptr<GenFromDistr<T>> _rng(new GenFromDistr<T>(innerGen));

		Vec simRtns(n);

		double u(0.);

		// single case
		if(m == 1){
			for(size_t i = 0;i < n;++i){
				up == Gaussian ? u = _rng->getOneGaussian() : u = _rng->getOneStudent(n);

                simRtns[i] = paths_[p].doOneSim(u, sigmastminus1[p], Returnstminus1[p][1], Returnstminus1[p][0]);
			}
		}
		else cout << "single asset case" << endl;

		return simRtns;
	};

    /*! multiple indep rtn with Cholesky decomposition to produce correl rtns

        mutliple indep rtn with principal component
	*/
	Mat DoMultiSimulation(double n = 1.e+05,underlyingProcess up = Gaussian){

        std::shared_ptr<GenFromDistr<T>> _rng(new GenFromDistr<T>(innerGen));

		Mat simRtns(A.rows(), Vec(n));

		Eigen::VectorXd u(m);Eigen::VectorXd v(m);

		// multiple asset case
		if(m != 1){
            for(size_t i = 0;i < n;++i){

                up == Gaussian ? u = _rng->getXGaussian(m) : u = _rng->getXStudent(n,m);

                Eigen::VectorXd w(A.rows());
                mt == Cholesky ? w = u.transpose() * A : u.transpose() * A.transpose();

                for(unsigned int j = 0;j < A.rows();++j)
                    simRtns[j][i] = paths_[j].doOneSim(w(j),  sigmastminus1[j], Returnstminus1[j][1], Returnstminus1[j][0]);

            }
		}
		else cout << endl << "Needs at least two assets" << endl;

		return simRtns;
	};

protected:
    matType mt;
	T innerGen; //rng, copula rng
	Eigen::MatrixXd A; //Cholesy, PC
	paths paths_;
	size_t m; // # of assets to simul
	Mat Returnstminus1;
	Vec sigmastminus1;
};

template<class T, class U> //!T rng, U path
class CopulaEngine :public MCEngine<T, U>{

public:

    typedef  std::vector<U> paths;

	/*! Constructor 1x overload

		multiple assets

		\param _rng pseudo random number generator
		\param _paths paths vector
		\param _C correlation matrix
	*/
	CopulaEngine(T& _rng_, const paths& _paths, const Eigen::MatrixXd _C)
	:MCEngine<T, U>( _rng_, _paths),C(_C)
	{
		_rng = shared_ptr<copulaRng>(new copulaRng(_paths.size(), _rng_));
	};


	CopulaEngine(T& _rng_, const paths& _paths)
	:MCEngine<T, U>( _rng, _paths)
	{
		C.resize(0,0);
		_rng = shared_ptr<copulaRng>(new copulaRng(_paths.size(), _rng_));
	};

	CopulaEngine(const CopulaEngine& other);
	virtual ~CopulaEngine(){}

	void setCorrelMat(const Eigen::MatrixXd& _C){C = _C;}

	Vec DoSimulation(size_t p =0,double n = 1.e+05,
                     	double dof = 3., double gamma = 1.,
                     	copulaType ct = Gauss, underlyingProcess ud = Gaussian
					 ){

		// underlying marginals
		boost::math::students_t t(dof);
		boost::math::normal_distribution<> snd(0., 1.);

		Vec simRtns(n);

		for(unsigned int i=0;i < n;++i){

			// Generate vector of d-dimensional copula
			Vec q = _generateDraws(ct, dof, gamma);

            // Generate standardized residuals on filtered residuals
            if(q[p] == 1.) q[p] -= .01;
            double StdResiduals = (ud != Gaussian ? boost::math::quantile(t,q[p]) : boost::math::quantile(snd,q[p]));

            // unstandardized residuals + filtered part of process
			simRtns[i] = MCEngine<T, U>::paths_[p].doOneSim(StdResiduals ,MCEngine<T, U>::sigmastminus1[0],
                                            MCEngine<T, U>::Returnstminus1[0][1], MCEngine<T, U>::Returnstminus1[0][0]);
     	}

		return simRtns;
	};

    /*! multiple indep rtn with Cholesky decomposition to produce correl rtns

        mutliple indep rtn with principal component
	*/
	Mat DoMultiSimulation(double n = 1.e+05,double dof = 3., double gamma = 1.,
                     		copulaType ct = Gauss, underlyingProcess ud = Gaussian
						){

        // underlying marginals
		boost::math::students_t t(dof);
		boost::math::normal_distribution<> snd(0., 1.);

		size_t _m = MCEngine<T, U>::m;

        Mat simRtns(_m,Vec(n));

		for(unsigned int i=0;i < n;++i){

			// Generate vector of d-dimensional copula
			Vec q = _generateDraws(ct, dof, gamma);

			for(unsigned int j = 0;j < _m;j++){

				// Generate standardized residuals on filtered residuals
				if(q[j] == 1.) q[j] -= .01;
				double StdResiduals = (ud != Gaussian ? boost::math::quantile(t,q[j]) : boost::math::quantile(snd,q[j]));

				simRtns[j][i] = MCEngine<T, U>::paths_[j].doOneSim(StdResiduals, MCEngine<T, U>::sigmastminus1[j],
                                                    MCEngine<T, U>::Returnstminus1[j][1], MCEngine<T, U>::Returnstminus1[j][0]);
			}
     	}

		return simRtns;
	};

protected:
	Vec _generateDraws(copulaType ct = Gauss,double dof = 3., double gamma = 1.){

		Vec q; // Generate vector of d-dimensional copula

		switch(ct){
			case 0:	{q = _rng->getGaussiancopula(C);break;}
			case 1:	{q = _rng->getStudentcopula(C, dof);break;}
			case 2:	{q = _rng->getClaytoncopula(gamma);break;}
			case 3:	{q = _rng->getGumbelcopula(gamma);break;}
			default: {break;}
		}

		return q;
	}

private:
    Eigen::MatrixXd C;
	shared_ptr<copulaRng> _rng;
};

#endif //MC_ENGINE_H
