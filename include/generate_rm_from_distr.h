//generate_rm_from_distr.h

#include <random>

#ifndef DIST_RANDOM_GEN_H
#define DIST_RANDOM_GEN_H

static std::normal_distribution<double> gaussian(0,1);

//! Generate random number according to specified distribution
template<class T>
class GenFromDistr{

public:
	GenFromDistr(T& _rng):_rng_(_rng){}
	~GenFromDistr(){}

	double operator()(){return T();}

    double getOneUniform(double a = 0., double b = 1.){

		std::uniform_real_distribution<double> d(a,b);

		return d(_rng_);
	}

	//! Generate vector dim x of Uniform random number
	Eigen::VectorXd getXUniform(unsigned int x){

		Eigen::VectorXd tmp(x);

		for(size_t i = 0;i < x;++i)
            tmp(i) = this->getOneUniform();

		return tmp;
	}

	double getOneGaussian(){
		return gaussian(_rng_);
	}

	//! Generate vector dim x of Gaussian random number
	Eigen::VectorXd getXGaussian(unsigned int x){

		Eigen::VectorXd tmp(x);

		for(size_t i = 0;i < x;++i)
            tmp(i) = this->getOneGaussian();

		return tmp;
	}

	/*!
		\param n degree of freedom
	*/
	double getOneChisquare(double n){
		std::chi_squared_distribution<double> d(n);

		return d(_rng_);
	}

	/*!
		\param n degree of freedom
	*/
	double getOneStudent(double n){
		std::student_t_distribution<double> d(n);

		return d(_rng_);
	}

	//! Generate vector dim x of Student random number
	Eigen::VectorXd getXStudent(unsigned int x,double n){

		Eigen::VectorXd tmp(x);

		for(size_t i = 0;i < x;++i)
            tmp(i) = this->getOneStudent(n);

		return tmp;
	}

	/*!
		\param l lambda
	*/
	double getOneExpDistr(double l){
		std::exponential_distribution<double> d(l);

		return d(_rng_);
	}

	/*!
		\param a \f$alpha\f$
		\param b \f$beta\f$
	*/
	double getOneGamma(double a, double b){
		std::gamma_distribution<double> d(a,b);

		return d(_rng_);
	}

private:
    T _rng_;
};

#endif //DIST_RANDOM_GEN_H

