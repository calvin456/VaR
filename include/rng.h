//rng.h

#include<memory>
#include<random>
#include<vector>
#include<stdint.h>
#include<math.h>

#include <Eigen/Dense>

#include"generate_rm_from_distr.h"

using namespace std;

#ifndef RANDOM_GEN_H
#define RANDOM_GEN_H

typedef std::vector<double> Vec;

static std::uniform_real_distribution<double> udist(0., 1.);

//! Mersenne-Twister pseudo random number generator
struct rng : std::mt19937 {
    explicit rng ( std::mt19937::result_type val = 12411 ){ seed(val); }
};


/*!
    \param Clayton lower tail
    \param Gumbel upper tail
*/
enum copulaType{Gauss, t,Clayton, Gumbel}; //extreme value copula, empirical copula

/*! Simulate d variables via copula of choice

	Gaussian, Student, Clayton, Gumbel
*/
class copulaRng{

public:

	copulaRng(unsigned int _d, rng& _rng_);
	~copulaRng(){}

	/*! Gaussian copula

		\param _C correlation matrix
		\result vector of d-size draws from \f$U(0,1)\f$
	*/
	Vec getGaussiancopula(Eigen::MatrixXd C);

	/*! Student's t copula

		\param _C correlation matrix
		\param v \f$\nu\f$ degree of freedom
		\result vector of d-size draws from \f$U(0,1)\f$

	*/
	Vec getStudentcopula(Eigen::MatrixXd C, double v);

	/*!	Clayon copula. More emphasis on the left tail of the distribution

		\param gamma \f$\gamma\f$
		\result vector of d-size draws from \f$U(0,1)\f$
	*/
	Vec getClaytoncopula(double gamma);

	/*!	Gumbel copula. More emphasis on the right tail of the distribution

		\param gamma \f$\gamma\f$
		\result vector of d-size draws from \f$U(0,1)\f$
	*/
	Vec getGumbelcopula(double gamma);

private:
	unsigned int d;
	shared_ptr<GenFromDistr<rng>> _rng;
};

/*! Simulate one stable distr value

	The stable distribution family is also sometimes referred to as the Lévy alpha-stable distribution.
*/
template<typename T>
double getOneStableDist(T& _rng_, double alpha, double beta, double gamma = 1., double delta = 0.){

	double a(M_PI/2.);

	double theta = _rng_->getOneUniform(-a,a);

	double W = _rng_->getOneExpDistr(1.);

	double theta0 = atan(beta * tan(M_PI * alpha/2.)/alpha);

	double Z(0.), X(0.);

	if(alpha != 1.){

        double tmp(cos(alpha * theta0 + (alpha - 1.)*theta)/W);
        double sgn; std::ignbit(tmp) == 1 ? sgn = -1. : sgn = 1.;

		Z = (sin(alpha*(theta0 + theta))/pow(cos(alpha* theta0)*cos(theta), 1./alpha)) *
				sgn * pow(abs(tmp), (1. - alpha)/alpha);

	}
	else Z = 1./a * ((a + beta * theta) * tan(theta) - beta * log((a * W * cos(theta))/(a + beta * theta)));

	if(!(gamma == 1. && delta == 0)){
		if(alpha != 1.) X = gamma * Z + delta;
		else X = gamma * Z + (delta + beta * 1./a * gamma * log(gamma)) ;
	}

	return X == 0 ? Z : X;
}

#endif //RANDOM_GEN_H
