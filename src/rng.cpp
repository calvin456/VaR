//rng.cpp

#include<memory>
#include<math.h>
#include<cmath>

#include <boost/math/distributions.hpp>

#include "rng.h"

using namespace std;

copulaRng::copulaRng(unsigned int _d, rng& _rng_)
:d(_d)
{
    _rng = shared_ptr<GenFromDistr<rng>>(new GenFromDistr<rng>(_rng_));
}

// Gaussian
Vec copulaRng::getGaussiancopula(Eigen::MatrixXd C){

	boost::math::normal_distribution<> dist(0.,1.);

	//d dimension normal(0, cov)
	Eigen::VectorXd X = _rng->getXGaussian(d );

	Eigen::MatrixXd A( C.llt().matrixL() );

	Eigen::VectorXd Z = X.transpose() * A ;

	Vec sample;

	for(size_t i = 0;i < d;++i)
        sample.push_back(boost::math::cdf(dist,Z(i)));
	
	return sample;
}

// Studen's t-distr
Vec copulaRng::getStudentcopula(Eigen::MatrixXd C, double v){

	boost::math::students_t_distribution<> dist(v);

	//d dimension t-distr(0, cov, degree of freedom) central

	//1. Find the Cholesky matrix of Sigma -> L
	Eigen::MatrixXd L( C.llt().matrixL() );

	//2. Simulate a vector of n N(0,1) iid -> Y
	Eigen::VectorXd Y = _rng->getXGaussian(d );

	//3. Simulate a Chi-Square(v) -> S
	double S = _rng->getOneChisquare(v);
	//4. Compute vector Z = sqrt(v/S) * L * Y
	Eigen::VectorXd Z = L * Y;
	Z *= sqrt(v/S);

	//5. Finally, U = cdf_student(Z)
	Vec sample;

	for(size_t i = 0;i < d;++i)
		sample.push_back(boost::math::cdf(dist,Z(i)));

	return sample;
}

// Clayton
Vec copulaRng::getClaytoncopula(double gamma){

	double X = _rng->getOneGamma(1./gamma, 1.); //1 dimension gamma_dist(1./gamma, 1.)

	Eigen::VectorXd U = _rng->getXUniform(d); //d dimension uniform(0,1)

	Vec sample;

	for(size_t i = 0;i < d;++i)
		sample.push_back(pow(1.- log(U(i)/X) , -1./gamma));

	return sample;
}

// Gumbel
Vec copulaRng::getGumbelcopula(double gamma){

	double theta = -pow(-cos(M_PI/(2. * gamma)),gamma); 

	//1 dimension stable_dist(1./theta, 1.,gamma,0.)
	double X = getOneStableDist<shared_ptr<GenFromDistr<rng>>>(_rng,1./gamma, 1.,theta,0.); 

	//d dimension uniform(0,1)
	Eigen::VectorXd U = _rng->getXUniform(d);

	Vec sample;

	for(size_t i = 0;i < d;++i)
        sample.push_back(exp(-pow(abs(log(abs(U(i)/X))) , 1./gamma)));
	
	return sample;
}

