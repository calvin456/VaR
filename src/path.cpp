//paths.cpp

#include<math.h>

#include "path.h"

using namespace std;

Path1x1::Path1x1():Path()
{}

double Path1x1::doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1) const {

    return Returnt + Sigmat * u;

}

Eigen::VectorXd Path1x1::doSim(const Vec& u,
                                    const Vec& Sigmat,const Vec& Returnt) const {
    size_t n(Returnt.size());
    Eigen::VectorXd simRtn(n);

    for(size_t i =0;i < n;++i)  simRtn(i) = u[i];

    return simRtn;
}



AR1xGARCH11::AR1xGARCH11(double _mu, double _correl,
						 double _alpha, double _beta, double _gamma)
:Path(),mu(_mu),correl(_correl),alpha(_alpha),beta(_beta),gamma(_gamma)
{}

double AR1xGARCH11::doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1) const {

	double Etat(Returnt - mu + correl * Returntminus1); 

	double tmp(alpha + beta * Etat * Etat + gamma * Sigmat * Sigmat);

	double Sigmatplus1 = sqrt(tmp < 0 ? 0 : tmp); 

	double Etatplus1(Sigmatplus1 * u); 

	return mu + correl * Returnt + Etatplus1;
}

Eigen::VectorXd AR1xGARCH11::doSim(const Vec& u,
                                    const Vec&  Sigmat,const Vec& Returnt) const{

    size_t n(Returnt.size());
    Eigen::VectorXd simRtn(n);

    for(size_t i =1;i < n;++i)
        simRtn(i) = this->doOneSim(u[i-1],Sigmat[i-1],Returnt[i],Returnt[i-1] );

    return simRtn;
}

double GARCH11::doOneSim(const double& u,
                                const double& Sigmat, const double& Returnt, const double& Returntminus1) const{

    double Etat(Returnt);

	double Sigmatplus1 = sqrt(alpha + beta * Etat * Etat + gamma * Sigmat * Sigmat);

	double Etatplus1(Sigmatplus1 * u);

	return Etatplus1;
}

Eigen::VectorXd GARCH11::doSim(const Vec& u,
                                    const Vec&  Sigmat,const Vec& Returnt) const{
    size_t n(Returnt.size());
    Eigen::VectorXd simRtn(n);

    for(size_t i =1;i < n;++i)
        simRtn(i) = this->doOneSim(u[i-1],Sigmat[i-1],Returnt[i],0.);

    return simRtn;
}



