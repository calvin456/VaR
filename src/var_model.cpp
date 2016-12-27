//var_model.cpp

#include <math.h>
#include <algorithm>
#include <functional>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "var_model.h"

using namespace std;
using namespace boost::accumulators;

//Define container for muliple criteria sorting
struct Container{
                double _return; //returns
                double _prob; //weighted prob,
            };

bool sortByReturn(const Container &lhs, const Container &rhs) { return lhs._return < rhs._return; }

VaR::VaR(double _alpha):alpha(_alpha){}

//copy constructor implementation
VaR::VaR(const VaR& other):    
	alpha(other.alpha)
{}

double VaR::getAlpha() const {return alpha;}

void VaR::setAlpha(double _alpha) {alpha = _alpha;}

ParametricVaR::ParametricVaR(double _alpha, bool _logNormal):VaR(_alpha),logNormal(_logNormal)
{}

//copy constructor implementation
ParametricVaR::ParametricVaR(const ParametricVaR& other):
	VaR(other),logNormal(other.logNormal) 
{}


double ParametricVaR::NormalVaR(double u, double sigma) const{

		double za;

		za =  boost::math::quantile(snd, getAlpha());

		return (u + sigma * za);
}

double ParametricVaR::LogNormalVaR(double u, double sigma) const {

		double za;

		za =  boost::math::quantile(snd, getAlpha());

		return -(1. - exp(u + sigma * za));
}

RiskMetricsVaR::RiskMetricsVaR(double _alpha, double _lambda, bool _logNormal)
:ParametricVaR(_alpha, _logNormal), lambda(_lambda){}

//copy constructor implementation
RiskMetricsVaR::RiskMetricsVaR(const RiskMetricsVaR& other):
	ParametricVaR(other), lambda(other.lambda)
{}

double RiskMetricsVaR::operator()(double _meanReturn,
                                  double _sigmatminus1, double _returnt) const{

    double sigmat = sqrt(lambda * _sigmatminus1 * _sigmatminus1 + (1. - lambda) * _returnt * _returnt);

	double tmp = logNormal ? LogNormalVaR(_meanReturn, sigmat) : NormalVaR(_meanReturn, sigmat);

	return tmp >= 0 ? 0. : tmp;

}

double RiskMetricsVaR::operator()(double _meanReturn, const Vec& returns) const
{
	double sigmaT = 0.;
    size_t T(returns.size());

    for(size_t i = 1; i < T;++i){
        sigmaT = sqrt(lambda * sigmaT * sigmaT + (1. - lambda) * returns[i] * returns[i]);
    }

	double tmp = logNormal ? LogNormalVaR(_meanReturn, sigmaT) : NormalVaR(_meanReturn, sigmaT);

	return tmp >= 0 ? 0. : tmp;
}

double RiskMetricsVaR::operator()(const Vec& returns) const
{
    double sigmaT = 0.;
    double _meanReturn =0;
    size_t T(returns.size());

    for(size_t i = 1; i < T;++i){
        sigmaT = sqrt(lambda * sigmaT * sigmaT + (1. - lambda) * returns[i] * returns[i]);
        _meanReturn += returns[i];
    }

    _meanReturn =_meanReturn/(T-1);

    double tmp = logNormal ? LogNormalVaR(_meanReturn, sigmaT) : NormalVaR(_meanReturn, sigmaT);

	return tmp >= 0 ? 0. : tmp;
}

GarchVaR::GarchVaR(double _alpha, double _a, double _b,  double _c,bool _logRtns)
:ParametricVaR(_alpha, _logRtns), a(_a), b(_b), c(_c){}

//copy constructor implementation
GarchVaR::GarchVaR(const GarchVaR& other):	
	ParametricVaR(other), a(other.a), b(other.b),c(other.c)
{}

double GarchVaR::operator()(double _meanReturn,
                            double _sigmatminus1, double _returntminus1) const{

    double sigmat = sqrt(a + c * _sigmatminus1 * _sigmatminus1 + b * _returntminus1 * _returntminus1);

	double tmp = logNormal ? LogNormalVaR(_meanReturn, sigmat) : NormalVaR(_meanReturn, sigmat);

	return tmp >= 0 ? 0. : tmp;

}

double GarchVaR::operator()(double _meanReturn, const Vec& returns) const
{
	double sigmaT = 0.;

    for(size_t i = 0; i < returns.size();++i)
        sigmaT += pow(c,static_cast<double>(i)) * returns[i] * returns[i];

    sigmaT *= b;
    sigmaT += a/(1.-c);

	double tmp = logNormal ? LogNormalVaR(_meanReturn, sqrt(sigmaT)) : NormalVaR(_meanReturn, sqrt(sigmaT));

	return tmp >= 0 ? 0. : tmp;
}

double GarchVaR::operator()(const Vec& returns) const
{
    double sigmaT = 0.;
    double _meanReturn =0;
    size_t T(returns.size());

    for(size_t i = 0; i < T;++i){
        sigmaT += pow(c,static_cast<double>(i)) * returns[i] * returns[i];
        _meanReturn += returns[i];
    }

    sigmaT *= b;
    sigmaT += a/(1.-c);
    _meanReturn =_meanReturn/(T-1);

	double tmp = logNormal ? LogNormalVaR(_meanReturn, sigmaT) : NormalVaR(_meanReturn, sigmaT);

	return tmp >= 0 ? 0 : tmp;
}

NoneParametricVaR::NoneParametricVaR(double _alpha):VaR(_alpha){}

NoneParametricVaR::NoneParametricVaR(const NoneParametricVaR& other)
:VaR(other)
{}

HistoricalVaR::HistoricalVaR(double _alpha, double _lambda, WeightingScheme _ws)
:NoneParametricVaR(_alpha), lambda(_lambda), ws(_ws){}

//copy constructor implementation
HistoricalVaR::HistoricalVaR(const HistoricalVaR& other):	
	NoneParametricVaR(other), lambda(other.lambda), ws(other.ws)
{}

double HistoricalVaR::operator()(double _sigmatminus1,
                                 const Vec& returns) const
{
 	double T = returns.size();

	double alpha(getAlpha());

	unsigned int c(alpha * T);
	c == 0 ? c : c-= 1;

	double tmp;

	int input(ws);

	std::vector<double> _returns(returns);

	switch(input){

	case 0:
        {
            // sort returns find cth arg - Histogram method
            std::sort(_returns.begin(), _returns.end());

            tmp = _returns[c];
        }

        break;

	case 1:
        {
            std::vector<Container> prob(T);

            double a = (1. - lambda)/(1. - pow(lambda, T));

            for(size_t i = 0; i < T; ++i){
                prob[i]._return = _returns[T -1 - i];
                prob[i]._prob = a * pow(lambda, i);
            }

            // Sort Container by return function
            std::sort(prob.begin(), prob.end(),sortByReturn);

            double sumProb(0);

            for(size_t i = 0; i < T; ++i){
                sumProb += prob[i]._prob;

                if(sumProb >= alpha){
                    tmp = prob[i]._return;
                    break;
                }
            }

        }


        break;

    case 2:

        {
            std::vector<double> sigma;

            sigma.push_back(_sigmatminus1 * _sigmatminus1); 

            for(size_t i = 1; i < T;++i)
                sigma.push_back(lambda * sigma[i - 1] + (1. - lambda) * returns[i - 1] * returns[i - 1]);

            double sigmaT(sigma.back()); 

            std::vector<double> _wgthRtns;

            for(size_t i = 0; i < T; ++i){

                if(sigma[i] <= 0) _wgthRtns.push_back(_returns[i]);
                else _wgthRtns.push_back(sqrt(sigmaT / sigma[i]) * _returns[i] );
            }


            // sort returns find cth arg - Histogram method
            std::sort(_wgthRtns.begin(), _wgthRtns.end());

            tmp = _wgthRtns[c];
        }

        break;

    default:
        break;

	}

	return tmp >= 0 ? 0. : tmp;
}

ExtremeValueVaR::ExtremeValueVaR(double _alpha):VaR(_alpha){}

//copy constructor implementation
ExtremeValueVaR::ExtremeValueVaR(const ExtremeValueVaR& other):   
	VaR(other)
{}

PoTVaR::PoTVaR(double _u,double _beta, double _xi,double _alpha)
: ExtremeValueVaR(_alpha),xi(_xi),beta(_beta),u(_u){

}

//copy constructor implementation
PoTVaR::PoTVaR(const PoTVaR& other): 
	ExtremeValueVaR(other), xi(other.xi), beta(other.beta), u(other.u)
{}

double PoTVaR::operator()(const double& ratio) const
{
	double a = 1./ratio * getAlpha();

	return -(u +  beta * (pow(a,- xi) - 1.)/ xi );
}

double PoTVaR::expectedShortfall(const double& ratio) const
{
	return ((this->operator()(ratio) - (beta - xi * u)))/(1. - xi);

}

double PoTVaR::expectedShortfall(const Vec& returns) const{

	double ratio(0.);

	for(size_t i =0;i < returns.size();++i){
        if(returns[i] < 0)
            if(-returns[i] > u) ratio +=1;
	}

	ratio /= returns.size();
    if(ratio < .05) ratio = .05;


	return this->expectedShortfall(ratio);
}

double PoTVaR::operator()(const Vec& returns) const{

	double ratio(0.);

	for(size_t i =0;i < returns.size();++i){
        if(returns[i] < 0)
            if(-returns[i] > u) ratio +=1;
	}

	ratio /= returns.size();
    if(ratio < .05) ratio = .05;

	return this->operator()(ratio);
}



