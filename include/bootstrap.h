// bootstrap.h

#include<vector>
#include<map>
#include<memory>
#include <functional>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "rng.h"

using namespace std;
using namespace boost::accumulators;

#ifndef BOOTSTRAP_H
#define BOOTSTRAP_H

/*! Compute historical VaR with bootstrap resampling

\param U VaR model
\param T compute return, portfolio
*/
template<class T, class U>
class Bootstrap{

public:

    Bootstrap():_rng(rng()){}
    ~Bootstrap(){}

	//!compute mean and standard error of VaR estimate through bootstrap
	/*!
		\param _cr T class (compute return, portfolio)
		\param model U class (VaR model)
		\param nSim number of sample drawn
		\result mean and standard error of VaR estimate
	*/

    std::pair<double,double>  compute(shared_ptr<T>& _cr, const U& model, size_t p, double nSim = 1e+4){


        std::vector<double> _rtns = _cr->getReturns(p);

        unsigned int m(_cr->getRollingStdDev(p).size() -1);

        double _sigmatminus1 = _cr->getRollingStdDev(p).at(m -1);

        unsigned int n(_rtns.size());
        unsigned int window(_cr->getWindow());

        std::vector<double> rtns(window);

        for(unsigned int i = 0;i < window;++i)
            rtns[i] = _rtns[n - window + i];

        std::vector<double> simulations;

        for(unsigned int i = 0;i < static_cast<unsigned int>(nSim);++i){

            //Generate i th sampling on the original sample
            std::vector<double> _returns;

            for(unsigned int j = 0;j < static_cast<unsigned int>(window);++j)
                _returns.push_back(rtns[size_t(window * udist(_rng) - 1)]);

            simulations.push_back(model(_sigmatminus1, _returns)); // compute VaR
        }

        accumulator_set<double, features<tag::mean, tag::variance>> acc;
        acc = std::for_each(simulations.begin(), simulations.end(), acc);

        double _meanVaR = boost::accumulators::mean(acc); //boost mean
        double _sdevVaR = sqrt(boost::accumulators::variance(acc)); //boost stdev

        return std::pair<double,double>(_meanVaR, _sdevVaR);
    }

private:
    rng _rng;

};

#endif //BOOTSTRAP_H
