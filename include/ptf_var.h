//ptf_var.h

#include <memory>

#include "portfolio.h"
#include "pca.h"
#include "rng.h"
#include "mc_engine.h"

using namespace std;

#ifndef PTF_VAR_H
#define PTF_VAR_H

//! Compute ptf's VaR using parametric approach
class VaRPtfCompute{

public:
	VaRPtfCompute(const shared_ptr<Portfolio>& _ptf, double _alpha = .05);
	virtual  ~VaRPtfCompute(){}
	VaRPtfCompute(const VaRPtfCompute& other); //= default; //copy constructor

	/*! \brief Compute dollar ptf's VaR

		$ VaRp = alpha * sDevp * P0 = alpha * sqrt(x' * Cov * x)

		Computing ptf VaR in such way enables to derive analytical solution for iVaR, mVaR, cVaR.
	*/
	double getPtfVaR() const;

	/*! \brief Measure the change in VaR if new position is added to the ptf

		Use approximation for incremental VaR

	    VaR(P0 + amount) - VaR(P0) =~ (deltaVaR)' * amount
	*/
	double computeIncrementalVaR(double amount) const;
	/*! \brief Measure the change in ptf VaR resulting adding add $ to a component

		delta VaR/ delta x[i] = ... = VaRp/P * beta[i]
	*/
	Vec computeMarginalVaR() const;

	/*! component VaR = ... = PtfVar * beta[i] * weights[i]

		sum of component VaR = ptf VaR
	*/
	Vec computeComponentVaR() const;
	//! Compute VaR of each single position
	Vec computeIndividualVaR() const;

	inline void setAlpha(double _alpha);

protected:
	shared_ptr<Portfolio> ptf;
	double alpha;
	Vec beta; //linear regression's beta coef ptf's rtn vs asset i rtn
	size_t nbAssets;
};



/*! \brief Compute VaR of a portfolio through Monte Carlo simulation

    Simulate each component separately

*/
template<class T, class V> //T model, V process
class VaRPtfMCCompute:public VaRPtfCompute{

public:

    typedef MCEngine<rng,V> engine;

    typedef std::vector<V> processes;

	VaRPtfMCCompute(const shared_ptr<Portfolio>& _ptf,T& _model, const processes& _processes_, rng& _rng_, double _alpha = .05):
	VaRPtfCompute(_ptf, _alpha),model(_model), _rng(_rng_), _processes(_processes_)
{
        model.setAlpha(alpha);

        Eigen::MatrixXd C = ptf->getVarCov();
        double n = C.rows(); double m = C.cols();

        //VaRPtfCompute::
        _ptf->isFixedIncome() == true ?  mt = pc : mt = Cholesky;

        Eigen::MatrixXd _A;

        if(mt == Cholesky){
            _A = C.llt().matrixL();
            A = _A;
        }

        else{
            // convert to req format
            vector<float> vec;

            for(size_t i = 0;i < n;++i)
                for(size_t j = 0;j < m;++j)
                    vec.push_back(C(i,j));

            std::shared_ptr<Pca> pca(new Pca());

            int init_result = pca->Calculate(vec, n, m);
            if(init_result == 1) {cout << "correl mat is not positive semi-definite" << endl;}

            vector<float> scores = pca->scores(); //Rotated data

            unsigned int kaiser = pca->kaiser(); //Kaiser criterion 99%

            if (kaiser < 3.) kaiser = 3.;

            unsigned int nrows = pca->nrows();

            _A.resize(nrows, kaiser);

            for(size_t i = 0;i < nrows;++i)
                for(size_t j = 0;j < kaiser;++j)
                    _A(i,j) = scores[j + kaiser*i];

            A = _A;
        }

    };

	virtual  ~VaRPtfMCCompute(){};

	double computeVaR(){

        engine _engine(_rng, _processes, A, mt);

        Mat rtns(nbAssets,Vec(2)); Vec sdevs;

        unsigned int window(ptf->getWindow());

        for(size_t j = 0;j < nbAssets;++j){

            size_t m(ptf->getComponentReturns(j).size()-1);

            rtns[j][0] =  ptf->getComponentReturns(j).at(m - 1);
            rtns[j][1] = ptf->getComponentReturns(j).at(m);

            sdevs.push_back(ptf->getComputeReturnStdDev(j).at(m - window));
        }

        _engine.setValues(rtns,sdevs);

        Mat simRtns = _engine.DoMultiSimulation();

        ptf->setReturns(simRtns); ptf->computeRtn(nbAssets);

        return model(0,ptf->getReturns());
    };

	//!compute VaR over whole path
    Vec computeVaRWholePath(){

        engine _engine(_rng, _processes, A, mt);

        unsigned int n(ptf->getReturns().size());
		unsigned int window(ptf->getWindow());

		size_t m(ptf->getRollingMean().size()-1);

		Mat rtns(nbAssets, Vec()); Mat sdevs(nbAssets, Vec());

		for(size_t i = 0;i < nbAssets;++i){
            rtns[i] = ptf->getComponentReturns(i);
            sdevs[i] = ptf->getComputeReturnStdDev(i);
		}

        Mat _rtns(nbAssets,Vec(2));

        Vec _VaRs;

        for(size_t i = 0;i < n - window;++i){

            Vec _sdevs;

            for(size_t j = 0;j < nbAssets;++j){

                size_t q; i > m ? q = m -1 : q = i;

                _rtns[j][0] = rtns[j].at(q + window - 1);
                _rtns[j][1] = rtns[j].at(q + window);

                _sdevs.push_back(sdevs[j].at(q));
            }

            _engine.setValues(_rtns,_sdevs);

            Mat simRtns = _engine.DoMultiSimulation(1e+04,Gaussian);

            ptf->setReturns(simRtns); ptf->computeRtn(nbAssets);

            _VaRs.push_back(model(0, ptf->getReturns()));
        }

        return _VaRs;
    };

private:
    T model;
    rng _rng;
    processes _processes;
	Eigen::MatrixXd A;
	matType mt;
};

inline void VaRPtfCompute::setAlpha(double _alpha){alpha = _alpha;}

#endif //PTF_VAR_H


