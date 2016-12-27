//var_bridge.h

#include<memory>
#include"var_model.h"

#ifndef VAR_MODELS_BRIGE_H
#define VAR_MODELS_BRIGE_H

class VaRBridge{

public:
	VaRBridge();
	~VaRBridge(){}
	VaRBridge(const VaRBridge& other) = default;
	inline double getAlpha() const;
	inline void setAlpha(double _alpha);

	inline double operator()(double _meanReturn,
							 const Vec& returns) const;

	inline double operator()(double _meanReturn,
							 double _sigmatminus1,double _returnt) const;

	inline double operator()(const Vec& returns) const;
private:
	std::unique_ptr<VaR> model;
};

inline double VaRBridge::getAlpha() const {return model->getAlpha();}

inline void VaRBridge::setAlpha(double _alpha){model->setAlpha(_alpha);}

inline double VaRBridge::operator()(double _meanReturn,
									const Vec& returns) const{
	return model->operator()(_meanReturn, returns);
}


inline double VaRBridge::operator()(const Vec& returns) const{
	return model->operator()(returns);
}

inline double VaRBridge::operator()(double _meanReturn,
									double _sigmatminus1,double _returnt) const{
	return model->operator()(_meanReturn, _sigmatminus1, _returnt);

}

#endif //VAR_MODELS_BRIGE_H
