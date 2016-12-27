//instrument.h

#include<vector>

using namespace std;

#ifndef INSTRUMENTS_H
#define INSTRUMENTS_H

typedef std::vector<double> Vec;

/*! Does not attempt to make full re-evaluation on the ptf. 

	Use approximation approach instead
*/
class Instrument{

public:
	Instrument(){};
	Instrument(const Instrument& other);
	virtual ~Instrument(){}

	//! Compute an approximation on P&L of instrument following shock
	virtual double operator()(double rtn) const = 0;

private:

};

//! Sensitivity one for one. FX, index, commodity prices

class DeltaOne: public Instrument{

public:
	DeltaOne();
	virtual ~DeltaOne(){}
	double operator()(double rtn) const;
private:

};

class Equity: public Instrument{

public:
	Equity(double _beta = 1.);
	virtual ~Equity(){}
	double operator()(double rtn) const;

private:
	double beta;
};

class Derivatives: public Instrument{

public:
	Derivatives(double _delta, double _gamma);
	virtual ~Derivatives(){}
	double operator()(double rtn) const;

private:
	double delta;
	double gamma;
};

//!dc : modified duration and convexity (%change)
//!DV01, PV01 : dollar value chge per basis point
//!DV01 != PV01 value not the same !!!!!
//!key rate duration

enum durationType {dc, DV01, krd};

class FI: public Instrument{

public:
	FI(double _duration, double _convexity = 0., durationType _dt = dc);
	FI(Vec& _duration);
	virtual ~FI(){}
	double operator()(double rtn) const;
	double operator()(const Vec& rtns) const;

private:
	Vec duration;
	double convexity;
	durationType dt;
};

#endif //INSTRUMENTS_H
