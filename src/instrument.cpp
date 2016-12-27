//instrument.cpp

#include<iostream>

#include "instrument.h"

DeltaOne::DeltaOne():Instrument(){}

double DeltaOne:: operator()(double rtn) const {return rtn;}

Equity::Equity(double _beta):Instrument(),beta(_beta){}

double Equity::operator()(double rtn) const {return beta * rtn;}

Derivatives::Derivatives(double _delta, double _gamma):Instrument(),delta(_delta),gamma(_gamma){}

double Derivatives::operator()(double rtn) const {return delta * rtn - .5 * gamma * rtn * rtn;}

FI::FI(double _duration, double _convexity, durationType _dt):Instrument(),convexity(_convexity), dt(_dt){

	duration.clear();
	duration.push_back(_duration);
}


FI::FI(std::vector<double>& _duration):Instrument(),dt(krd){

	duration.clear();

	for(size_t i = 0;i < _duration.size();++i)
		duration.push_back(_duration[i]);
}

double FI::operator()(double rtn) const {

	if(dt !=2)
		return duration[0] * rtn - .5 * convexity * rtn * rtn;
	else{
		cout << "not to be used with key rate duration" << endl;
		return 0.;
	}

}

double FI::operator()(const Vec& rtns) const {

	if(dt !=2){
        cout << "not to be used with key rate duration" << endl;
		return 0.;

	}
	else{
		double tmp(0);
		for(size_t i = 0;i < rtns.size();++i) tmp += duration[i] * rtns[i];
		return tmp;
	}

}
