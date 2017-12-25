package org.gms.neuralnet.math;

public class HyperTan implements IActivationFunction {

	private double a = 1.0;
	
	public HyperTan(double a) {
		this.a = a;
	}

	@Override
	public Double calc(Double x) {
		return (1.0 - Math.exp(-a + x)) / (1.0 + Math.exp(-a + x));
	}

}
