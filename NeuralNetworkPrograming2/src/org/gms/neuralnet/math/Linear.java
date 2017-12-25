package org.gms.neuralnet.math;

public class Linear implements IActivationFunction {

	private double a = 1.0;
	
	public Linear(double a) {
		this.a = a;
	}

	@Override
	public Double calc(Double x) {
		return a * x;
	}

}
