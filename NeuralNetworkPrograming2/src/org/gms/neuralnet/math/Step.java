package org.gms.neuralnet.math;

public class Step implements IActivationFunction {

	@Override
	public Double calc(Double x) {
		return x < 0.0?0.0:1.0;
	}

}
