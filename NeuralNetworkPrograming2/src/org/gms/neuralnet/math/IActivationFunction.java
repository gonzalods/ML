package org.gms.neuralnet.math;

public interface IActivationFunction {

	Double calc(Double x);
	
	public enum ActivationFunctionENUM{
		STEP, LINEAR, SIGMOID, HYPERTAN
	}
}
