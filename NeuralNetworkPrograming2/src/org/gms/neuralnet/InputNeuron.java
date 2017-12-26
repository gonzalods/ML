package org.gms.neuralnet;

import org.gms.neuralnet.math.IActivationFunction;

public class InputNeuron extends Neuron{

	public InputNeuron(int numberOfInputs, IActivationFunction iaf) {
		super(numberOfInputs, iaf);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void calc() {
		outputBeforeActivation = 0.0;
		if(numberOfInputs > 0) {
			if(input != null) {
				for(int i = 0;i < numberOfInputs;i++) {
					double valor = input.get(i);
					outputBeforeActivation += valor;
				}
			}
		}
		output = activationFunction.calc(outputBeforeActivation);
	}
}
