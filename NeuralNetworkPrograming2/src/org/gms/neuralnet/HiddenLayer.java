package org.gms.neuralnet;

import org.gms.neuralnet.math.IActivationFunction;

public class HiddenLayer extends NeuralLayer {

	public HiddenLayer(int numberOfNeurons, IActivationFunction iaf, int numberoOfInputs) {
		this.numberOfNeuronsInLayer = numberOfNeurons;
		this.activationFunction = iaf;
		this.numberOfInputs = numberoOfInputs;
		init();
	}
}
