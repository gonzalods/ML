package org.gms.neuralnet;

import org.gms.neuralnet.math.IActivationFunction;

public class OutputLayer extends NeuralLayer {

	public OutputLayer(int numberOfNeurons, IActivationFunction iaf, int numberOfinputs) {
		this.numberOfNeuronsInLayer = numberOfNeurons;
		this.activationFunction = iaf;
		this.numberOfInputs = numberOfinputs;
		init();
	}
}
