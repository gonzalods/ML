package org.gms.neuralnet;

public class InputLayer extends NeuralLayer {

	public InputLayer(int numberOfInputs) {
		this.numberOfInputs = numberOfInputs;
		this.init();
	}
}
