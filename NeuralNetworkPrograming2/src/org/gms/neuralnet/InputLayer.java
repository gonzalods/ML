package org.gms.neuralnet;

import java.util.ArrayList;
import java.util.Collections;

import org.gms.neuralnet.math.Linear;

public class InputLayer extends NeuralLayer {

	public InputLayer(int numberOfInputs) {
		this.numberOfInputs = numberOfInputs;
		this.numberOfNeuronsInLayer = numberOfInputs;
		this.init();
	}
	
	@Override
	public void init(){
		neuron = new ArrayList<>(numberOfNeuronsInLayer);
		activationFunction = new Linear(1.0);
		for(int i = 0;i < numberOfNeuronsInLayer;i++) {
			try {
				neuron.get(i).setActivationFunction(activationFunction);
				neuron.get(i).init();
			}catch (IndexOutOfBoundsException e) {
				neuron.add(new InputNeuron(1, activationFunction));
				neuron.get(i).init();
			}
		}
		output = new ArrayList<>(numberOfNeuronsInLayer);		
	}

	@Override
	public void calc() {
		for(int i = 0;i < numberOfNeuronsInLayer;i++) {
			
			neuron.get(i).setInputs(Collections.singletonList(input.get(i)));
			neuron.get(i).calc();
			try {
				output.set(i,neuron.get(i).getOutput());
			}catch (IndexOutOfBoundsException e) {
				output.add(neuron.get(i).getOutput());
			}
		}
	}
	
	
}
