package org.gms.neuralnet;

import java.util.ArrayList;
import java.util.List;

import org.gms.neuralnet.math.IActivationFunction;

public abstract class NeuralLayer {

	protected int numberOfNeuronsInLayer;
	protected ArrayList<Neuron> neuron;
	protected IActivationFunction activationFunction;
	protected NeuralLayer previousLayer;
	protected NeuralLayer nextLayer;
	protected List<Double> input;
	protected List<Double> output;
	protected int numberOfInputs;
	
	protected void init() {
		neuron = new ArrayList<>(numberOfNeuronsInLayer);
		for(int i = 0;i < numberOfNeuronsInLayer;i++) {
			try {
				neuron.get(i).setActivationFunction(activationFunction);
				neuron.get(i).init();
			}catch (IndexOutOfBoundsException e) {
				neuron.add(new Neuron(numberOfInputs, activationFunction));
				neuron.get(i).init();
			}
		}
		output = new ArrayList<>(numberOfNeuronsInLayer);
	}
	
	public void calc() {
		for(int i = 0;i < numberOfNeuronsInLayer;i++) {
			neuron.get(i).setInputs(input);
			neuron.get(i).calc();
			try {
				output.set(i,neuron.get(i).getOutput());
			}catch (IndexOutOfBoundsException e) {
				output.add(neuron.get(i).getOutput());
			}
		}
	}

	public int getNumberOfNeuronsInLayer() {
		return numberOfNeuronsInLayer;
	}
	
	public NeuralLayer getPreviousLayer() {
		return previousLayer;
	}

	public List<Double> getOutput() {
		return output;
	}

	public void setNextLayer(NeuralLayer nextLayer) {
		this.nextLayer = nextLayer;
		this.nextLayer.previousLayer = this;
	}
	
	
	public void setInput(List<Double> input) {
		this.input = input;
	}
	
	public Neuron getNeuron(int index){
		return neuron.get(index);
	}
}
