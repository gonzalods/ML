package org.gms.neuralnet;

import java.util.ArrayList;
import java.util.List;

import org.gms.neuralnet.math.IActivationFunction;
import org.gms.neuralnet.math.RandomNumberGenerator;

public class Neuron {

	protected ArrayList<Double> weight;
	protected List<Double> input;
	protected Double output;
	protected Double outputBeforeActivation;
	protected int numberOfInputs = 0;
	protected double bias = 1.0;
	protected IActivationFunction activationFunction;
	
	public Neuron(int numberOfInputs, IActivationFunction iaf) {
		this.numberOfInputs = numberOfInputs;
		weight = new ArrayList<>(this.numberOfInputs+1);
		input = new ArrayList<>(this.numberOfInputs);
		activationFunction = iaf;
	}
	
	public void init() {
		for(int i = 0;i <= numberOfInputs;i++) {
			double newWeight = RandomNumberGenerator.GenerateNext();
			try {
				weight.set(i, newWeight);
			}catch (IndexOutOfBoundsException e) {
				weight.add(newWeight);
			}
		}
	}
	
	public void calc() {
		outputBeforeActivation = 0.0;
		if(numberOfInputs > 0) {
			if(input != null && weight != null) {
				for(int i = 0;i <= numberOfInputs;i++) {
					double valor = (i == numberOfInputs?bias:input.get(i));
					outputBeforeActivation += valor*weight.get(i);
				}
			}
		}
		output = activationFunction.calc(outputBeforeActivation);
	}
	
	public void setActivationFunction(IActivationFunction af) {
		this.activationFunction = af;
	}
	
	public void setInputs(List<Double> input) {
		this.input = input;
	}
	
	public Double getOutput() {
		return output;
	}
}
