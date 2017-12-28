package org.gms.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.gms.neuralnet.math.IActivationFunction;

public class NeuralNet {

	private InputLayer inputLayer;
	private List<HiddenLayer> hiddenLayer;
	private OutputLayer outputLayer;
	private int numberOfHiddenLayers;
	private int numberOfInputs;
	private int numberOfOutputs;
	private List<Double> input;
	private List<Double> output;
	
	public NeuralNet(int numberOfInputs, int numberOfOutputs,
			int[] numberOfHiddenNeurons, IActivationFunction[] hiddenActFunc,
			IActivationFunction outActFunc) {
		this.numberOfInputs = numberOfInputs;
		this.numberOfOutputs = numberOfOutputs;
		input = new ArrayList<>(this.numberOfInputs);
		inputLayer = new InputLayer(this.numberOfInputs);
		numberOfHiddenLayers = numberOfHiddenNeurons.length;
		hiddenLayer = new ArrayList<>(numberOfHiddenLayers);
		for(int i = 0;i < this.numberOfHiddenLayers;i++) {
			if(i == 0) {
				hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i], hiddenActFunc[i], inputLayer.getNumberOfNeuronsInLayer()));
				inputLayer.setNextLayer(hiddenLayer.get(i));
			}else {
				hiddenLayer.add(new HiddenLayer(numberOfHiddenNeurons[i], hiddenActFunc[i], hiddenLayer.get(i-1).getNumberOfNeuronsInLayer()));
				hiddenLayer.get(i-1).setNextLayer(hiddenLayer.get(i));
			}
		}
		if(numberOfHiddenLayers > 0) {
			outputLayer = new OutputLayer(this.numberOfOutputs, outActFunc, hiddenLayer.get(numberOfHiddenLayers-1).getNumberOfNeuronsInLayer());
			hiddenLayer.get(numberOfHiddenLayers-1).setNextLayer(outputLayer);
		}else {
			outputLayer=new OutputLayer(numberOfInputs, outActFunc, this.numberOfOutputs);
		    inputLayer.setNextLayer(outputLayer);
		}
	}
	
	public void calc() {
		inputLayer.setInput(input);
		inputLayer.calc();
		for(int i = 0;i < numberOfHiddenLayers;i++) {
			HiddenLayer hl = hiddenLayer.get(i);
			hl.setInput(hl.getPreviousLayer().getOutput());
			hl.calc();
		}
		outputLayer.setInput(outputLayer.getPreviousLayer().getOutput());
		outputLayer.calc();
		this.output = outputLayer.getOutput();
	}

	public Double[] getOutput() {
		return output.toArray(new Double[] {});
	}
	
	public void setInput(Double[] input) {
		this.input = Arrays.asList(input);
	}
	public void print() {
		System.out.println(this);
		System.out.println("\tInptus: " + numberOfInputs);
		System.out.println("\tOutputs: " + numberOfOutputs);
		System.out.println("\tHidden Layers: " + numberOfHiddenLayers);
		for(int i = 0;i < numberOfHiddenLayers;i++) {
			System.out.println("\tHidden Layer " + i + ": " + hiddenLayer.get(i).getNumberOfNeuronsInLayer() + " Neurons");
		}
	}

	public int getNumberOfHiddenLayers() {
		return numberOfHiddenLayers;
	}

	public int getNumberOfOutputs() {
		return numberOfOutputs;
	}

	public int getNumberOfInputs() {
		return numberOfInputs;
	}

	public List<HiddenLayer> getHiddenLayer() {
		return hiddenLayer;
	}
	
	public HiddenLayer getHiddenLayer(int index) {
		return hiddenLayer.get(index);
	}

	public OutputLayer getOutputLayer() {
		return outputLayer;
	}
	
}
