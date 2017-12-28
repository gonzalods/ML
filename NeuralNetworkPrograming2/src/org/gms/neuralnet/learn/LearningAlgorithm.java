package org.gms.neuralnet.learn;

import org.gms.neuralnet.NeuralNet;
import org.gms.neuralnet.data.NeuralDataSet;
import org.gms.neuralnet.exception.NeuralException;

public abstract class LearningAlgorithm {

	protected NeuralNet neuralNet;

	public enum LearningMode {ONLINE, BATCH};
	protected enum LearningParadigm {SUPERVISED, UNSUPERVISED};
    protected LearningMode learningMode;
    protected LearningParadigm learningParadigm;

	//Controlan el número de iteraciones para calcular los pesos
    protected int maxEpoch = 100;				// Número máximo de iteraciones
	protected int epoch = 0;					// Contador de iteraciones
	protected double minOverallError = 0.001;	// Error mínimo
	
	protected double learningRate = 0.1;		// Valor alfa.

	protected NeuralDataSet trainingDataSet;
	protected NeuralDataSet testingDataSet;
	protected NeuralDataSet validatingDataSet;
	
	public boolean printTraining = false;
	
	/*
	 * 
	 */
	public abstract void train() throws NeuralException;
	/*
	 * 
	 */
	public abstract void forward() throws NeuralException;
	public abstract void forward(int i) throws NeuralException;
	public abstract Double calcNewWeight(int layer, int input, int neuron) throws NeuralException;
	public abstract Double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException;
	public abstract void print();
	
	public int getMaxEpoch() {
		return maxEpoch;
	}
	public void setMaxEpoch(int maxEpoch) {
		this.maxEpoch = maxEpoch;
	}
	public double getMinOverallError() {
		return minOverallError;
	}
	public void setMinOverallError(double minOverallError) {
		this.minOverallError = minOverallError;
	}
	public double getLearningRate() {
		return learningRate;
	}
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	public int getEpoch() {
		return epoch;
	}
	public LearningMode getLearningMode() {
		return learningMode;
	}
	public void setLearningMode(LearningMode learningMode) {
		this.learningMode = learningMode;
	}
	public void setTestingDataSet(NeuralDataSet testingDataSet) {
		this.testingDataSet = testingDataSet;
	}
	
	
	
}
