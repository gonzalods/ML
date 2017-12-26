package org.gms.neuralnet.learn;

import java.util.ArrayList;

import org.gms.neuralnet.NeuralNet;
import org.gms.neuralnet.data.NeuralDataSet;
import org.gms.neuralnet.exception.NeuralException;

public class DeltaRule extends LearningAlgorithm {
    public ArrayList<ArrayList<Double>> error;
    public ArrayList<Double> generalError;
    public ArrayList<Double> overallError;
    public double overallGeneralError;
    
    public ArrayList<ArrayList<Double>> testingError;
    public ArrayList<Double> testingGeneralError;
    public ArrayList<Double> testingOverallError;
    public double testingOverallGeneralError;
    
    
    public double degreeGeneralError=2.0;
    public double degreeOverallError=0.0;
    
    public enum ErrorMeasurement {SimpleError, SquareError,NDegreeError,MSE}
    
    public ErrorMeasurement generalErrorMeasurement=ErrorMeasurement.SquareError;
    public ErrorMeasurement overallErrorMeasurement=ErrorMeasurement.MSE;
    
    private int currentRecord=0;
    
    private ArrayList<ArrayList<ArrayList<Double>>> newWeights;
	
    public DeltaRule(NeuralNet _neuralNet){
    	// TODO Auto-generated method stub
    }
    
    public DeltaRule(NeuralNet _neuralNet,NeuralDataSet _trainDataSet){
    	//TODO
    }
    
    public DeltaRule(NeuralNet _neuralNet,NeuralDataSet _trainDataSet
            ,LearningMode _learningMode){
    	// TODO Auto-generated method stub
    }
    
    
	@Override
	public void setTestingDataSet(NeuralDataSet testingDataSet) {
		// TODO Auto-generated method stub
		super.setTestingDataSet(testingDataSet);
	}

	@Override
	public void train() throws NeuralException {
		// TODO Auto-generated method stub

	}

	@Override
	public void forward() throws NeuralException {
		// TODO Auto-generated method stub

	}

	@Override
	public void forward(int i) throws NeuralException {
		// TODO Auto-generated method stub

	}

	@Override
	public Double calcNewWeight(int layer, int input, int neuron) throws NeuralException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Double calcNewWeight(int layer, int input, int neuron, double error) throws NeuralException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void print() {
		// TODO Auto-generated method stub
		
	}

	public void setGeneralErrorMeasurement(ErrorMeasurement errorMeasurement){
		//TODO
	}
}
