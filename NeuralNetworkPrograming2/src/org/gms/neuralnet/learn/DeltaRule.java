package org.gms.neuralnet.learn;

import java.util.ArrayList;

import org.gms.neuralnet.NeuralNet;
import org.gms.neuralnet.Neuron;
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
    
    private boolean normalization;
	
    public DeltaRule(NeuralNet neuralNet){
    	learningParadigm = LearningParadigm.SUPERVISED;
    	this.neuralNet = neuralNet;
    	newWeights = new ArrayList<>();
    	int numberOfHiddenLayers = neuralNet.getNumberOfHiddenLayers();
    	for(int l = 0;l <= numberOfHiddenLayers;l++){
    		int numberOfNeuronsInLayer, numberOfInputsInNeuron;
    		newWeights.add(new ArrayList<ArrayList<Double>>());
    		if(l < numberOfHiddenLayers){
    			numberOfNeuronsInLayer = neuralNet.getHiddenLayer(l).getNumberOfNeuronsInLayer();
    			for(int j = 0;j < numberOfNeuronsInLayer;j++){
    				numberOfInputsInNeuron = neuralNet.getHiddenLayer(l).getNeuron(j).getNumberOfInputs();
    				newWeights.get(l).add(new ArrayList<>());
    				for(int i = 0;i < numberOfInputsInNeuron;i++){
    					newWeights.get(l).get(j).add(0.0);
    				}
    			}
    		}else{
    			numberOfNeuronsInLayer = neuralNet.getOutputLayer().getNumberOfNeuronsInLayer();
    			for(int j = 0;j < numberOfNeuronsInLayer;j++){
    				numberOfInputsInNeuron = neuralNet.getOutputLayer().getNeuron(j).getNumberOfInputs();
    				newWeights.get(l).add(new ArrayList<>());
    				for(int i = 0;i < numberOfInputsInNeuron;i++){
    					newWeights.get(l).get(j).add(0.0);
    				}
    			}
    		}
    	}
    }
    
    public DeltaRule(NeuralNet neuralNet,NeuralDataSet trainDataSet){
    	this(neuralNet);
    	this.trainingDataSet = trainDataSet;
    	if(trainingDataSet.inputNorm!=null){
    		this.normalization = true;
    	}
    	generalError = new ArrayList<>();
    	error = new ArrayList<>();
    	overallError = new ArrayList<>();
    	for(int i = 0;i < trainDataSet.numberOfRecords;i++){
    		generalError.add(null);
    		error.add(new ArrayList<>());
    		for(int j = 0;j <= neuralNet.getNumberOfOutputs();j++){
    			if(i == 0){
    				overallError.add(null);
    			}
    			error.get(i).add(null);
    		}
    	}
    }
    
    public DeltaRule(NeuralNet neuralNet,NeuralDataSet trainDataSet
            ,LearningMode learningMode){
    	this(neuralNet,trainDataSet);
    	this.learningMode = learningMode;
    }
    
    
	@Override
	public void setTestingDataSet(NeuralDataSet testingDataSet) {
		this.testingDataSet = testingDataSet;
		if(testingDataSet.inputNorm != null){
			normalization = true;
		}
		testingGeneralError = new ArrayList<>();
		testingError = new ArrayList<>();
		testingOverallError = new ArrayList<>();
		for(int i = 0;i < testingDataSet.numberOfRecords;i++){
			testingGeneralError.add(null);
			testingError.add(new ArrayList<>());
			for(int j = 0;j < neuralNet.getNumberOfOutputs();j++){
				if(i == 0){
					testingOverallError.add(null);
				}
				testingError.get(i).add(null);
			}
		}
	}

	@Override
	public void train() throws NeuralException {
		if(neuralNet.getNumberOfHiddenLayers() > 0){
			throw new NeuralException("Delta rule can be used only with single layer neural network");
		}else{
			switch (learningMode) {
			case BATCH:
				epoch = 0;
				forward();
				if(printTraining){
					print();
				}
				while(epoch < maxEpoch && overallGeneralError > minOverallError){
					epoch++;
					for(int j = 0;j < neuralNet.getNumberOfOutputs();j++){
						for(int i = 0;i < neuralNet.getNumberOfInputs();i++){
							newWeights.get(0).get(j).set(i, calcNewWeight(0, i, j));
						}
					}
					applyNewWeights();
					forward();
					if(printTraining){
						print();
					}
				}
				break;

			case ONLINE:
				epoch = 0;
				int k = 0;
				currentRecord = 0;
				forward(k);
				if(printTraining){
					print();
				}
				while(epoch < maxEpoch && overallGeneralError > minOverallError){
                    for(int j = 0;j < neuralNet.getNumberOfOutputs();j++){
                        for(int i = 0;i <= neuralNet.getNumberOfInputs();i++){
                            newWeights.get(0).get(j).set(i,calcNewWeight(0,i,j));
                        }
                    }   
                    applyNewWeights();
                    currentRecord=++k;
                    if(k>=trainingDataSet.numberOfRecords){
                        k=0;
                        currentRecord=0;
                        epoch++;
                    }
                    
                    forward(k);
                    if(printTraining){
                        print();
                    }                        
                }
				break;
			}
		}

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
		if(layer > 0){
			throw new NeuralException("Delta rule can be used only with single"
                    + " layer neural network");
		}else{
			Double deltaWeight = learningRate;
			Neuron currentNeuron = neuralNet.getOutputLayer().getNeuron(neuron);
			switch (learningMode) {
			case BATCH:
				ArrayList<Double> derivativeResult = currentNeuron
					.derivativeBatch(trainingDataSet.getArrayInputData(normalization));
				
				break;

			default:
				break;
			}
			return null;
		}
		
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
		switch (errorMeasurement) {
		case SimpleError:
			degreeGeneralError=1;
			break;
		case SquareError:
		case MSE:
			degreeGeneralError=2;
			break;
		}
		this.generalErrorMeasurement = errorMeasurement;
	}
	
	public void setOverallErrorMeasurement(ErrorMeasurement errorMeasurement){
		switch (errorMeasurement) {
		case SimpleError:
			degreeOverallError=1;
			break;
		case SquareError:
		case MSE:
			degreeOverallError=2;
			break;
		}
		this.overallErrorMeasurement = errorMeasurement;
	}
	
	public void applyNewWeights(){
		//TODO
	}
}
