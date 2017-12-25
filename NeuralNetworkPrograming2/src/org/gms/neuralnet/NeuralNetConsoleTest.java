package org.gms.neuralnet;

import java.util.Arrays;

import org.gms.neuralnet.math.IActivationFunction;
import org.gms.neuralnet.math.Linear;
import org.gms.neuralnet.math.RandomNumberGenerator;
import org.gms.neuralnet.math.Sigmoid;

public class NeuralNetConsoleTest {

	public static void main(String[] args) {
		RandomNumberGenerator.seed=0;
		
		int numberOfInputs=2;
	    int numberOfOutputs=1;
	    int[] numberOfHiddenNeurons= { 3 };
	    IActivationFunction[] hiddenAcFnc = { new Sigmoid(1.0) } ;
	    Linear outputAcFnc = new Linear(1.0);
	    System.out.println("Creating Neural Network...");
	    NeuralNet nn = new NeuralNet(numberOfInputs,numberOfOutputs,
	          numberOfHiddenNeurons,hiddenAcFnc,outputAcFnc);
	    System.out.println("Neural Network created!");
	    nn.print();
	    
	    Double [] neuralInput = { 1.5 , 0.5 };
	    Double [] neuralOutput;
	    System.out.println("Feeding the values ["+String.valueOf(neuralInput[0])+" ; "+
	                  String.valueOf(neuralInput[1])+"] to the neural network");
	    nn.setInput(neuralInput);
	    nn.calc();
	    neuralOutput=nn.getOutput();
	    
	    System.out.println("Output generated:" + Arrays.toString(neuralOutput));
	    
	    neuralInput[0] = 1.0;
	    neuralInput[1] = 2.1;

	    System.out.println("Feeding the values ["+String.valueOf(neuralInput[0])+" ; "+
                String.valueOf(neuralInput[1])+"] to the neural network");
	    nn.setInput(neuralInput);
	    nn.calc();
	    neuralOutput=nn.getOutput();
	    
	    System.out.println("Output generated:" + Arrays.toString(neuralOutput));
	}

}
