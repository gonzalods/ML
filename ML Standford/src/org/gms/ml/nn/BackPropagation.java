package org.gms.ml.nn;

import static org.gms.ml.utils.AlgebraUtils.*;
import static org.gms.ml.utils.OutputUtils.*;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.apache.commons.math3.stat.StatUtils;
import org.gms.ml.CostFunction;
import org.gms.ml.optimization.Fmincg;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class BackPropagation {

	public static void main(String[] args) throws IOException {
		
		int input_layer_size  = 400;  
		int hidden_layer_size = 25;
		int num_labels = 10;
		double lambda = 0.0;
		
		fprintf("Loading and Visualizing Data ...\n");
		
		MatFileReader mfr = new MatFileReader(new File("ex3data1.mat"));
		MLDouble XMat = (MLDouble) mfr.getMLArray("X");
		MLDouble yMat = (MLDouble) mfr.getMLArray("y");

		double[][] Xd = XMat.getArray();
		double[][] yd = yMat.getArray();

		RealMatrix X = MatrixUtils.createRealMatrix(Xd);
		final RealVector y = MatrixUtils.createRealMatrix(yd).getColumnVector(0);
		
		mfr = new MatFileReader(new File("ex3weights.mat"));
		RealMatrix Theta1 = MatrixUtils.createRealMatrix(((MLDouble)mfr.getMLArray("Theta1")).getArray());
		RealMatrix Theta2 = MatrixUtils.createRealMatrix(((MLDouble)mfr.getMLArray("Theta2")).getArray());
		
		fprintf("\nFeedforward Using Neural Network ...\n");
		RealVector nn_params = rollMatrixs(Theta1, Theta2);

		NNCostFunction nnCostFunction = new NNCostFunction(input_layer_size, hidden_layer_size, num_labels, y, X, lambda);
		Object[] res = nnCostFunction.execute(nn_params);
		fprintf("Cost at parameters (loaded from ex4weights): %f " +
		         "\n(this value should be about 0.287629)\n", res[0]);
		
		fprintf("\nChecking Cost Function (w/ Regularization) ... \n");

		lambda = 1.0;
		nnCostFunction.setLambda(lambda);
		res = nnCostFunction.execute(nn_params);
		fprintf("Cost at parameters (loaded from ex4weights): %f " +
		         "\n(this value should be about 0.383770)\n", res[0]);
		
		fprintf("\nEvaluating sigmoid gradient...\n");
		RealVector g = sigmoidGradient(MatrixUtils.createRealVector(new double[]{-1.0, -0.5, 0.0, 0.5, 1.0}));
		fprintf("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ");
		fprintf("%f ", g);
		fprintf("\n\n");
		
		fprintf("\nInitializing Neural Network Parameters ...\n");
		Theta1 = randInitializeWeights(25, 400);
		Theta2 = randInitializeWeights(10, 25);
		
		RealVector initial_nn_params = rollMatrixs(Theta1, Theta2);
		
		fprintf("\nChecking Backpropagation... \n");
		checkNNGradients();
		
		fprintf("\nChecking Backpropagation (w/ Regularization) ... \n");
		checkNNGradients(3);
		
		lambda = 3.0;
		nnCostFunction.setLambda(lambda);
		res  = nnCostFunction.execute(nn_params);

		fprintf("\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f " +
		"\n(for lambda = 3, this value should be about 0.576051)\n\n", lambda, res[0]);
		
		fprintf("\nTraining Neural Network... \n");

		lambda = 1;
		nnCostFunction.setLambda(lambda);
		res = Fmincg.fmincg(nnCostFunction, initial_nn_params, 50);
		nn_params =  (RealVector)res[0];

		Theta1 = reshape(Arrays.copyOfRange(nn_params.toArray(), 0, hidden_layer_size * (input_layer_size +1)), 
				hidden_layer_size, input_layer_size + 1);
		
		Theta2 = reshape(Arrays.copyOfRange(nn_params.toArray(), hidden_layer_size * (input_layer_size + 1), nn_params.toArray().length+1), 
				num_labels, hidden_layer_size + 1);
		
		RealVector pred = predict(Theta1, Theta2, X);

		double mean = StatUtils.mean(compareEqVectors(pred, y).toArray());
		fprintf("\nTraining Set Accuracy: %f\n", mean * 100);
		
	}
	
	private static RealMatrix randInitializeWeights(int layerOut, int layerIn){
		final RandomDataGenerator nrg = new RandomDataGenerator();
		double epsilonInit = 0.12;
		RealMatrix result = MatrixUtils.createRealMatrix(layerOut, layerIn + 1);
		result.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
			@Override
			public double visit(int row, int column, double value) {
				return nrg.nextUniform(0, 1) * 2 * (epsilonInit) - epsilonInit;
			}
		});
		return result;
	}
	
	public static RealMatrix debugInitializeWeights(int layerOut, int layerIn){
		final Sin sin = new Sin();
		RealMatrix result = MatrixUtils.createRealMatrix(layerOut, layerIn + 1);
		result.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
			int count = 0;
			@Override
			public double visit(int row, int column, double value) {
				return sin.value(count++) / 10;
			}
		});
		return result;
	}

	public static RealVector sigmoidGradient(RealVector z){
		RealVector result = z.copy();
		final Sigmoid sigmoid = new Sigmoid();
		result = result.mapToSelf((x) -> sigmoid.value(x) * (1 - sigmoid.value(x)));
		return result;
	}
	
//	public static RealMatrix sigmoidGradient(RealMatrix Z){
//		RealMatrix result = Z.copy();
//		result.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
//			Sigmoid sigmoid = new Sigmoid();
//			@Override
//			public double visit(int row, int column, double value) {
//				return sigmoid.value(value);
//			}
//		});
//		return result;
//	}
	
	public static void checkNNGradients(double ... lambdas){
		double lambda = 0.0;
		if(lambdas != null && lambdas.length != 0){
			lambda = lambdas[0];
		}
		
		int input_layer_size = 3;
		int hidden_layer_size = 5;
		int num_labels = 3;
		int m = 5;
		
		RealMatrix Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
		RealMatrix Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
		
		RealMatrix X  = debugInitializeWeights(m, input_layer_size - 1);
		RealMatrix y = MatrixUtils.createColumnRealMatrix(new double[m]);
		y.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
			@Override
			public double visit(int row, int column, double value) {
				return 1 + (column % num_labels);
			}
		});
		
		RealVector theta_all = rollMatrixs(Theta1, Theta2);
		NNCostFunction nnCostFunction = new NNCostFunction(input_layer_size, hidden_layer_size, num_labels, y.getColumnVector(0), X, lambda);
		Object[] res = nnCostFunction.execute(theta_all);
		RealVector grad = (RealVector)res[1];
		RealVector numgrad = computeNumericalGradient(nnCostFunction, theta_all);
		for(int i = 0;i < grad.getDimension();i++){
			System.out.printf("\t%f\t%f%n", numgrad.getEntry(i), grad.getEntry(i));
		}
		fprintf("The above two columns you get should be very similar.\n" +
		         "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n");
		
		double diff = numgrad.subtract(grad).getL1Norm() / numgrad.add(grad).getL1Norm();

		fprintf("If your backpropagation implementation is correct, then \n" +
		         "the relative difference will be small (less than 1e-9). \n" +
		         "\nRelative Difference: %g\n", diff);
	}
	
	public static RealVector computeNumericalGradient(CostFunction nnCostFunction, RealVector theta){
		
		RealVector numgrad = MatrixUtils.createRealVector(new double[theta.getDimension()]);
		RealVector perturb = MatrixUtils.createRealVector(new double[theta.getDimension()]);
		int numele = theta.getDimension();
		double e = 1e-4;
		for(int p = 0;p < numele;p++){
			perturb.setEntry(p, e);
			double loss1 = (Double) nnCostFunction.execute(theta.subtract(perturb))[0];
			double loss2 = (Double) nnCostFunction.execute(theta.add(perturb))[0];
			numgrad.setEntry(p, ((loss2 - loss1) / (2 * e)));
			perturb.setEntry(p, 0.0);
		}
		return numgrad;
	}
	
	public static RealVector predict(RealMatrix Theta1, RealMatrix Theta2, RealMatrix X){
		
		int m = X.getRowDimension();

		RealVector pred = MatrixUtils.createRealVector(new double[m]);

		RealMatrix A1 = addColumToMatrix(ones(m), X).multiply(Theta1.transpose());
		RealMatrixChangingVisitor sigmoidChangingVisitor = new DefaultRealMatrixChangingVisitor(){
			Sigmoid sigmoid = new Sigmoid();
			@Override
			public double visit(int row, int column, double value) {
				return sigmoid.value(value);
			}
		}; 
		A1.walkInColumnOrder(sigmoidChangingVisitor);
		RealMatrix H = addColumToMatrix(ones(m), A1).multiply(Theta2.transpose());
		H.walkInColumnOrder(sigmoidChangingVisitor);
		for(int i = 0;i < m;i++) {
			pred.setEntry(i, H.getRowVector(i).getMaxIndex() + 1);
		}
		
		return pred;
	}
}
