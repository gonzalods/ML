package org.gms.ml.logistic;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorFormat;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class Prueba {

	public static void main(String[] args) throws IOException {
		
		int input_layer_size = 400;
		int num_labels = 10;
		
		System.out.println("Loading and Visualizing Data ...");
		
		MatFileReader mfr = new MatFileReader(new File("ex3data1.mat"));
		MLDouble XMat = (MLDouble)mfr.getMLArray("X");
		MLDouble yMat = (MLDouble)mfr.getMLArray("y");
		
		double[][] Xd = XMat.getArray();
		double[][] yd = yMat.getArray();
		
		RealMatrix X = MatrixUtils.createRealMatrix(Xd);
		RealMatrix y = MatrixUtils.createRealMatrix(yd);
		
		int m = X.getRowDimension();
		
		RealMatrix theta_t = MatrixUtils.createColumnRealMatrix(new double[]{-2.0,-1.0,1.0,2.0});
		RealMatrix X_t = MatrixUtils.createRealMatrix(5, 4);
		double[] ones = new double[5];
		Arrays.fill(ones, 1.0);
		double[] data_t = new double[15];
		Arrays.fill(data_t, 1.0);
		Arrays.parallelPrefix(data_t, (left, rigth) -> {return left + rigth;});
		
		X_t.setSubMatrix(reshape(data_t, 5, 3).getData(), 0, 1);
		X_t = X_t.scalarMultiply(1.0/10.0);
		X_t.setColumn(0, ones);
		RealMatrix y_t = MatrixUtils.createColumnRealMatrix(new double[] {1,0,1,0,1});
		double lambda_t = 3;
		print(X_t);
		print(y_t);
		
		Object[] cost_and_grad = lrCostFunction(theta_t, X_t, y_t, lambda_t);
		fprintf("\nTesting lrCostFunction() with regularization");
		fprintf("\nCost: %f\n", cost_and_grad[0]);
		fprintf("Expected cost: 2.534819\n");
		fprintf("Gradients:\n");
		fprintf(" %f \n", (RealMatrix)cost_and_grad[1]);
		fprintf("Expected gradients:\n");
		fprintf(" 0.146561\n -0.548558\n 0.724722\n 1.398003\n");
	}

	private static void fprintf(String string, Object ... object) {
		System.out.printf(string, object);
		
	}
	
	public static RealMatrix reshape(double[] data, int rows, int cols){
		RealMatrix matrix = MatrixUtils.createRealMatrix(rows, cols);
		int ixdata = 0;
		for(int i = 0;i < cols;i++){
			for(int j = 0;j < rows;j++){
				matrix.setEntry(j, i, data[ixdata++]);
			}
		}
		return matrix;
	}
	
	public static Object[] lrCostFunction(RealMatrix theta, RealMatrix X, RealMatrix y, double lambda) {
		
		Object[] result = new Object[2];
		int m = y.getRowDimension();
		double J = 0;
		RealMatrix grad = MatrixUtils.createRealMatrix(theta.getRowDimension(),theta.getColumnDimension());
		
		RealMatrix Theta_X = X.multiply(theta);
		RealMatrix y_negative = y.scalarMultiply(-1);
		RealMatrix one_minus_y = y.scalarMultiply(-1).scalarAdd(1);
		RealMatrix one_minus_sigmoid = sigmoid(Theta_X).scalarMultiply(-1).scalarAdd(1);
		RealMatrix first_part = y_negative.transpose().multiply(log(sigmoid(Theta_X)));
		RealMatrix second_part = one_minus_y.transpose().multiply(log(one_minus_sigmoid));
		double sum_regu = lambda / (2 * m) * sum(power(theta.getSubMatrix(1, theta.getRowDimension()-1, 0, theta.getColumnDimension()-1)));
		
		J = (1.0 / m) * sum(first_part.subtract(second_part)) + sum_regu;
		result[0] = J;
		
		RealMatrix error = MatrixUtils.createRealMatrix(Theta_X.getRowDimension(), Theta_X.getColumnDimension());
		error.setRowMatrix(0, sigmoid(Theta_X.getSubMatrix(0, 0, 0, 0)).subtract(y.getSubMatrix(0, 0, 0, 0)));
		error.setSubMatrix(sigmoid(Theta_X.getSubMatrix(1, m-1, 0, 0)).subtract(y.getSubMatrix(1, m-1, 0, 0)).getData(), 1, 0);
		
		grad.setRowMatrix(0, X.transpose().getSubMatrix(0, 0, 0, X.getRowDimension()-1).multiply(error).scalarMultiply(1.0/m));
		RealMatrix grad_part = X.transpose().getSubMatrix(1, X.getColumnDimension()-1, 0, X.getRowDimension()-1).multiply(error).scalarMultiply(1.0/m);
		RealMatrix regu_part = theta.getSubMatrix(1, theta.getRowDimension()-1, 0, 0).scalarMultiply(lambda/m);
		grad.setSubMatrix(grad_part.add(regu_part).getData(),1,0);
		result[1] = grad;
		return result;
	}
		
	public static RealMatrix sigmoid(RealMatrix Z) {
		RealMatrix result = Z.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			@Override
			public double visit(int row, int column, double value) {
				return 1.0 / (1.0 + Math.exp(-value));
			}
		});
		return result;
	}
	
	public static RealMatrix log(RealMatrix Z) {
		RealMatrix result = Z.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			@Override
			public double visit(int row, int column, double value) {
				return Math.log(value);
			}
		});
		return result;
	}
	
	public static double sum(RealMatrix matrix) {
		double result = 0;
		result = matrix.walkInRowOrder(new DefaultRealMatrixPreservingVisitor() {
			double sum = 0.0;
			@Override
			public void visit(int row, int column, double value) {
				sum += value;
			}
			@Override
			public double end() {
				return sum;
			}
		});
		return result;
	}
	
	public static RealMatrix power(RealMatrix Z) {
		RealMatrix result = Z.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			@Override
			public double visit(int row, int column, double value) {
				return Math.pow(value, 2);
			}
		});
		return result;
	}
	
	public static void print(RealMatrix matrix) {
		RealMatrixFormat formater = new RealMatrixFormat("\n", "\n", "\t", "", "\n", "   ");
		System.out.println(formater.format(matrix));
		//System.out.println(MatrixUtils.OCTAVE_FORMAT.format(matrix));
	}
	
	private static void fprintf(String string, RealMatrix matrix) {
		double[][] data = matrix.getData();
		for(double[] rows:data) {
			for(double col:rows) {
				System.out.printf(string, col);
			}
		}
	}
}
