package org.gms.ml.logistic;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;

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
		
		RealVector theta_t = MatrixUtils.createRealVector(new double[]{-2.0,-1.0,1.0,2.0});
		RealMatrix X_t = MatrixUtils.createRealMatrix(5, 4);
		double[] ones = new double[5];
		Arrays.fill(ones, 1.0);
		double[] data_t = new double[15];
		Arrays.fill(data_t, 1.0);
		Arrays.parallelPrefix(data_t, (left, rigth) -> {return left + rigth;});
		
		X_t.setColumn(0, ones);
		X_t.setSubMatrix(reshape(data_t, 5, 3).getData(), 0, 1);
		
		System.out.println(RealVectorFormat.getInstance().format(theta_t));
		System.out.println(MatrixUtils.OCTAVE_FORMAT.format(X_t));
		
//		System.out.printf("dimensions X: %d x %d%n", X.getRowDimension(), X.getColumnDimension());
//		System.out.printf("dimensions y: %d x %d%n", y.getRowDimension(), y.getColumnDimension());
		//System.out.printf("dimensions y: %d x %d%n", theta_t.getDimension(), y.getColumnDimension());
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
}
