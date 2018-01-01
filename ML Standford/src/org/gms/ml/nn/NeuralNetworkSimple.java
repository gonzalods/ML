package org.gms.ml.nn;

import static org.gms.ml.utils.OutputUtils.*;

import java.io.File;
import java.io.IOException;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class NeuralNetworkSimple {

	public static void main(String[] args) throws IOException {
		
//		int input_layer_size  = 400;
//		int hidden_layer_size = 25;   
//		int num_labels = 10;       

		fprintf("Loading and Visualizing Data ...\n");
		
		MatFileReader mfr = new MatFileReader(new File("ex3data1.mat"));
		MLDouble XMat = (MLDouble) mfr.getMLArray("X");
		MLDouble yMat = (MLDouble) mfr.getMLArray("y");

		double[][] Xd = XMat.getArray();
		double[][] yd = yMat.getArray();

		RealMatrix X = MatrixUtils.createRealMatrix(Xd);
		final RealVector y = MatrixUtils.createRealMatrix(yd).getColumnVector(0);
		
		int m = X.getRowDimension();
		
		fprintf("\nLoading Saved Neural Network Parameters ...\n");
		mfr = new MatFileReader(new File("ex3weights.mat"));
		
		RealMatrix Theta1 = MatrixUtils.createRealMatrix(((MLDouble)mfr.getMLArray("Theta1")).getArray());
		RealMatrix Theta2 = MatrixUtils.createRealMatrix(((MLDouble)mfr.getMLArray("Theta2")).getArray());

		
		RealVector pred = predict(Theta1, Theta2, X);
		double mean = StatUtils.mean(compareVector(pred, y).toArray());
		fprintf("\nTraining Set Accuracy: %f\n", mean * 100);
		
	}

	private static RealVector predict(RealMatrix Theta1, RealMatrix Theta2, RealMatrix X) {
		int m = X.getRowDimension();
		int n = X.getColumnDimension();
		RealVector pred = MatrixUtils.createRealVector(new double[m]);
		
		RealMatrix A1 = MatrixUtils.createRealMatrix(m, n + 1);
		RealVector ones = A1.getColumnVector(0);
		ones.set(1.0);
		A1.setColumnVector(0, ones);
		A1.setSubMatrix(X.getData(), 0, 1);

		RealMatrix Sig = A1.multiply(Theta1.transpose());
		Sig.walkInColumnOrder(new SigmoidRealMatrixChangeVisitor());
		RealMatrix A2 = MatrixUtils.createRealMatrix(m, Sig.getColumnDimension() + 1);
		ones = A1.getColumnVector(0);
		ones.set(1.0);
		A2.setColumnVector(0, ones);
		A2.setSubMatrix(Sig.getData(), 0, 1);
		Sig = A2.multiply(Theta2.transpose());
		Sig.walkInColumnOrder(new SigmoidRealMatrixChangeVisitor());
		
		//print(Sig);
		
		for(int i = 0;i < m;i++) {
			pred.setEntry(i, Sig.getRowVector(i).getMaxIndex() + 1);
		}

		return pred;
	}

	public static class SigmoidRealMatrixChangeVisitor extends DefaultRealMatrixChangingVisitor{
		Sigmoid sigmoid = new Sigmoid();
		@Override
		public double visit(int row, int column, double value) {
			return sigmoid.value(value);
		}
	}
}
