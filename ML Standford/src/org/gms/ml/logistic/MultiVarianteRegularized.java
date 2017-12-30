package org.gms.ml.logistic;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.stat.StatUtils;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class MultiVarianteRegularized {

	public static void main(String[] args) throws IOException {

		int num_labels = 10;

		System.out.println("Loading and Visualizing Data ...");

		MatFileReader mfr = new MatFileReader(new File("ex3data1.mat"));
		MLDouble XMat = (MLDouble) mfr.getMLArray("X");
		MLDouble yMat = (MLDouble) mfr.getMLArray("y");

		double[][] Xd = XMat.getArray();
		double[][] yd = yMat.getArray();

		RealMatrix X = MatrixUtils.createRealMatrix(Xd);
		final RealVector y = MatrixUtils.createRealMatrix(yd).getColumnVector(0);

		RealVector theta_t = MatrixUtils.createRealVector(new double[] { -2.0, -1.0, 1.0, 2.0 });
		RealMatrix X_t = MatrixUtils.createRealMatrix(5, 4);
		double[] ones = new double[5];
		Arrays.fill(ones, 1.0);
		double[] data_t = new double[15];
		Arrays.fill(data_t, 1.0);
		Arrays.parallelPrefix(data_t, (left, rigth) -> {
			return left + rigth;
		});

		X_t.setSubMatrix(reshape(data_t, 5, 3).getData(), 0, 1);
		X_t = X_t.scalarMultiply(1.0 / 10.0);
		X_t.setColumn(0, ones);
		RealVector y_t = MatrixUtils.createRealVector(new double[] { 1, 0, 1, 0, 1 });
		double lambda_t = 3;

		Object[] cost_and_grad = lrCostFunction(theta_t, X_t, y_t, lambda_t);
		fprintf("\nTesting lrCostFunction() with regularization");
		fprintf("\nCost: %f\n", cost_and_grad[0]);
		fprintf("Expected cost: 2.534819\n");
		fprintf("Gradients:\n");
		fprintf(" %f \n", (RealVector) cost_and_grad[1]);
		fprintf("Expected gradients:\n");
		fprintf(" 0.146561\n -0.548558\n 0.724722\n 1.398003\n");

		fprintf("\nTraining One-vs-All Logistic Regression...\n");
		double lambda = 0.1;

		Object[] rs = oneVsAll(X, y, num_labels, lambda);
		RealMatrix all_theta = (RealMatrix)rs[0];
		
		RealVector pred = predictOneVsAll(all_theta, X);

		double mean = StatUtils.mean(compareVector(pred, y).toArray());
		fprintf("\nTraining Set Accuracy: %f\n", mean * 100);
		

	}

	private static RealVector predictOneVsAll(RealMatrix all_theta, RealMatrix X) {
		int m = X.getRowDimension();
		int n = X.getColumnDimension();
		
		RealMatrix X_amp = MatrixUtils.createRealMatrix(m, n + 1);
		RealVector ones = X_amp.getColumnVector(0);
		ones.set(1.0);
		X_amp.setColumnVector(0, ones);
		X_amp.setSubMatrix(X.getData(), 0, 1);
		
		RealVector pred = MatrixUtils.createRealVector(new double[m]);
		RealMatrix X_all_theta = sigmoid(X_amp.multiply(all_theta.transpose()));
		for(int i = 0;i < m;i++) {
			pred.setEntry(i, X_all_theta.getRowVector(i).getMaxIndex() + 1);
		}
		
		return pred;
	}

	private static Object[] oneVsAll(RealMatrix Xini, RealVector y, int num_labels, double lambda) {
		int m = Xini.getRowDimension();
		int n = Xini.getColumnDimension();
		
		RealMatrix all_theta = MatrixUtils.createRealMatrix(num_labels, n + 1);
		RealVector ones = MatrixUtils.createRealVector(new double[m]);
		ones.set(1.0);
		RealMatrix X = MatrixUtils.createRealMatrix(m, n+1);
		X.setColumnVector(0, ones);
		X.setSubMatrix(Xini.getData(), 0, 1);
		
		for (int i = 1;i <= num_labels; i++) {
			final int ic = i;
			RealVector init_theta = MatrixUtils.createRealVector(new double[n + 1]);
			Object[] res = Fmincg.fmincg(init_theta, X, y.map((x -> x == ic?1:0)), lambda, 50);
			all_theta.setRowVector(i-1, (RealVector)res[0]);
//			all_theta(i,:) = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
//				                           initial_theta, options);
		}
		
		return new Object[] {all_theta};
	}

	public static void fprintf(String string, Object... object) {
		System.out.printf(string, object);

	}

	public static RealMatrix reshape(double[] data, int rows, int cols) {
		RealMatrix matrix = MatrixUtils.createRealMatrix(rows, cols);
		int ixdata = 0;
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) {
				matrix.setEntry(j, i, data[ixdata++]);
			}
		}
		return matrix;
	}

	public static Object[] lrCostFunction(RealVector theta, RealMatrix X, RealVector y, double lambda) {

		Object[] result = new Object[2];
		int m = y.getDimension();
		double J = 0;
		RealVector grad = MatrixUtils.createRealVector(new double[theta.getDimension()]);
		Sigmoid sigmoid = new Sigmoid();
		Log log = new Log();

		RealVector theta_X = X.operate(theta);
		RealVector y_negative = y.mapMultiply(-1);
		RealVector one_minus_y = y.map((x) -> 1 - x);
		RealVector one_minus_sigmoid = theta_X.map((x) -> 1 - sigmoid.value(x));
		double first_part = y_negative.dotProduct(theta_X.map((x) -> log.value(sigmoid.value(x))));
		double second_part = one_minus_y.dotProduct(one_minus_sigmoid.map((x) -> log.value(x)));
		double sum_regu = lambda / (2 * m) * theta.getSubVector(1, theta.getDimension() - 1)
				.dotProduct(theta.getSubVector(1, theta.getDimension() - 1));

		J = (1.0 / m) * (first_part - second_part) + sum_regu;
		result[0] = J;

		RealVector error = MatrixUtils.createRealVector(new double[theta_X.getDimension()]);
		error.setEntry(0, sigmoid.value(theta_X.getEntry(0)) - y.getEntry(0));
		error.setSubVector(1,
				theta_X.getSubVector(1, m - 1).map((x) -> sigmoid.value(x)).subtract(y.getSubVector(1, m - 1)));

		grad.setEntry(0,
				X.transpose().getSubMatrix(0, 0, 0, m - 1).operate(error).mapMultiplyToSelf(1.0 / m).getEntry(0));
		RealVector grad_part = X.transpose().getSubMatrix(1, X.getColumnDimension() - 1, 0, X.getRowDimension() - 1)
				.operate(error).mapMultiplyToSelf(1.0 / m);
		RealVector regu_part = theta.getSubVector(1, theta.getDimension() - 1).mapMultiplyToSelf(lambda / m);
		grad.setSubVector(1, grad_part.add(regu_part));
		result[1] = grad;

		return result;
	}

	public static RealVector compareVector(final RealVector a, final RealVector b) {
		RealVector result = MatrixUtils.createRealVector(new double[a.getDimension()]);
		result.walkInDefaultOrder(new RealVectorChangingVisitor() {
			@Override
			public double visit(int index, double value) {
				return a.getEntry(index) == b.getEntry(index)?1.0:0.0 ;
			}
			@Override
			public double end() {
				return 0;
			}
			@Override
			public void start(int arg0, int arg1, int arg2) {}
		});
	
		return result;
	}
	public static RealMatrix sigmoid(RealMatrix Z) {
		RealMatrix result = Z.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			Sigmoid sigmoid = new Sigmoid();
			@Override
			public double visit(int row, int column, double value) {
				return sigmoid.value(value);
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
		DecimalFormat fmt = new DecimalFormat("0.0#####");
		RealMatrixFormat formater = new RealMatrixFormat("\n", "\n", "\t", "", "\n", "   ", fmt);
		System.out.println(formater.format(matrix));
		// System.out.println(MatrixUtils.OCTAVE_FORMAT.format(matrix));
	}

	public static void fprintf(String string, RealVector vector) {
		double[] data = vector.toArray();
		for (double row : data) {
			System.out.printf(string, row);
		}
	}

	// private static void fprintf(String string, RealMatrix matrix) {
	// double[][] data = matrix.getData();
	// for(double[] rows:data) {
	// for(double col:rows) {
	// System.out.printf(string, col);
	// }
	// }
	// }
}
