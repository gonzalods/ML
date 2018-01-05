package org.gms.ml.nn;

import static org.gms.ml.utils.AlgebraUtils.*;
import static org.gms.ml.utils.OutputUtils.*;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.NormalizedRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.RandomGeneratorFactory;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class BackPropagation {

	private static RandomGenerator rg = RandomGeneratorFactory.createRandomGenerator(new Random());
	public static void main(String[] args) throws IOException {
		
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
		
		System.out.printf("Dimensiones Theta1 %dx%d%n",Theta1.getRowDimension(), Theta1.getColumnDimension());
		System.out.printf("Dimensiones Theta2 %dx%d%n",Theta2.getRowDimension(), Theta2.getColumnDimension());
		
//		RealVector theta_all = rollMatrixs(Theta1, Theta2);
//		Object[] res = nnCostFunction(theta_all, X, y, 2);
//		
//		fprintf("Función de coste: %.6e%n", res[0]);
//		
//		double[] gradAll = ((RealVector)res[1]).toArray();
//		RealMatrix Grad1 = reshape(Arrays.copyOfRange(gradAll, 0, 10025), 25, 401);
//		RealMatrix Grad2 = reshape(Arrays.copyOfRange(gradAll, 10025, gradAll.length+1), 10, 26);
//		System.out.println("Grad1");
//		fprintf(Grad1);
//		System.out.println("Theta1");
//		fprintf(Theta1);
//		System.out.println("Grad2");
//		fprintf(Grad2);
//		System.out.println("Theta2");
//		fprintf(Theta2);
		
//		generateRandomMatrix(Theta1);
//		generateRandomMatrix(Theta2);
		
		RealVector theta_all = rollMatrixs(Theta1, Theta2);
		Object[] res = nnCostFunction(theta_all, X, y, 2);
		fprintf("Función de coste: %.6e%n", res[0]);
		
		res = nnCostFunction((RealVector)res[1], X, y, 2);
		fprintf("Función de coste: %.6e%n", res[0]);
		
//		double[] gradAll = ((RealVector)res[1]).toArray();
//		RealMatrix Grad1 = reshape(Arrays.copyOfRange(gradAll, 0, 10025), 25, 401);
//		RealMatrix Grad2 = reshape(Arrays.copyOfRange(gradAll, 10025, gradAll.length+1), 10, 26);
//		System.out.println("Grad1");
//		fprintf(Grad1);
//		System.out.println("Theta1");
//		fprintf(Theta1);
//		System.out.println("Grad2");
//		fprintf(Grad2);
//		System.out.println("Theta2");
//		fprintf(Theta2);
		
	}
	
	private static void generateRandomMatrix(RealMatrix M){
		final NormalizedRandomGenerator nrg = new GaussianRandomGenerator(rg);
		M.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
			@Override
			public double visit(int row, int column, double value) {
				return nrg.nextNormalizedDouble();
			}
		});
	}
	public static Object[] nnCostFunction(RealVector theta, RealMatrix X, RealVector y, double lambda){
		int m = y.getDimension();
		double J = 0;
		RealVector grad = null;
		
		RealMatrix Theta1 = reshape(Arrays.copyOfRange(theta.toArray(), 0, 10025), 25, 401); 
		RealMatrix Theta2 = reshape(Arrays.copyOfRange(theta.toArray(), 10025, theta.toArray().length+1), 10, 26);
		
		RealMatrix A1 = addColumToMatrix(ones(m), X); 				// A1 = mx401
		RealMatrix A2 = sigmoid(A1.multiply(Theta1.transpose()));	// mx401 * 401x25 = mx25
		A2 = addColumToMatrix(ones(m), A2);							// A2 = mx26
		RealMatrix H = sigmoid(A2.multiply(Theta2.transpose()));	// H = mx26 * 26x10 = mx10
//		int k = H.getColumnDimension();
		
//		
		RealMatrix Y = createMatrixFromVectors(compareEqVector(y, 1),compareEqVector(y, 2),compareEqVector(y, 3),
				compareEqVector(y, 4),compareEqVector(y, 5),compareEqVector(y, 6),compareEqVector(y, 7),compareEqVector(y, 8),
				compareEqVector(y, 9),compareEqVector(y, 10));
		
		J = jota(Theta1, Theta2, H, Y, lambda, m);
		
		RealMatrix Delta1 = MatrixUtils.createRealMatrix(Theta1.getRowDimension(), Theta1.getColumnDimension()); // 25x401
		RealMatrix Delta2 = MatrixUtils.createRealMatrix(Theta2.getRowDimension(), Theta2.getColumnDimension()); // 10x26

		for(int i = 0;i < m;i++){
			RealMatrix a1i = A1.getRowMatrix(i).transpose();								// a1i		= 401x1
			final RealMatrix a2i = A2.getRowMatrix(i).transpose();							// a2i 		= 26x1
			RealMatrix hi = H.getRowMatrix(i).transpose();									// hi  		= 10x1
			
			RealMatrix deltaih = hi.subtract(Y.getRowMatrix(i).transpose());				
//                      10x1   = 10x1   -       1x10   ->         10x1
			RealMatrix deltai2 = Theta2.transpose().multiply(deltaih);
//			     	    26x1   = 10x26 -> 26x10	       *     10x1
			deltai2.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
				@Override
				public double visit(int row, int column, double value) {
					double result = value;
					if(column != 0){
						double av = a2i.getEntry(row, column);
						result *= (av * (1 - av));
					}
					return result;
				}
			});
			Delta1 = Delta1.add(deltai2.getSubMatrix(1, deltai2.getRowDimension()-1, 0, deltai2.getColumnDimension()-1)
																	.multiply(a1i.transpose()));
//          25x401   25x401	 +	 			25x1                        *    401x1 -> 1x401
			Delta2 = Delta2.add(deltaih.multiply(a2i.transpose()));
//          10x26  = 10x26       10x1      *    26x1 -> 1x26
		}
		
		RealMatrix D1 = MatrixUtils.createRealMatrix(Theta1.getRowDimension(), Theta1.getColumnDimension());
		RealMatrix D2 = MatrixUtils.createRealMatrix(Theta2.getRowDimension(), Theta2.getColumnDimension());
		
		
		RealMatrix SumPart = Delta1.getSubMatrix(0, Delta1.getRowDimension()-1, 1, Delta1.getColumnDimension()-1)
				.add(Theta1.getSubMatrix(0, Theta1.getRowDimension()-1, 1, Theta1.getColumnDimension()-1).scalarMultiply(lambda));
		D1.setColumnVector(0, Delta1.getColumnVector(0).mapMultiply(1.0/m));
		D1.setSubMatrix(SumPart.scalarMultiply(1.0/m).getData(), 0, 1);
		
		SumPart = Delta2.getSubMatrix(0, Delta2.getRowDimension()-1, 1, Delta2.getColumnDimension()-1)
				.add(Theta2.getSubMatrix(0, Theta2.getRowDimension()-1, 1, Theta2.getColumnDimension()-1).scalarMultiply(lambda));
		D2.setColumnVector(0, Delta2.getColumnVector(0).mapMultiplyToSelf(1.0/m));
		D2.setSubMatrix(SumPart.scalarMultiply(1.0/m).getData(), 0, 1);
		
		grad = rollMatrixs(D1,D2);
		return new Object[]{J, grad};
	}
	
	private static double jota(RealMatrix Theta1, RealMatrix Theta2, RealMatrix H, RealMatrix Y, double lambda, int m){
		
		double sum_reg = sum(power(Theta1.getSubMatrix(0, Theta1.getRowDimension()-1, 1, Theta1.getColumnDimension()-1),2));
			   sum_reg += sum(power(Theta2.getSubMatrix(0, Theta2.getRowDimension()-1, 1, Theta2.getColumnDimension()-1),2));
		
		sum_reg = lambda / (2.0 * m) * sum_reg;
		
		RealMatrix Y_negative = Y.scalarMultiply(-1.0);
		RealMatrix Ones_minus_y = Y_negative.scalarAdd(1.0); 
		RealMatrix Ones_minus_sigmoid = sigmoid(H).scalarMultiply(-1.0).scalarAdd(1.0);
		RealMatrix First_part = ebeProduct(Y_negative, log(H));
		RealMatrix Second_part = ebeProduct(Ones_minus_y, log(Ones_minus_sigmoid));
		
		return (1.0 / m) * sum(First_part.subtract(Second_part)) + sum_reg;
	}
	private static RealVector gradientChecking(RealVector theta,RealMatrix H, RealMatrix Y, double lambda, int m){
		double epsilon = 1e-4;
		RealVector grad = MatrixUtils.createRealVector(new double[theta.getDimension()]);
		
		RealMatrix Theta1 = reshape(Arrays.copyOfRange(theta.toArray(), 0, 10025), 25, 401); 
		RealMatrix Theta2 = reshape(Arrays.copyOfRange(theta.toArray(), 10025, theta.toArray().length+1), 10, 26);
		
		RealMatrix Theta_plus = Theta1.scalarAdd(epsilon);
		RealMatrix Theta_minus = Theta1.scalarAdd(-epsilon);
//		grad() = 
		
		return null;
	}

}
