package org.gms.ml.nn;

import static org.gms.ml.utils.AlgebraUtils.*;
import static org.gms.ml.utils.OutputUtils.*;

import java.io.File;
import java.io.IOException;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLDouble;

public class BackPropagation {

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
		RealVector theta_all = rollMatrixs(Theta1, Theta2);
		
		Object[] res = nnCostFunction(theta_all, X, y, 2);
		
		fprintf("Función de coste: %.6e%n", res[0]);
		
	}
	
	public static Object[] nnCostFunction(RealVector theta, RealMatrix X, RealVector y, double lambda){
		int m = y.getDimension();
		double J = 0;
		RealVector grad = MatrixUtils.createRealVector(new double[theta.getDimension()]);
		
		RealMatrix Theta1 = reshape(theta.toArray(), 25, 401); 
		RealMatrix Theta2 = reshape(theta.toArray(), 10, 26);
//		RealMatrix Theta3 = reshape(theta.toArray(), 10, 11);
		
		RealMatrix A1 = addColumToMatrix(ones(m), X); 				// A1 = mx401
		RealMatrix A2 = sigmoid(A1.multiply(Theta1.transpose()));	// mx401 * 401x25 = mx25
		A2 = addColumToMatrix(ones(m), A2);							// A2 = mx26
//		RealMatrix A3 = sigmoid(A2.multiply(Theta2.transpose()));
//		A3 = addColumToMatrix(ones(m), A3);
//		RealMatrix H = sigmoid(A3.multiply(Theta3.transpose()));
		RealMatrix H = sigmoid(A2.multiply(Theta2.transpose()));	// H = mx26 * 26x10 = mx10
		int k = H.getColumnDimension();
		
		double sum_reg = sum(power(Theta1.getSubMatrix(0, Theta1.getRowDimension()-1, 1, Theta1.getColumnDimension()-2),2));
		sum_reg += sum(power(Theta2.getSubMatrix(0, Theta2.getRowDimension()-1, 1, Theta2.getColumnDimension()-2),2));
//		sum_reg += sum(power(Theta3.getSubMatrix(0, Theta3.getRowDimension(), 1, Theta3.getColumnDimension()),2));
		
		sum_reg = lambda / (2 * m) * sum_reg;
		
		RealMatrix Y = createMatrixFromVectors(compareEqVector(y, 1),compareEqVector(y, 2),compareEqVector(y, 3),
				compareEqVector(y, 4),compareEqVector(y, 5),compareEqVector(y, 6),compareEqVector(y, 7),compareEqVector(y, 8),
				compareEqVector(y, 9),compareEqVector(y, 10));
		RealMatrix Y_negative = Y.scalarMultiply(-1.0);
		RealMatrix Ones_minus_y = Y_negative.scalarAdd(1.0); 
		RealMatrix Ones_minus_sigmoid = sigmoid(H).scalarMultiply(-1.0).scalarAdd(1.0);
		RealMatrix First_part = ebeProduct(Y_negative, log(H));
		RealMatrix Second_part = ebeProduct(Ones_minus_y, log(Ones_minus_sigmoid));
		
		J = (1 / m) * sum(First_part.subtract(Second_part)) + sum_reg;
		
		RealMatrix Delta1 = MatrixUtils.createRealMatrix(Theta1.getRowDimension(), Theta1.getColumnDimension()); // 25x401
		RealMatrix Delta2 = MatrixUtils.createRealMatrix(Theta2.getRowDimension(), Theta2.getColumnDimension()); // 10x26
		RealMatrix Delta3 = MatrixUtils.createRealMatrix(k, 1); // 10x1
		for(int i = 0;i < m;i++){
//			RealVector a1i = A1.getRowVector(i).getSubVector(1, A1.getColumnDimension()-2);
			RealVector a2i = A2.getRowVector(i).getSubVector(1, A2.getColumnDimension()-2); // a2i 		= 1x25
			RealVector hi = H.getRowVector(i);												// hi  		= 1x10
			RealVector deltaih = hi.subtract(Y.getRowVector(i));							// deltaih 	= 1x10 - 1x10 = 1x10
			RealVector deltai2 = Theta2.getSubMatrix(0, Theta2.getRowDimension()-1, 1, Theta2.getColumnDimension()-2) 
					.transpose().operate(deltaih).ebeMultiply(a2i).ebeMultiply(a2i.map((x) -> 1 - x));
//			RealVector deltai1 = Theta1.getSubMatrix(0, Theta1.getRowDimension()-1, 1, Theta2.getColumnDimension()-2) 
//					.transpose().operate(deltaih).ebeMultiply(a1i).ebeMultiply(a1i.map((x) -> 1 - x));
			System.out.println(deltai2);
//			Delta1 = Delta1.add(A1.getRowVector(i).ebeDivide(deltai2));
		}
		
		return new Object[]{J, grad};
	}

}
