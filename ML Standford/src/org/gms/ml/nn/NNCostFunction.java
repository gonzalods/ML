package org.gms.ml.nn;

import static org.gms.ml.utils.AlgebraUtils.addColumToMatrix;
import static org.gms.ml.utils.AlgebraUtils.compareEqVector;
import static org.gms.ml.utils.AlgebraUtils.createMatrixFromVectors;
import static org.gms.ml.utils.AlgebraUtils.ebeProduct;
import static org.gms.ml.utils.AlgebraUtils.log;
import static org.gms.ml.utils.AlgebraUtils.ones;
import static org.gms.ml.utils.AlgebraUtils.power;
import static org.gms.ml.utils.AlgebraUtils.reshape;
import static org.gms.ml.utils.AlgebraUtils.rollMatrixs;
import static org.gms.ml.utils.AlgebraUtils.sigmoid;
import static org.gms.ml.utils.AlgebraUtils.sum;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.gms.ml.CostFunction;

public class NNCostFunction implements CostFunction {

	private int numInputs;
	private int numNodesInHiddenLayer;
	private int numLabels;
	private RealVector y;
	private RealMatrix X;
	private double lambda;
	
	
	public NNCostFunction(int numInputs, int numNodesInHiddenLayer, int numLabels, RealVector y, RealMatrix x,
			double lambda) {
		super();
		this.numInputs = numInputs;
		this.numNodesInHiddenLayer = numNodesInHiddenLayer;
		this.numLabels = numLabels;
		this.y = y;
		X = x;
		this.lambda = lambda;
	}

	@Override
	public Object[] execute(RealVector theta) {
		int m = y.getDimension();
		double J = 0;
		RealVector grad = null;
		
		RealMatrix Theta1 = reshape(Arrays.copyOfRange(theta.toArray(), 0, numNodesInHiddenLayer * (numInputs +1))
				, numNodesInHiddenLayer, numInputs + 1); 
		RealMatrix Theta2 = reshape(Arrays.copyOfRange(theta.toArray(), numNodesInHiddenLayer * (numInputs + 1), theta.toArray().length+1), 
				numLabels, numNodesInHiddenLayer + 1);

		RealMatrix A1 = addColumToMatrix(ones(m), X); 				// A1 = mx401
		RealMatrix Z2 = A1.multiply(Theta1.transpose());
 		RealMatrix A2 = sigmoid(Z2);								// mx401 * 401x25 = mx25
		A2 = addColumToMatrix(ones(m), A2);							// A2 = mx26
		RealMatrix Z3 = A2.multiply(Theta2.transpose());
		RealMatrix H = sigmoid(Z3);									// H = mx26 * 26x10 = mx10
		
		
		List<RealVector> vectors = new ArrayList<>();
		for(int i = 0;i < numLabels;i++){
			vectors.add(compareEqVector(y, i + 1.0));
		}
		RealMatrix Y = createMatrixFromVectors(vectors.toArray(new RealVector[vectors.size()]));
		
		J = jota(Theta1, Theta2, H, Y, m);
		
		RealMatrix Delta1 = MatrixUtils.createRealMatrix(Theta1.getRowDimension(), Theta1.getColumnDimension()); // 25x401
		RealMatrix Delta2 = MatrixUtils.createRealMatrix(Theta2.getRowDimension(), Theta2.getColumnDimension()); // 10x26

		for(int i = 0;i < m;i++){
			RealMatrix a1i = A1.getRowMatrix(i).transpose();								// a1i		= 401x1
			final RealMatrix a2i = A2.getRowMatrix(i).transpose();							// a2i 		= 26x1
			RealMatrix hi = H.getRowMatrix(i).transpose();									// hi  		= 10x1
			
			RealMatrix deltaih = hi.subtract(Y.getRowMatrix(i).transpose());				
//                      10x1   = 10x1   -       1x10   ->         10x1
			RealMatrix deltai2 = Theta2.transpose().multiply(deltaih).getSubMatrix(1, Theta2.transpose().getRowDimension()-1, 0, deltaih.getColumnDimension()-1);
//			     	    25x1   = (10x26 -> 26x10	       *     10x1   )  -> 25x1 
			deltai2.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
				@Override
				public double visit(int row, int column, double value) {
					double result = value;
//					if(column != 0){
						double av = a2i.getEntry(row + 1, column);
						result *= (av * (1 - av));
//					}
					return result;
				}
			});
			Delta1 = Delta1.add(deltai2.multiply(a1i.transpose()));
//          25x401   25x401	 +	25x1       *    401x1 -> 1x401
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
	
	private double jota(RealMatrix Theta1, RealMatrix Theta2, RealMatrix H, RealMatrix Y, int m){
		
		double sum_reg = sum(power(Theta1.getSubMatrix(0, Theta1.getRowDimension()-1, 1, Theta1.getColumnDimension()-1),2));
			   sum_reg += sum(power(Theta2.getSubMatrix(0, Theta2.getRowDimension()-1, 1, Theta2.getColumnDimension()-1),2));
		
		sum_reg = lambda / (2.0 * m) * sum_reg;
		
		RealMatrix Y_negative = Y.scalarMultiply(-1.0);
		RealMatrix Ones_minus_y = Y_negative.scalarAdd(1.0); 
		RealMatrix Ones_minus_sigmoid = H.scalarMultiply(-1.0).scalarAdd(1.0);
		RealMatrix First_part = ebeProduct(Y_negative, log(H));
		RealMatrix Second_part = ebeProduct(Ones_minus_y, log(Ones_minus_sigmoid));
		
		return (1.0 / m) * sum(First_part.subtract(Second_part)) + sum_reg;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

}
