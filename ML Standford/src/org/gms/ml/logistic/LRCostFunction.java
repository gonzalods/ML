package org.gms.ml.logistic;

import org.apache.commons.math3.analysis.function.Log;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.gms.ml.CostFunction;

public class LRCostFunction implements CostFunction {

	private RealVector y;
	private RealMatrix X;
	private double lambda;
	
	public LRCostFunction(RealVector y, RealMatrix x, double lambda) {
		super();
		this.y = y;
		X = x;
		this.lambda = lambda;
	}

	@Override
	public Object[] execute(RealVector theta) {
		
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

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

}
