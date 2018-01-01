package org.gms.ml.utils;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class AlgebraUtils{

	public static RealMatrix ones(int rows, int columns) {
		RealMatrix res = MatrixUtils.createRealMatrix(rows, columns);
		for(int i = 0;i < columns;i++) {
			res.setColumnVector(i, ones(rows));
		}
		return res;
	}
	
	public static RealVector ones(int dim) {
		RealVector res = MatrixUtils.createRealVector(new double[dim]);
		res.set(1.0);
		return res;
	}
	
	public static RealMatrix addColumToMatrix(RealVector v, RealMatrix M) {
		int m = M.getRowDimension();
		int n = M.getColumnDimension();
		RealMatrix X = MatrixUtils.createRealMatrix(m, n+1);
		X.setColumnVector(0, v);
		X.setSubMatrix(M.getData(), 0, 1);
		return X;
	}
	
}
