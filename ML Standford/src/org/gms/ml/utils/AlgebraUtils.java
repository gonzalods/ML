package org.gms.ml.utils;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.DefaultRealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.gms.ml.exception.MatrixDimensionException;

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
	
	public static RealMatrix createMatrixFromVectors(RealVector ... v) {
		int m = v[0].getDimension();
		RealMatrix X = MatrixUtils.createRealMatrix(m, v.length);
		for (int i = 0; i < v.length; i++) {
			X.setColumnVector(i, v[i]);
		}
		return X;
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
	
	public static RealMatrix power(RealMatrix Z, int exp) {
		RealMatrix result = Z.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			@Override
			public double visit(int row, int column, double value) {
				return Math.pow(value, exp);
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
	
	public static RealMatrix ebeProduct(final RealMatrix op1, final RealMatrix op2){
		int m1 = op1.getRowDimension();
		int n1 = op1.getColumnDimension();
		int m2 = op2.getRowDimension();
		int n2 = op2.getColumnDimension();
		if(m1 != m2 || n1 != n2){
			throw new MatrixDimensionException(m1, m1, n1, n2);
		}
		RealMatrix result = MatrixUtils.createRealMatrix(m1, n1);;
		result.walkInColumnOrder(new DefaultRealMatrixChangingVisitor(){
			@Override
			public double visit(int row, int column, double value) {
				return op1.getEntry(row, column) * op2.getEntry(row, column);
			}
		});
		return result;
	}
	public static RealVector compareEqVector(final RealVector a, final double valor) {
		RealVector result = MatrixUtils.createRealVector(new double[a.getDimension()]);
		result.walkInDefaultOrder(new RealVectorChangingVisitor() {
			@Override
			public double visit(int index, double value) {
				return a.getEntry(index) == valor?1.0:0.0 ;
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
	
	public static RealVector compareEqVectors(final RealVector a, final RealVector b) {
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
	
	public static RealVector rollMatrixs(RealMatrix ...matrixs){
		int dim = 0;
		for (int i = 0; i < matrixs.length; i++) {
			dim += matrixs[i].getRowDimension() * matrixs[i].getColumnDimension();
		}
		double[] vector =new double[dim];
		int count = 0;
		for (int i = 0; i < matrixs.length; i++) {
			RealMatrix matrix = matrixs[i];
			for(int j = 0;j < matrix.getRowDimension();j++){
				double[] row = matrix.getRowVector(j).toArray();
				System.arraycopy(row, 0, vector, count, row.length);
				count += row.length;
			}
		}
		
		return MatrixUtils.createRealVector(vector);
		
	}
}
