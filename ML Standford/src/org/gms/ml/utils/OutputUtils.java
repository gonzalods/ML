package org.gms.ml.utils;

import java.text.DecimalFormat;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixFormat;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

public class OutputUtils {

	private OutputUtils() {}
	
	public static void fprintf(String string, Object... object) {
		System.out.printf(string, object);
	}
	
	public static void print(RealMatrix matrix) {
		DecimalFormat fmt = new DecimalFormat(" 0.00000E00;-0.000000E00");
		RealMatrixFormat formater = new RealMatrixFormat("\n", "\n", "\t", "fin", "\n", "   ", fmt);
		System.out.println(formater.format(matrix));
		// System.out.println(MatrixUtils.OCTAVE_FORMAT.format(matrix));
	}

	public static void fprintf(String string, RealVector vector) {
		double[] data = vector.toArray();
		for (double row : data) {
			System.out.printf(string, row);
		}
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
}
