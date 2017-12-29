package org.gms.ml.logistic;

import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Fmincg {

	private final static double RHO = 0.01;
	private final static double SIG = 0.5;
	private final static double INT = 0.1;
	private final static double EXT = 3.0;
	private final static int MAX = 20;
	private final static int RATIO = 100;
	
	public static Object[] fmincg(RealMatrix init_theta, RealMatrix X, RealMatrix y, double lambda, int maxIter){
		
		String S = "Iteration ";
		int red = 1;
		int i = 0;
		boolean ls_failed = false;
		RealVector fX = MatrixUtils.createRealVector(new double[]{});
		Object[] rc = Prueba.lrCostFunction(init_theta, X, y, lambda);
		RealMatrix df1 = (RealMatrix)rc[1];
		double f1 = (Double)rc[0];
//		i = i + (length<0);                                            % count epochs?!
		RealMatrix s = df1.scalarMultiply(-1.0);
		double d1 = s.scalarMultiply(-1).transpose().multiply(s).getEntry(0, 0); 
		double z1 = 1.0 / (1.0 - d1);
		
		while(i < maxIter){
			i++;													  //i = i + (length>0);   % count iterations?!
			
			RealMatrix init_theta0 = init_theta.copy(), df0 = df1.copy();
			double f0 = f1;
			init_theta = init_theta.add(s.scalarMultiply(z1));
			rc = Prueba.lrCostFunction(init_theta, X, y, lambda);
//			i = i + (length<0);                                          % count epochs?!
			RealMatrix df2 = (RealMatrix)rc[1];
			double f2 = (Double)rc[0];
			double d2 = df2.transpose().multiply(s).getEntry(0, 0);
			double f3 = f1, d3 = d1, z3 = z1;
//			if length>0, M = MAX; else M = min(MAX, -length-i); end
			int M = MAX;
			boolean success = false;
			double limit = -1;
			while(true){
				double z2, A = 0, B = 0;
				while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (M > 0)) {
					limit = z1;
					if(f2 > f1){
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3*z3+f2-f3);
					}else{
						A = 6*(f2-f3)/z3+3*(d2+d3);
						B = 3*(f3-f2)-z3*(d3+2*d2);
						z2 = (Math.sqrt(B*B-A*d2*z3*z3) - B) / A;
					}
					if(Double.isNaN(z2) || Double.isInfinite(z2)){
						z2 = z3 / 2;
					}
					z2 = Double.max(Double.min(z2, INT*z3), (1-INT)*z3);
					z1 = z1 + z2;
					init_theta = init_theta.add(s.scalarMultiply(z2));
					rc = Prueba.lrCostFunction(init_theta, X, y, lambda);
					f2 = (Double)rc[0];
					d2 = df2.transpose().multiply(s).getEntry(0, 0);
					M--; //i = i + (length<0);                           % count epochs?!
					d2 = df2.transpose().multiply(s).getEntry(0, 0);
					z3 = z3 - z2;
				}
				if (f2 > f1+z1*RHO*d1 || d2 > -SIG*d1) 
					break;
				else if(d2 > SIG*d1){
					success = true;
					break;
				}else if(M == 0){
					break;
				}
				A = 6*(f2-f3)/z3+3*(d2+d3);
				B = 3*(f3-f2)-z3*(d3+2*d2);
				z2 = -d2*z3*z3/(B + Math.sqrt(B*B-A*d2*z3*z3));
				if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0){
					if(limit < -0.5){
						z2 = z1 * (EXT-1);
					}else{
						z2 = (limit-z1)/2; 
					}
				}else if((limit > -0.5) && (z2+z1 > limit)){
					 z2 = (limit-z1)/2; 
				}else if((limit < -0.5) && (z2+z1 > z1*EXT)){
					z2 = z1*(EXT-1.0); 
				}else if(z2 < -z3*INT){
					z2 = -z3*INT;
				}else if((limit > -0.5) && (z2 < (limit-z1)*(1.0-INT))){
					z2 = (limit-z1)*(1.0-INT);
				}
				f3 = f2; d3 = d2; z3 = -z2;
				z1 = z1 + z2;
				init_theta = init_theta.add(s.scalarMultiply(z2));
				rc = Prueba.lrCostFunction(init_theta, X, y, lambda);
				f2 = (Double)rc[0];
				d2 = df2.transpose().multiply(s).getEntry(0, 0);
				M--; //i = i + (length<0);                           % count epochs?!
				d2 = df2.transpose().multiply(s).getEntry(0, 0);
			}
			if(success){
				f1 = f2;
				fX = fX.append(f1);  
			}
			
		}
		return null;
	}
	
	
	private static RealMatrix scalarDiv(double numerator, RealMatrix denominator){
		RealMatrix result = denominator.copy();
		result.walkInRowOrder(new DefaultRealMatrixChangingVisitor() {
			@Override
			public double visit(int row, int column, double value) {
				return numerator/value;
			}
		});
		return result;
	}
}
