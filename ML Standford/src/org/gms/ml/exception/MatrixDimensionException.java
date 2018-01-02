package org.gms.ml.exception;

@SuppressWarnings("serial")
public class MatrixDimensionException extends RuntimeException {

	public MatrixDimensionException(){
		super("Dimension incoreccta");
	}
	
	public MatrixDimensionException(int r1, int r2, int c1, int c2){
		super(String.format("Dimensiones incompatibles: %dx%d - %dx%d", r1,c1,r2,c2));
	}
}
