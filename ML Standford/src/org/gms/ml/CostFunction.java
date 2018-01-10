package org.gms.ml;

import org.apache.commons.math3.linear.RealVector;

public interface CostFunction {

	Object[] execute(RealVector theta);
}
