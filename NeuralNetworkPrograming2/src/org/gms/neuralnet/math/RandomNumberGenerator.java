package org.gms.neuralnet.math;

import java.util.Random;

public class RandomNumberGenerator {

	public static int seed = 0;
	
	public static Random r;
	
	public static double GenerateNext() {
		if(r == null) {
			r = new Random(seed);
		}
		return r.nextDouble();
	}
}
