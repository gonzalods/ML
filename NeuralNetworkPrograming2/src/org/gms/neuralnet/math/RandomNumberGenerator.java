package org.gms.neuralnet.math;

import java.util.Random;

public class RandomNumberGenerator {

	public static int seed = 0;
	
	public static double GenerateNext() {
		return new Random(seed).nextDouble();
	}
}
