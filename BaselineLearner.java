// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
import java.util.Random;

class BaselineLearner extends SupervisedLearner
{
	double[] mode;

	BaselineLearner(Random r) { super(r); }

	String name()
	{
		return "Baseline";
	}

	void train(Matrix features, Matrix labels, int[] indices, int batch_size, double momentum)
	{
		mode = new double[labels.cols()];
		for(int i = 0; i < labels.cols(); i++)
		{
			if(labels.valueCount(i) == 0)
				mode[i] = labels.columnMean(i);
			else
				mode[i] = labels.mostCommonValue(i);
		}
	}

	Vec predict(Vec in)
	{
		return new Vec(mode);
	}
}
