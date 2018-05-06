// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

/// Replaces nominal values with categorical distributions in multiple dimensions, and vice-versa.
public class NomCat extends PreprocessingOperation
{
	private int[] m_vals;
	private Matrix m_template = new Matrix();

	/// Creates a new NomCat instance
	public NomCat() {}

	/// Decide how many dims are needed for each column
	@Override
	public void train(Matrix data) {
		int totalVals = 0;
		m_vals = new int[data.cols()];

		for (int i = 0; i < m_vals.length; i++) {
			int n = data.valueCount(i);
			if (n < 3)
				n = 1;
				
			m_vals[i] = n;
			totalVals += n;
		}

		m_template.setSize(0, totalVals);
	}

	/// Returns a zero-row matrix with the number of continuous columns that transform will output
	@Override
	public Matrix outputTemplate() {
		return m_template;
	}

	/// Re-represent each nominal attribute with a categorical distribution of continuous values
	@Override
	public void transform(double[] in, double[] out) {
		if (in.length != m_vals.length)
			throw new RuntimeException("NomCat.transform received unexpected in-vector size. Expected " + m_vals.length + ", got " + in.length);

		int outPos = 0;
		for (int i = 0; i < in.length; i++) {
			if (m_vals[i] == 1)
				out[outPos++] = in[i];
			else {

				int outStart = outPos;
				for (int j = 0; j < m_vals[i]; j++)
					out[outPos++] = 0.0;

				if (in[i] != Matrix.UNKNOWN_VALUE) {
					if (in[i] >= m_vals[i])
						throw new RuntimeException("Value out of range. Expected [0-"
							+ (m_vals[i] - 1) + "], got " + in[i]);

					out[(int)(outStart + in[i])] = 1.0;
				}
			}
		}
	}

	/// Re-encode categorical distributions as nominal values by finding the mode
	@Override
	public void untransform(double[] in, double[] out) {
		if (in.length != m_template.cols())
			throw new RuntimeException("NomCat.untransform received unexpected in-vector size. Expected " + m_template.cols() + ", got " + in.length);

		int inPos = 0;
		for (int i = 0; i < m_vals.length; i++) {
			if (m_vals[i] == 1)
				out[i] = in[inPos++];
			else {
				int startIn  = inPos;
				int maxIndex = 0;
				inPos++;

				for (int j = 1; j < m_vals[i]; j++) {
					if (in[inPos] > in[startIn + maxIndex])
						maxIndex = inPos - startIn;
					inPos++;
				}
				out[i] = (double)maxIndex;
			}
		}
	}
}
