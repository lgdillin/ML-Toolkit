// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

/// Replaces missing contiguous values with the mean
/// Replaces missing categorical values with the most-common value (or mode)
public class Imputer extends PreprocessingOperation
{
	private double[] m_centroid;
	private Matrix m_template = new Matrix();

	/// Creates a new Imputer instance
	public Imputer() {}

	/// Calculates the mean or mode for each column as appropriate.
	public void train(Matrix data) {
		m_centroid = new double[data.cols()];
		m_template.copyMetaData(data);

		for (int i = 0; i < data.cols(); i++) {
			if (data.valueCount(i) == 0)
				m_centroid[i] = data.columnMean(i);
			else
				m_centroid[i] = data.mostCommonValue(i);

			if(m_centroid[i] != m_centroid[i]) // if the centroid is NaN
				m_centroid[i] = 0.0;
		}
	}

	/// Returns an empty matrix that has the necessary meta-data.
	public Matrix outputTemplate() { return m_template; }

	/// Replaces unknown values in the input vector with the mean or mode, as appropriate.
	public void transform(double[] in, double[] out) {
		if(m_centroid == null)
			System.out.println("null");
		if (in.length != m_centroid.length)
			throw new RuntimeException("Imputer.transform received unexpected in-vector size. Expected " + m_centroid.length + ", got" + in.length);

		for (int i = 0; i < m_centroid.length; i++) {
			if (in[i] == Matrix.UNKNOWN_VALUE)
				out[i] = m_centroid[i];
			else
				out[i] = in[i];
		}
	}

	/// Unknown values cannot be recovered, so the input vector is simply copied into the output vector.
	public void untransform(double[] in, double[] out) {
		if (in.length != m_centroid.length)
			throw new RuntimeException("Imputer.untransform received unexpected in-vector size. Expected " + m_centroid.length + ", got" + in.length);

		for (int i = 0; i < m_centroid.length; i++)
			out[i] = in[i];
	}
}
