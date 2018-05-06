// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

/// Continuous elements are normalized to the range [0, 1] during processing
/// and returned to their original range afterwards.
public class Normalizer extends PreprocessingOperation
{
	private double[] m_inputMins;
	private double[] m_inputMaxs;
	private Matrix m_template = new Matrix();

	/// Creates a new Normalizer instance
	public Normalizer() {}

	/// Computers the min and max of each column
	@Override
	public void train(Matrix data) {
		m_template.copyMetaData(data);
		m_inputMins = new double[data.cols()];
		m_inputMaxs = new double[data.cols()];

		for(int i = 0; i < data.cols(); i++) {
			if(data.valueCount(i) == 0) {

				// Compute the min and max
				m_inputMins[i] = data.columnMin(i);
				m_inputMaxs[i] = Math.max(m_inputMins[i] + 1e-9, data.columnMax(i));
			} else {

				// Don't do nominal attributes
				m_inputMins[i] = Matrix.UNKNOWN_VALUE;
				m_inputMaxs[i] = Matrix.UNKNOWN_VALUE;
			}
		}
	}

	/// Returns a zero-row matrix with the same column meta-data as the one that was passed to train.
	@Override
	public Matrix outputTemplate() { return m_template; }

	/// Normalize continuous features.
	@Override
	public void transform(double[] in, double[] out) {
		if(m_inputMins == null)
			throw new RuntimeException("Tried to use a Normalizer transform that had not been trained");
		if(in.length != m_inputMins.length)
			throw new RuntimeException("Normalizer.transform received unexpected in-vector size. Expected " + m_inputMins.length + ", got " + in.length);

		for(int c = 0; c < in.length; c++) {
			if(m_inputMins[c] == Matrix.UNKNOWN_VALUE) // if the attribute is nominal...
				out[c] = in[c];
			else {
				if(in[c] == Matrix.UNKNOWN_VALUE)
					out[c] = Matrix.UNKNOWN_VALUE;
				else
					out[c] = (in[c] - m_inputMins[c]) / (m_inputMaxs[c] - m_inputMins[c]);
			}
		}
	}

	/// De-normalize continuous values.
	@Override
	public void untransform(double[] in, double[] out) {
		if(m_inputMins == null)
			throw new RuntimeException("Tried to use a Normalizer transform that had not been trained");
		if(in.length != m_inputMins.length)
			throw new RuntimeException("Normalizer.untransform received unexpected in-vector size. Expected " + m_inputMins.length + ", got " + in.length);

		for(int c = 0; c < in.length; c++) {
			if(m_inputMins[c] == Matrix.UNKNOWN_VALUE) // if the attribute is nominal...
				out[c] = in[c];
			else {
				if(in[c] == Matrix.UNKNOWN_VALUE)
					out[c] = Matrix.UNKNOWN_VALUE;
				else
					out[c] = in[c] * (m_inputMaxs[c] - m_inputMins[c]) + m_inputMins[c];
			}
		}
	}
}
