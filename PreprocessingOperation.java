// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

abstract public class PreprocessingOperation 
{
	/// Trains this preprocessing operation
	abstract void train(Matrix data);
	
	/// Returns an example of an output matrix. The meta-data of this matrix
	///shows how output will be given. (This matrix contains no data because it has zero rows.
	///It is only used for the meta-data.
	abstract Matrix outputTemplate();
	
	/// Transform a single instance
	abstract void transform(double[] in, double[] out);
	
	/// Untransform a single instance
	abstract void untransform(double[] in, double[] out);
}
