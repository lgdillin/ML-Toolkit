/// A tensor class.
class Tensor extends Vec {
	int[] dims;
	int numElements;

	/// General-purpose constructor. Example:
	/// Tensor t(v, {5, 7, 3});
	Tensor(Vec vals, int[] _dims) {
		super(vals, 0, vals.size());
		dims = new int[_dims.length];
		int tot = 1;

		for(int i = 0; i < _dims.length; i++) {
			dims[i] = _dims[i];
			tot *= _dims[i];
		}

		if(tot != vals.size())
			throw new RuntimeException("Mismatching sizes. Vec has " + Integer.toString(vals.size()) + ", Tensor has " + Integer.toString(tot));

		// Store the total number of elements
		numElements = tot;
	}

	/// Copy constructor. Copies the dimensions. Wraps the same vector.
	Tensor(Tensor copyMe) {
		super((Vec)copyMe, 0, copyMe.size());
		dims = new int[copyMe.dims.length];
		for(int i = 0; i < copyMe.dims.length; i++)
			dims[i] = copyMe.dims[i];

		numElements = copyMe.numElements;
	}

	/// This is pretty expensive just to print the dims
	void printDims() {
		String out = "Dims: ";
		for(int i = 0; i < this.dims.length; ++i) {
			out += this.dims[i] + ", ";
		}
		System.out.println(out);
	}

	int extra_dimensions() {
		int ed = 1;

		if(this.dims.length < 3)
			return 1; // this is a 2-tensor
		else {
			for(int i = 2; i < this.dims.length; ++i) {
				ed *= this.dims[i];
			}
		}

		return ed;
	}

	// Find the tensor with more dimensions and return those extra dimensions
	static int[] dims_difference(Tensor a, Tensor b) {
		if(a.dims.length == b.dims.length)
			return null;

		int index = 0;
		int[] difference;
		if(a.dims.length > b.dims.length) {
			difference = new int[a.dims.length - b.dims.length];
			for(int i = b.dims.length; i < a.dims.length; ++i) {
				difference[index] = a.dims[i];
				++index;
			}
		} else {
			difference = new int[b.dims.length - a.dims.length];
			for(int i = a.dims.length; i < b.dims.length; ++i) {
				difference[index] = b.dims[i];
				++index;
			}
		}

		return difference;
	}

	/// return dimensions of a 2-tensor
	int[] reduced_dimensions() {
		if(this.dims.length < 3)
			throw new IllegalArgumentException("already a 2 tensor");

		int[] rd = new int[2];

		for(int i = 0; i < 2; ++i) {
			rd[i] = this.dims[i];
		}
		return rd;
	}

	/// Provides a generic template for computing tensors of incompatible size
	/// NOTE: Assumes two of the tensor are of the same size
	static void safety_template(Tensor _A, Tensor _B, Tensor _C, int mode, int dims_flag) {
		if(dims_flag < 1 || dims_flag > 4)
			throw new IllegalArgumentException("invalid flag value ");
		// dims_flag = 1: size(in) = size(kernel) < size(out)
		// dims_flag = 2: size(in) < size(kernel) = size(out)
		// dims_flag = 3: size(in) > size(kernel) = size(out)
		// dims_flag = 4: size(in) = size(kernel) > size(out)

		Tensor in, kernel, out;

		// conditions
		boolean back_prop = false;
		if(mode == -1) {
			back_prop = true;
			in = _C;
			kernel = _B;
			out = _A;
		} else if(mode == 1) {
			in = _A;
			kernel = _B;
			out = _C;
		} else {
			in = _A;
			kernel = _C;
			out = _B;
		}


		// Size of the left is the same as the size of the right,
		// and both are smaller than their output
		if(dims_flag == 1) {
			int out_extra_tensors = 1;
			int kernel_tensor_size = 1;
			int out_tensor_size = 1;
			int[] out_tensor = new int[in.dims.length];
			int[] kernel_tensor = new int[in.dims.length];

			// find the reduced size of the tensor
			for(int i = 0; i < in.dims.length; ++i) {
				out_tensor_size *= out.dims[i];
				kernel_tensor_size *= kernel.dims[i];

				out_tensor[i] = out.dims[i];
				kernel_tensor[i] = kernel.dims[i];
			}

			// Find how  many extra tensors there are of the output
			for(int i = in.dims.length; i < out.dims.length; ++i) {
				out_extra_tensors *= out.dims[i];
			}

			/// Convolve either forward or backward based on mode
			if(back_prop) {
				int oPos = out.size();
				int kPos = kernel.size();

				for(int i = out_extra_tensors-1; i >= 0; --i) {
					oPos -= out_tensor_size;
					kPos -= kernel_tensor_size;

					Vec v = new Vec(out, oPos, out_tensor_size);
					Vec w = new Vec(kernel, kPos, kernel_tensor_size);

					Tensor o = new Tensor(v, out_tensor);
					Tensor k = new Tensor(w, kernel_tensor);
					convolve(in, k, o, back_prop, 1);
				}

				//System.out.println("out: " + out);
				//System.out.println("A: " + _C);
				_C = out;
			} else if(!back_prop) {
				int oPos = 0;
				int kPos = 0;
				for(int i = 0; i < out_extra_tensors; ++i) {
					Vec v = new Vec(out, oPos, out_tensor_size);
					Vec w = new Vec(kernel, kPos, kernel_tensor_size);

					Tensor o = new Tensor(v, out_tensor);
					Tensor k = new Tensor(w, kernel_tensor);
					convolve(in, kernel, o, back_prop, 1);

					oPos += out_tensor_size;
					kPos += kernel_tensor_size;
				}
			}
		}

		// The size of the left is smaller than the size of the right,
		// the right and the output are the same dimensions
		if(dims_flag == 2) {
			// backprop cannot happen here

			int kernel_extra_tensors = 1; // same size as out
			int kernel_tensor_size = 1;
			int out_tensor_size = 1;
			int[] kernel_tensor = new int[in.dims.length];
			int[] out_tensor = new int[in.dims.length];

			for(int i = 0; i < in.dims.length; ++i) {
				kernel_tensor_size *= kernel.dims[i];
				out_tensor_size *= out.dims[i];

				kernel_tensor[i] = kernel.dims[i];
				out_tensor[i] = out.dims[i];
			}

			for(int i = in.dims.length; i < kernel.dims.length; ++i) {
				kernel_extra_tensors *= kernel.dims[i];
			}

			int kPos = 0;
			int oPos = 0;
			for(int i = 0; i < kernel_extra_tensors; ++i) {
				Vec v = new Vec(kernel, kPos, kernel_tensor_size);
				Vec w = new Vec(out, oPos, out_tensor_size);

				Tensor k = new Tensor(v, kernel_tensor);
				Tensor o = new Tensor(w, out_tensor);

				convolve(in, k, o, back_prop, 1);
				kPos += kernel_tensor_size;
				oPos += out_tensor_size;
			}
		}

		// if the left is greater than the right,
		// and the right and out are the same size
		if(dims_flag == 3) {
			int in_extra_tensors = 1;
			int in_tensor_size = 1;
			int[] in_tensor = new int[kernel.dims.length];

			for(int i = 0; i < kernel.dims.length; ++i) {
				in_tensor_size *= in.dims[i];
				in_tensor[i] = in.dims[i];
			}

			for(int i = kernel.dims.length; i < in.dims.length; ++i) {
				in_extra_tensors *= in.dims[i];
			}

			int iPos = 0;
			for(int i = 0; i < in_extra_tensors; ++i) {
				Vec v = new Vec(in, iPos, in_tensor_size);
				Tensor input = new Tensor(v, in_tensor);

				convolve(input, kernel, out, back_prop, 1);
				iPos += in_tensor_size;
			}
		}

		// The left and the right are the same size
		// and they are both larger than the out
		if(dims_flag == 4) {
			int kernel_extra_tensors = 1;
			int kernel_tensor_size = 1;
			int in_tensor_size = 1;
			int[] in_tensor = new int[out.dims.length];
			int[] kernel_tensor = new int[out.dims.length];

			for(int i = 0; i < out.dims.length; ++i) {
				in_tensor_size *= in.dims[i];
				kernel_tensor_size *= kernel.dims[i];

				in_tensor[i] = in.dims[i];
				kernel_tensor[i] = kernel.dims[i];
			}

			for(int i = out.dims.length; i < in.dims.length; ++i) {
				kernel_extra_tensors *= kernel.dims[i];
			}

			if(back_prop) {
				int iPos = in.size();
				int kPos = kernel.size();
				for(int i = kernel_extra_tensors-1; i >= 0; --i) {
					iPos -= in_tensor_size;
					kPos -= kernel_tensor_size;

					Vec v = new Vec(in, iPos, in_tensor_size);
					Vec w = new Vec(kernel, kPos, kernel_tensor_size);

					Tensor input = new Tensor(v, in_tensor);
					Tensor k = new Tensor(w, kernel_tensor);
					convolve(input, k, out, back_prop, 1);
				}
			} else {
				int iPos = 0;
				int kPos = 0;
				for(int i = 0; i < kernel_extra_tensors; ++i) {
					Vec v = new Vec(in, iPos, in_tensor_size);
					Vec w = new Vec(kernel, kPos, kernel_tensor_size);

					Tensor input = new Tensor(v, in_tensor);
					Tensor k = new Tensor(w, kernel_tensor);
					convolve(input, k, out, back_prop, 1);
					iPos += in_tensor_size;
					kPos += kernel_tensor_size;
				}
			}
		}

	}

	/// Handles cases for the various types of conolvution
	static void safety_convolve(Tensor in, Tensor filter, Tensor out, int mode) {
		if(mode < -1 || mode > 1)
			throw new IllegalArgumentException("Invalid mode!");
		/// NOTE:
		// mode = -1: backProp
		// mode = 0: updateGradient
		// mode = 1: forwardProp/activation

		// Where A * B = C
		// dims_flag = 1: size(A) = size(B) < size(C)
		// dims_flag = 2: size(A) < size(B) = size(C)
		// dims_flag = 3: size(A) > size(B) = size(C)
		// dims_flag = 4: size(A) = size(B) > size(C)

		if(mode == 1) { // forward prop
			if(filter.dims.length < in.dims.length) {
				// in * filter = out
				// #in > #filter = #out
				safety_template(in, filter, out, mode, 3);
			} else if(in.dims.length < filter.dims.length) {
				// in * filter = out
				// #in < #filter = #out
				safety_template(in, filter, out, mode, 2);
			} else if(in.dims.length == filter.dims.length) {
				// regular convolution
				convolve(in, filter, out, false, 1);
			}
		} else if(mode == -1) { // backProp
			if(out.dims.length < in.dims.length) {
				// out * filter = in
				// #out = #filter < #in
				safety_template(in, filter, out, mode, 1);
			} else if(in.dims.length < out.dims.length) {
				// out * filter = in
				// #out = #filter > #in
				safety_template(in, filter, out, mode, 4);
			} else if(in.dims.length == out.dims.length) {
				convolve(out, filter, in, true, 1);
			}
		} else if(mode == 0) { //updateGradient (remember output = kernel)
			if(in.dims.length < out.dims.length) {
				// in * out = filter
				// #in < #out = #filter
				safety_template(in, filter, out, mode, 2);
			} else if(out.dims.length < in.dims.length) {
				// in * out = filter
				// #in > #out = #filter
				safety_template(in, filter, out, mode, 3);
			} else if(out.dims.length == in.dims.length) {
				convolve(in, out, filter, false, 1);
			}
		}
	}

	static void safty_convolve(Tensor in, Tensor filter, Tensor out, boolean back_prop) {
		int[] tensor_slice;
		int extra_dimensions_size = 1;
		int out_shift = 1;
		int shift_length = 1;

		// if the 'in' is larger than the 'filter' or vice versa
		if(filter.dims.length < in.dims.length) {
			// The # of extra dimensions
			int[] extra_dimensions = dims_difference(in, filter);

			// compute the number of slices we must iterate
			for(int i = 0; i < extra_dimensions.length; ++i) {
				extra_dimensions_size *= extra_dimensions[i];
			}

			// The slice of the larger tensor for computational compatibility
			tensor_slice = new int[in.dims.length - extra_dimensions.length];

			// Compute the tensor 'slice' for the larger of the in/filter
			for(int i = 0; i < filter.dims.length; ++i) {
				tensor_slice[i] = in.dims[i];
			}

			// Compute the quantity of elements we must shift by
			for(int i = 0; i < tensor_slice.length; ++i) {
				shift_length *= tensor_slice[i];
			}

			// Compute the convolution slice by slice
			if(back_prop == false) {
				int pos = 0;
				for(int i = 0; i < extra_dimensions_size; ++i) {
					// Wrap a vector of the larger tensor
					Vec v = new Vec(in, pos, shift_length);

					Tensor temp = new Tensor(v, tensor_slice);
					convolve(temp, filter, out, back_prop, 1);
					pos += shift_length;
				}
			} else if(back_prop == true) {
				throw new RuntimeException("This functionality is not yet implemented");
				// TODO: FINISH THIS CASE
				// int pos = 0;
				// for(int i = 0; i < ; ++i) {
				// 	// Wrap a vector
				// }
			}

		} else if(in.dims.length < filter.dims.length) {
			// The # of extra dimensions
			int[] extra_dimensions = dims_difference(in, filter);

			// compute the number of slices we must iterate
			for(int i = 0; i < extra_dimensions.length; ++i) {
				extra_dimensions_size *= extra_dimensions[i];
			}

			// Compute a slice of the output
			int[] out_slice = new int[filter.dims.length - extra_dimensions.length];

			// The slice of the larger tensor for computational compatibility
			tensor_slice = new int[filter.dims.length - extra_dimensions.length];

			// Compute the tensor 'slice' for the larger of the in/filter
			for(int i = 0; i < in.dims.length; ++i) {
				tensor_slice[i] = filter.dims[i];
				out_slice[i] = out.dims[i];
			}

			// Compute the quantity of elements we must shift by
			for(int i = 0; i < tensor_slice.length; ++i) {
				shift_length *= tensor_slice[i];
				out_shift *= out_slice[i];
			}

			// Compute the convolution slice by slice
			int filterPos = 0;
			int outPos = 0;
			for(int i = 0; i < extra_dimensions_size; ++i) {
				// Wrap a vector of the larger tensor
				Vec v = new Vec(filter, filterPos, shift_length);
				Vec w = new Vec(out, outPos, out_shift);

				Tensor tempFilter = new Tensor(v, tensor_slice);
				Tensor tempOut = new Tensor(w, out_slice);

				System.out.println("in: " + in.size());
				System.out.println("tempF: " + filter.size());
				System.out.println("tempO: " + out.size());

				// Problem with in tensor?
				convolve(in, tempFilter, tempOut, back_prop, 1);

				// System.out.println("in: " + in);
				// System.out.println("filter: " + tempFilter);
				// System.out.println("tempout: " + tempOut);
				filterPos += shift_length;
				outPos += out_shift;
			}

		} else if(filter.dims.length == in.dims.length)
			if(back_prop == false) // normal convolution
				convolve(in, filter, out, back_prop, 1);
			else if(back_prop == true) { // backProp
				if(in.dims.length == out.dims.length) { // regular backprop conv.
					convolve(in, filter, out, back_prop, 1);
				} else if(out.dims.length < in.dims.length) { // Assume in is _differ than out
					int[] extra_dimensions = dims_difference(in, out);

					for(int i = 0; i < extra_dimensions.length; ++i) {
						extra_dimensions_size *= extra_dimensions[i];
					}

					int in_shift_length = 1;
					int filter_shift_length = 1;

					// slice for computational compat.
					int[] in_tensor_slice = new int[in.dims.length - extra_dimensions.length];
					int[] filter_tensor_slice = new int[filter.dims.length - extra_dimensions.length];

					// compute the slice of the larger tensor
					for(int i = 0; i < out.dims.length; ++i) {
						in_tensor_slice[i] = in.dims[i];
						filter_tensor_slice[i] = filter.dims[i];
					}

					for(int i = 0; i < in_tensor_slice.length; ++i) {
						in_shift_length *= in_tensor_slice[i];
						filter_shift_length *= filter_tensor_slice[i];
					}

					int inPos = 0;
					int filterPos = 0;
					for(int i = 0; i < extra_dimensions_size; ++i) {
						// Wrap a vector of the larger tensor
						Vec v = new Vec(in, inPos, in_shift_length);
						Vec w = new Vec(filter, filterPos, filter_shift_length);

						Tensor inTemp = new Tensor(v, in_tensor_slice);
						Tensor filterTemp = new Tensor(w, filter_tensor_slice);

						convolve(inTemp, filterTemp, out, back_prop, 1);
						inPos += in_shift_length;
						filterPos += filter_shift_length;
					}
			} else {
				throw new RuntimeException("Something went wrong!");
			}
		}
	}

	/// Wraps the original convolve function to handle tensors of different dimensions
	static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter) {
		// neccesary values for the output tensor, will be needed in both cases
		int extraOutputDims = 0;
		int outputSize = 0;
		int[] reducedOutput;

		// if the tensors differ by some number of dimensions
		int extraInputDims = 0;
		int inputSize = 0;
		int[] reducedInput;

		int extraFilterDims = 0;
		int filterSize = 0;
		int[] reducedFilter;

		// if the filter has less dimensions than the input
		if(filter.dims.length < in.dims.length) {
			if(true)
				throw new RuntimeException("This code isn't finished yet!");

			// Creat a reduced-dimension array to match the filter
			reducedInput = new int[filter.dims.length];

			// Count the total number of extra elements
			extraInputDims = 1;
			for(int i = filter.dims.length; i < in.dims.length; ++i) {
				extraInputDims *= in.dims[i];
			}

			// Count elements of the input that fit in the dimensions of the filter
			inputSize = 1;
			for(int i = 0; i < filter.dims.length; ++i) {
				inputSize *= in.dims[i];
				reducedInput[i] = in.dims[i];
				//reducedOutput[i]
			}

			// Iteratively complete convolution
			int pos = 0; /// WARNING! pos isn't iterated?
			for(int i = 0; i < extraInputDims; ++i) {
				// Wrap an input vector
				Vec v = new Vec(in, pos, inputSize);
				Tensor t = new Tensor(v, reducedInput);

				convolve(in, t, out, flipFilter, 1);
			}

		}

		// if the input has less dimensions than the filter
		else if(in.dims.length < filter.dims.length) {
			// Create a reduced-dimension array to match the input
			reducedFilter = new int[in.dims.length];
			reducedOutput = new int[in.dims.length];

			// Count the number of extra elements
			extraFilterDims = 1;
			extraOutputDims = 1;
			for(int i = in.dims.length; i < filter.dims.length; ++i) {
				extraFilterDims *= filter.dims[i];
				extraOutputDims *= out.dims[i];
			}

			// Count elements of the filter that fit in the dimensions of the input
			filterSize = 1;
			outputSize = 1;
			for(int i = 0; i < in.dims.length; ++i) {
				filterSize *= filter.dims[i];
				outputSize *= out.dims[i];
				reducedFilter[i] = filter.dims[i];
				reducedOutput[i] = out.dims[i];
			}

			// Iteratively complete convolution
			int filterPos = 0;
			int outputPos = 0;
			for(int i = 0; i < extraFilterDims; ++i) {
				// Wrap an input vector
				Vec v = new Vec(filter, filterPos, filterSize);
				Tensor t = new Tensor(v, reducedFilter);

				// Wrap an output Vector
				Vec w = new Vec(out, outputPos, outputSize);
				Tensor o = new Tensor(w, reducedOutput);

				convolve(in, t, o, flipFilter, 1);

				filterPos += filterSize;
				outputPos += outputSize;
			}

		// If the output dimensions are smaller than the filter (probably backProp)
		//} else if(out.dims.length < filter.dims.length) {


		} else { // the dimensions match, just do regular convolution
			convolve(in, filter, out, flipFilter, 1);
		}


	}




	/// The result is added to the existing contents of out. It does not replace the existing contents of out.
	/// Padding is computed as necessary to fill the the out tensor.
	/// filter is the filter to convolve with in.
	/// If flipFilter is true, then the filter is flipped in all dimensions.
	static void convolve(Tensor in, Tensor filter, Tensor out, boolean flipFilter, int stride) {
		// if(out.numElements % filter.numElements != 0) {
		// 	throw new RuntimeException("output size: " + out.numElements
		// 		+ " / filter: " + filter.numElements + " does not have remainder 0!");
		// }


		// Precompute some values
		int dc = in.dims.length;
		if(dc != filter.dims.length)
			throw new RuntimeException("input # dims: " + dc + "!= filter # dims: " + filter.dims.length);
		if(dc != out.dims.length)
			throw new RuntimeException("input # dims: " + dc + "!= output # dims: " + out.dims.length);
		int[] kinner = new int[dc];
		int[] kouter = new int[dc];
		int[] stepInner = new int[dc];
		int[] stepFilter = new int[dc];
		int[] stepOuter = new int[dc];

		// Compute step sizes
		stepInner[0] = 1;
		stepFilter[0] = 1;
		stepOuter[0] = 1;
		for(int i = 1; i < dc; i++) {
			stepInner[i] = stepInner[i - 1] * in.dims[i - 1];
			stepFilter[i] = stepFilter[i - 1] * filter.dims[i - 1];
			stepOuter[i] = stepOuter[i - 1] * out.dims[i - 1];
		}
		int filterTail = stepFilter[dc - 1] * filter.dims[dc - 1] - 1;

		// Do convolution
		int op = 0;
		int ip = 0;
		int fp = 0;
		for(int i = 0; i < dc; i++) {
			kouter[i] = 0;
			kinner[i] = 0;
			int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
			int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
			kinner[i] += adj;
			fp += adj * stepFilter[i];
		}

		// kouter
		while(true) {
			double val = 0.0;

			// Fix up the initial kinner positions
			for(int i = 0; i < dc; i++) {
				int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
				int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
				kinner[i] += adj;
				fp += adj * stepFilter[i];
				ip += adj * stepInner[i];
			}

			// kinner
			while(true) {
				val += (in.get(ip) * filter.get(flipFilter ? filterTail - fp : fp));

				// increment the kinner position
				int i;
				for(i = 0; i < dc; i++) {
					kinner[i]++;
					ip += stepInner[i];
					fp += stepFilter[i];
					int padding = (stride * (out.dims[i] - 1) + filter.dims[i] - in.dims[i]) / 2;
					if(kinner[i] < filter.dims[i] && kouter[i] + kinner[i] - padding < in.dims[i])
						break;
					int adj = (padding - Math.min(padding, kouter[i])) - kinner[i];
					kinner[i] += adj;
					fp += adj * stepFilter[i];
					ip += adj * stepInner[i];
				}
				if(i >= dc)
					break;
			}
			out.set(op, out.get(op) + val);

			// increment the kouter position
			int i;
			for(i = 0; i < dc; i++) {
				kouter[i]++;
				op += stepOuter[i];
				ip += stride * stepInner[i];
				if(kouter[i] < out.dims[i])
					break;
				op -= kouter[i] * stepOuter[i];
				ip -= kouter[i] * stride * stepInner[i];
				kouter[i] = 0;
			}
			if(i >= dc)
				break;
		}
	}

	/// Throws an exception if something is wrong.
	static void test() {

		{
			// 1D test
			Vec in = new Vec(new double[]{2,3,1,0,1});
			Tensor tin = new Tensor(in, new int[]{5});

			Vec k = new Vec(new double[]{1, 0, 2});
			Tensor tk = new Tensor(k, new int[]{3});

			Vec out = new Vec(7);
			Tensor tout = new Tensor(out, new int[]{7});

			Tensor.convolve(tin, tk, tout, true, 1);

			//     2 3 1 0 1
			// 2 0 1 --->
			Vec expected = new Vec(new double[]{2, 3, 5, 6, 3, 0, 2});
			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}

		{
			// 2D test
			Vec in = new Vec(new double[] {
					1, 2, 3,
					4, 5, 6,
					7, 8, 9
			});

			Tensor tin = new Tensor(in, new int[]{3, 3});

			Vec k = new Vec(new double[] {
					1,  2,  1,
					0,  0,  0,
					-1, -2, -1
			});

			Tensor tk = new Tensor(k, new int[]{3, 3});

			Vec out = new Vec(9);
			Tensor tout = new Tensor(out, new int[]{3, 3});

			Tensor.convolve(tin, tk, tout, false, 1);

			Vec expected = new Vec(new double[] {
					-13, -20, -17,
					-18, -24, -18,
					13,  20,  17
			});

			if(Math.sqrt(out.squaredDistance(expected)) > 1e-10)
				throw new RuntimeException("wrong");
		}
	}



		// /// Reduces dimensionality of tensors down to series of convolutions of 2-tensors
		// static void convolve2D(Tensor in, Tensor filter, Tensor out) {
		// 	// if(in.dims.length < 3)
		// 	// 	throw new RuntimeException("");
		//
		// 	// Calculate the total number of 2-tensors in the input tensor
		// 	int numInputFrames = 1;
		// 	if(in.dims.length > 2) {
		// 		for(int i = 2; i < in.dims.length; ++i) {
		// 			numInputFrames *= in.dims[i];
		// 		}
		// 	}
		//
		// 	// Calculat number of filters
		// 	int numFilters = 1;
		// 	if(filter.dims.length > 2) {
		// 		for(int i = 2; i < filter.dims.length; ++i) {
		// 			numFilters *= filter.dims[i];
		// 		}
		// 	}
		//
		// 	// Calculate number of output?
		// 	int numOutputFrames = 1;
		// 	if(out.dims.length > 2) {
		// 		for(int i = 0; i < out.dims.length) {
		// 			numOutputFrames *= out.dims[i];
		// 		}
		// 	}
		//
		// 	// convolve each input 2-tensor with a filter
		// 	if(in.dims.length > filter.dims.length) {
		//
		// 	}


		// }

		private static void convolve(Tensor smaller, Tensor larger, Tensor out, int largerDims,
			int extraDims, int[] dims, boolean flipFilter) {

			int pos = 0;
			for(int i = 0; i < extraDims; ++i) {
				// Wrap a compatible tensor
				Vec v = new Vec(larger, pos, largerDims);
				Tensor t = new Tensor(v, dims);


			}
		}
}
