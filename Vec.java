// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.util.Iterator;
import java.lang.StringBuilder;

/// Represents a vector of doubles
public class Vec {
	protected double[] vals;
	protected int start;
	protected int len;

	public int size() { return len; }
	public double[] vals() { return vals; }

	public void set(int index, double value) {
		//if(start + index > len)
		//	throw new IllegalArgumentException("index not in vector!");
		vals[start + index] = value;
	}

	public double get(int index) {
		//if(star)
		return vals[start + index];
	}

	/// Makes an vector of the specified size
	public Vec(int size) {
		if(size == 0)
			vals = null;
		else
			vals = new double[size];
		start = 0;
		len = size;
	}

	public Vec(Vec that) {
		vals = new double[that.size()];
		for(int i = 0; i < that.size(); i++)
			vals[i] = that.get(i);
		start = 0;
		len = that.size();
	}

	/// Wraps the specified array of doubles
	public Vec(double[] data) {
		vals = data;
		start = 0;
		len = data.length;
	}

	/// This is NOT a copy constructor. It wraps the same buffer of values as v.
	public Vec(Vec v, int begin, int length) {
		if(v.size() < begin + length)
			throw new IllegalArgumentException("v.size() " + v.size()
				+ " < " + begin + " + " + length + " : out of bounds!");
		vals = v.vals;
		start = v.start + begin;
		len = length;
	}

	/// Unmarshalling constructor
	public Vec(Json n) {
		vals = new double[n.size()];
		for(int i = 0; i < n.size(); i++)
			vals[i] = n.getDouble(i);
		start = 0;
		len = n.size();
	}

	public double[] getData() {
		return vals;
	}



	public Json marshal() {
		Json list = Json.newList();
		for(int i = 0; i < len; i++)
			list.add(vals[start + i]);
		return list;
	}

	// Add a new entry into a vector
	public void fill(double val) {
		for(int i = 0; i < len; i++)
			vals[start + i] = val;
	}

	// Print the contents of a vector
	public String toString() {
		StringBuilder sb = new StringBuilder();

		if(len > 0) {
			sb.append(Double.toString(vals[start]));
			for(int i = 1; i < len; i++) {
				sb.append(",");
				sb.append(Double.toString(vals[start + i]));
			}
		}

		return sb.toString();
	}

	public double squaredMagnitude() {
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += vals[start + i] * vals[start + i];
		return d;
	}

	public void normalize() {
		double mag = squaredMagnitude();

		if(mag <= 0.0) {
			fill(0.0);
			vals[0] = 1.0;
		} else {
			double s = 1.0 / Math.sqrt(mag);
			for(int i = 0; i < len; i++)
				vals[i] *= s;
		}
	}

	public void copy(Vec that) {
		vals = new double[that.size()];
		for(int i = 0; i < that.size(); i++)
			vals[i] = that.get(i);
		start = 0;
		len = that.size();
	}

	// Add two compatible vectors component-wise
	public void add(Vec that) {
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		for(int i = 0; i < len; i++)
			vals[start + i] += that.get(i);
	}

	public void addEntry(int index, double value) {
		if(start + index > len)
			throw new IllegalArgumentException("index not in vector!");
		vals[start + index] += value;
	}

	public void scale(double scalar) {
		for(int i = 0; i < len; i++)
			vals[start + i] *= scalar;
	}

	/// Makes the highest value in the vector a 1, and all others 0
	public void oneHot() {
		double max = this.get(0);
		int maxIndex = 0;

		// Obtain the maximum value
		for(int i = 0; i < this.size(); ++i) {
			if(this.get(i) > max) {
				max = this.get(i);
				maxIndex = i;
			}
		}

		for(int i = 0; i < this.size(); ++i) {
			if(i != maxIndex)
				this.vals[i] = 0.0;
		}
		this.vals[maxIndex] = 1.0;
	}

	// public void oneHotLabel() {
	// 	target = new Vec(10);
	// 	target.vals[(int) trainLabels.row(i).get(0)] = 1;
	// }

	public void addScaled(double scalar, Vec that) {
		if(that.size() != this.size())
			throw new IllegalArgumentException("that: " + that.size() + " != "
				+ "this: " + this.size());
		for(int i = 0; i < len; i++)
			vals[start + i] += scalar * that.get(i);
	}

	public double dotProduct(Vec that) {
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < len; i++)
			d += get(i) * that.get(i);
		return d;
	}

	public double squaredDistance(Vec that) {
		if(that.size() != this.size())
			throw new IllegalArgumentException("mismatching sizes");
		double d = 0.0;
		for(int i = 0; i < len; i++)
		{
			double t = get(i) - that.get(i);
			d += (t * t);
		}
		return d;
	}

	public double findMin() {
		double min = vals[0];
		for(int i = 0; i < vals.length; ++i) {
			if(vals[i] < min)
				min = vals[i];
		}
		return min;
	}

	public double findMax() {
		double max = vals[0];
		for(int i = 0; i < vals.length; ++i) {
			if(vals[i] > max)
				max = vals[i];
		}
		return max;
	}

	public double findMean() {
		double sum = 0;
		for(int i = 0; i < vals.length; ++i) {
			sum += vals[i];
		}

		return (sum / vals.length);
	}

	/// Adds all the values within the vec into a single value
	public double innerSum() {
		double sum = 0;
		for(int i = 0; i < len; ++i) {
			sum += vals[start + i];
		}
		return sum;
	}


}
