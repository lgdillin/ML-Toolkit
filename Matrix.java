// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Iterator;
import java.lang.StringBuilder;
import java.util.Comparator;
import java.util.Random;


/// This stores a matrix, A.K.A. data set, A.K.A. table. Each element is
/// represented as a double value. Nominal values are represented using their
/// corresponding zero-indexed enumeration value. For convenience,
/// the matrix also stores some meta-data which describes the columns (or attributes)
/// in the matrix.
public class Matrix
{
	/// Used to represent elements in the matrix for which the value is not known.
	public static final double UNKNOWN_VALUE = -1e308;

	// Data
	private ArrayList<double[]> m_data = new ArrayList<double[]>(); //matrix elements

	// Meta-data
	private String m_filename;                          // the name of the file
	private ArrayList<String> m_attr_name;                 // the name of each attribute (or column)
	private ArrayList<HashMap<String, Integer>> m_str_to_enum; // value to enumeration
	private ArrayList<HashMap<Integer, String>> m_enum_to_str; // enumeration to value


	/// Creates a 0x0 matrix. (Next, to give this matrix some dimensions, you should call:
	///    loadARFF
	///    setSize
	///    newColumn, or
	///    copyMetaData
	@SuppressWarnings("unchecked")
	public Matrix()
	{
		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
	}


	public Matrix(int rows, int cols)
	{
		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(rows, cols);
	}

	/// Build a matrix based on a 2x2 int array
	public Matrix(int[] dims) {
		if(dims.length > 2)
			throw new IllegalArgumentException("Matrices are 2 dimensional!");

		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(dims[0], dims[1]);
	}


	public Matrix(Matrix that)
	{
		m_filename = that.m_filename;
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(that.rows(), that.cols());
		copyBlock(0, 0, that, 0, 0, that.rows(), that.cols()); // (copies the meta data too)
	}


	public Matrix(Json n)
	{
		int rowCount = n.size();
		int colCount = n.get(0).size();
		this.m_filename    = "";
		this.m_attr_name   = new ArrayList<String>();
		this.m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		this.m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		setSize(rowCount, colCount);
		for(int i = 0; i < rowCount; i++)
		{
			double[] mrow = m_data.get(i);
			Json jrow = n.get(i);
			for(int j = 0; j < colCount; j++)
			{
				mrow[j] = jrow.getDouble(j);
			}
		}
	}


	/// Marshals this object into a Json DOM
	public Json marshal()
	{
		Json list = Json.newList();
		for(int i = 0; i < rows(); i++)
		{
			double[] r = m_data.get(i);
			Json list2 = Json.newList();
			for(int j = 0; j < r.length; j++)
				list2.add(r[j]);
			list.add(list2);
		}
		return list;
	}


	/// Loads the matrix from an ARFF file
	public void loadARFF(String filename)
	{
		int attrCount = 0; // Count number of attributes
		int lineNum = 0; // Used for exception messages
		Scanner s = null;
		m_str_to_enum.clear();
		m_enum_to_str.clear();
		m_attr_name.clear();

		try
		{
			s = new Scanner(new File(filename));
			while (s.hasNextLine())
			{
				lineNum++;
				String line  = s.nextLine().trim();
				String upper = line.toUpperCase();

				if (upper.startsWith("@RELATION"))
					m_filename = line.split(" ")[1];
				else if (upper.startsWith("@ATTRIBUTE"))
				{
					HashMap<String, Integer> str_to_enum = new HashMap<String, Integer>();
					HashMap<Integer, String> enum_to_str = new HashMap<Integer, String>();
					m_str_to_enum.add(str_to_enum);
					m_enum_to_str.add(enum_to_str);

					Json.StringParser sp = new Json.StringParser(line);
					sp.advance(10);
					sp.skipWhitespace();


					/// Swap this line out for the ones below
					String attrName = sp.untilWhitespace();

					/// These lines are needed sometimes
					//sp.advance(1);
					//String attrName = sp.until('\'');
					//sp.advance(1);


					m_attr_name.add(attrName);
					sp.skipWhitespace();
					int valCount = 0;
					if(sp.peek() == '{')
					{
						sp.advance(1);
						while(sp.peek() != '}')
						{
							sp.skipWhitespace();
							String attrVal = sp.untilQuoteSensitive(',', '}');
							if(sp.peek() == ',')
								sp.advance(1);
							if(str_to_enum.containsKey(attrVal))
								throw new RuntimeException("Duplicate attribute value: " + attrVal);
							str_to_enum.put(attrVal, new Integer(valCount));
							enum_to_str.put(new Integer(valCount), attrVal);
							valCount++;
						}
						sp.advance(1);
					}
					attrCount++;
				}
				else if (upper.startsWith("@DATA"))
				{
					m_data.clear();

					while (s.hasNextLine())
					{
						lineNum++;
						line = s.nextLine().trim();
						if (line.startsWith("%") || line.isEmpty())
							continue;
						double[] row = new double[attrCount];
						m_data.add(row);
						Json.StringParser sp = new Json.StringParser(line);
						for(int i = 0; i < attrCount; i++)
						{
							sp.skipWhitespace();
							String val = sp.untilQuoteSensitive(',', '\n');

							int valueCount = m_enum_to_str.get(i).size();
							if (val.equals("?")) // Unknown values are always set to UNKNOWN_VALUE
							{
								row[i] = UNKNOWN_VALUE;
							}
							else if (valueCount > 0) // if it's nominal
							{
								HashMap<String, Integer> enumMap = m_str_to_enum.get(i);
								if (!enumMap.containsKey(val))
								{
									throw new IllegalArgumentException("Unrecognized enumeration value " + val + " on line: " + lineNum + ".");
								}

								row[i] = (double)enumMap.get(val);
							}
							else // else it's continuous
								row[i] = Double.parseDouble(val); // The attribute is continuous

							sp.advance(1);
						}
					}
				}
			}
		}
		catch (FileNotFoundException e)
		{
			throw new IllegalArgumentException("Failed to open file: " + filename + ".");
		}
		finally
		{
			s.close();
		}
	}


	/// Returns a string representation of this object
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		for(int j = 0; j < rows(); j++)
		{
			if(j > 0)
				sb.append("\n");
			sb.append(row(j).toString());
		}
		return sb.toString();
	}


	public void printRow(double[] row, PrintStream os)
	{
		if(row.length != cols())
			throw new RuntimeException("Unexpected row size");
		for (int j = 0; j < row.length; j++)
		{
			if (row[j] == UNKNOWN_VALUE)
				os.print("?");
			else
			{
				int vals = valueCount(j);
				if (vals == 0)
				{
					if(Math.floor(row[j]) == row[j])
						os.print((int)Math.floor(row[j]));
					else
						os.print(row[j]);
				}
				else
				{
					int val = (int)row[j];
					if (val >= vals)
						throw new IllegalArgumentException("Value out of range.");
					os.print(attrValue(j, val));
				}
			}

			if (j + 1 < cols())
				os.print(",");
		}
	}


	public void printRow(double[] row, PrintWriter os)
	{
		if(row.length != cols())
			throw new RuntimeException("Unexpected row size");
		for (int j = 0; j < row.length; j++)
		{
			if (row[j] == UNKNOWN_VALUE)
				os.print("?");
			else
			{
				int vals = valueCount(j);
				if (vals == 0)
				{
					if(Math.floor(row[j]) == row[j])
						os.print((int)Math.floor(row[j]));
					else
						os.print(row[j]);
				}
				else
				{
					int val = (int)row[j];
					if (val >= vals)
						throw new IllegalArgumentException("Value out of range.");
					os.print(attrValue(j, val));
				}
			}

			if (j + 1 < cols())
				os.print(",");
		}
	}


	/// Saves the matrix to an ARFF file
	public void saveARFF(String filename)
	{
		PrintWriter os = null;

		try
		{
			os = new PrintWriter(filename);
			// Print the relation name, if one has been provided ('x' is default)
			os.print("@RELATION ");
			os.println(m_filename.isEmpty() ? "x" : m_filename);

			// Print each attribute in order
			for (int i = 0; i < m_attr_name.size(); i++)
			{
				os.print("@ATTRIBUTE ");

				String attributeName = m_attr_name.get(i);
				os.print(attributeName.isEmpty() ? "x" : attributeName);

				int vals = valueCount(i);

				if (vals == 0) os.println(" REAL");
				else
				{
					os.print(" {");
					for (int j = 0; j < vals; j++)
					{
						os.print(attrValue(i, j));
						if (j + 1 < vals) os.print(",");
					}
					os.println("}");
				}
			}

			// Print the data
			os.println("@DATA");
			for (int i = 0; i < rows(); i++)
			{
				double[] row = m_data.get(i);
				printRow(row, os);
				os.println();
			}
		}
		catch (FileNotFoundException e)
		{
			throw new IllegalArgumentException("Error creating file: " + filename + ".");
		}
		finally
		{
			os.close();
		}
	}

	/// Makes a rows-by-columns matrix of *ALL CONTINUOUS VALUES*.
	/// This method wipes out any data currently in the matrix. It also
	/// wipes out any meta-data.
	public void setSize(int rows, int cols) {
		m_data.clear();

		// Set the meta-data
		m_filename = "";
		m_attr_name.clear();
		m_str_to_enum.clear();
		m_enum_to_str.clear();

		// Make space for each of the columns, then each of the rows
		newColumns(cols);
		newRows(rows);
	}

	/// Makes a rows-by-columns matrix of *ALL CONTINUOUS VALUES*.
	/// This method wipes out any data currently in the matrix. It also
	/// wipes out any meta-data.
	public void setSize(int[] dims) {
		if(dims.length > 2)
			throw new IllegalArgumentException("matrices are 2 dimensional!");

		m_data.clear();

		// Set the meta-data
		m_filename = "";
		m_attr_name.clear();
		m_str_to_enum.clear();
		m_enum_to_str.clear();

		// Make space for each of the columns, then each of the rows
		newColumns(dims[1]);
		newRows(dims[0]);
	}

	/// Clears this matrix and copies the meta-data from that matrix.
	/// In other words, it makes a zero-row matrix with the same number
	/// of columns as "that" matrix. You will need to call newRow or newRows
	/// to give the matrix some rows.
	@SuppressWarnings("unchecked")
	public void copyMetaData(Matrix that)
	{
		m_data.clear();
		m_attr_name = new ArrayList<String>(that.m_attr_name);

		// Make a deep copy of that.m_str_to_enum
		m_str_to_enum = new ArrayList<HashMap<String, Integer>>();
		for (HashMap<String, Integer> map : that.m_str_to_enum)
		{
			HashMap<String, Integer> temp = new HashMap<String, Integer>();
			for (Map.Entry<String, Integer> entry : map.entrySet())
				temp.put(entry.getKey(), entry.getValue());

			m_str_to_enum.add(temp);
		}

		// Make a deep copy of that.m_enum_to_string
		m_enum_to_str = new ArrayList<HashMap<Integer, String>>();
		for (HashMap<Integer, String> map : that.m_enum_to_str)
		{
			HashMap<Integer, String> temp = new HashMap<Integer, String>();
			for (Map.Entry<Integer, String> entry : map.entrySet())
				temp.put(entry.getKey(), entry.getValue());

			m_enum_to_str.add(temp);
		}
	}


	/// Adds a column with the specified name
	public void newColumn(String name)
	{
		m_data.clear();
		m_attr_name.add(name);
		m_str_to_enum.add(new HashMap<String, Integer>());
		m_enum_to_str.add(new HashMap<Integer, String>());
	}


	/// Adds a column to this matrix with the specified number of values. (Use 0 for
	/// a continuous attribute.) This method also sets the number of rows to 0, so
	/// you will need to call newRow or newRows when you are done adding columns.
	public void newColumn(int vals)
	{
		m_data.clear();
		String name = "col_" + cols();

		m_attr_name.add(name);

		HashMap<String, Integer> temp_str_to_enum = new HashMap<String, Integer>();
		HashMap<Integer, String> temp_enum_to_str = new HashMap<Integer, String>();

		for (int i = 0; i < vals; i++)
		{
			String sVal = "val_" + i;
			temp_str_to_enum.put(sVal, i);
			temp_enum_to_str.put(i, sVal);
		}

		m_str_to_enum.add(temp_str_to_enum);
		m_enum_to_str.add(temp_enum_to_str);
	}


	/// Adds a column to this matrix with 0 values (continuous data).
	public void newColumn()
	{
		this.newColumn(0);
	}


	/// Adds n columns to this matrix, each with 0 values (continuous data).
	public void newColumns(int n)
	{
		for (int i = 0; i < n; i++)
			newColumn();
	}


	/// Returns the index of the specified value in the specified column.
	/// If there is no such value, adds it to the column.
	public int findOrCreateValue(int column, String val)
	{
		Integer i = m_str_to_enum.get(column).get(val);
		if(i == null)
		{
			int nextVal = m_enum_to_str.get(column).size();
			Integer integ = new Integer(nextVal);
			m_enum_to_str.get(column).put(integ, val);
			m_str_to_enum.get(column).put(val, integ);
			return nextVal;
		}
		else
			return i.intValue();
	}


	/// Adds one new row to this matrix. Returns a reference to the new row.
	public double[] newRow()
	{
		int c = cols();
		if (c == 0)
			throw new IllegalArgumentException("You must add some columns before you add any rows.");
		double[] newRow = new double[c];
		m_data.add(newRow);
		return newRow;
	}


	/// Adds one new row to this matrix at the specified location. Returns a reference to the new row.
	public double[] insertRow(int i)
	{
		int c = cols();
		if (c == 0)
			throw new IllegalArgumentException("You must add some columns before you add any rows.");
		double[] newRow = new double[c];
		m_data.add(i, newRow);
		return newRow;
	}


	/// Removes the specified row from this matrix. Returns a reference to the removed row.
	public double[] removeRow(int i)
	{
		return m_data.remove(i);
	}


	/// Appends the specified row to this matrix.
	public void takeRow(double[] row)
	{
		if(row.length != cols())
			throw new IllegalArgumentException("Col size differs from the number of columns in this matrix.");
		m_data.add(row);
	}

	public void takeRow(Vec row) {
		if(row.size() != cols())
			throw new IllegalArgumentException("Vector differs from columns size");
		m_data.add(row.vals());
	}


	/// Adds 'n' new rows to this matrix
	public void newRows(int n)
	{
		for (int i = 0; i < n; i++)
			newRow();
	}


	/// Returns the number of rows in the matrix
	public int rows() { return m_data.size(); }


	/// Returns the number of columns (or attributes) in the matrix
	public int cols() { return m_attr_name.size(); }


	/// Returns the name of the specified attribute
	public String attrName(int col) { return m_attr_name.get(col); }


	/// Returns the name of the specified value
	public String attrValue(int attr, int val)
	{
		String value = m_enum_to_str.get(attr).get(val);
		if (value == null)
			throw new IllegalArgumentException("No name");
		else
			return value;
	}


	public String getString(int r, int c)
	{
		double val = m_data.get(r)[c];
		return attrValue(c, (int)val);
	}

	/// Returns the enumerated index of the specified string
	public int valueEnum(int attr, String val)
	{
		Integer i = m_str_to_enum.get(attr).get(val);
		if (i == null)
		{
			// Make a very detailed error message listing all possible choices
			String s = "";
			Iterator<Map.Entry<String,Integer>> it = m_str_to_enum.get(attr).entrySet().iterator();
			while(it.hasNext())
			{
				if(s.length() > 0)
					s += ", ";
				Map.Entry<String,Integer> entry = it.next();
				s += "\"" + entry.getKey() + "\"";
				s += "->";
				s += Integer.toString(entry.getValue().intValue());
			}
			throw new IllegalArgumentException("No such value: \"" + val + "\". Choices are: " + s);
		}
		else
			return i.intValue();
	}


	/// Returns a reference to the specified row
	public Vec row(int index) { return new Vec(m_data.get(index)); }


	/// Swaps the positions of the two specified rows
	public void swapRows(int a, int b)
	{
		double[] temp = m_data.get(a);
		m_data.set(a, m_data.get(b));
		m_data.set(b, temp);
	}


	/// Returns the number of values associated with the specified attribute (or column)
	/// 0 = continuous, 2 = binary, 3 = trinary, etc.
	public int valueCount(int attr) { return m_enum_to_str.get(attr).size(); }


	/// Copies that matrix
	void copy(Matrix that)
	{
		setSize(that.rows(), that.cols());
		copyBlock(0, 0, that, 0, 0, that.rows(), that.cols());
	}


	/// Returns the mean of the elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMean(int col)
	{
		double sum = 0.0;
		int count = 0;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
			{
				sum += val;
				count++;
			}
		}

		return sum / count;
	}


	/// Returns the minimum element in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMin(int col)
	{
		double min = Double.MAX_VALUE;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
				min = Math.min(min, val);
		}

		return min;
	}

	/// returns the single largest value in the matrix
	public double maxValue() {
		double max = this.row(0).get(0);

		for(int i = 0; i < this.rows(); ++i) {
			for(int j = 0; j < this.cols(); ++j) {
				if(this.row(i).get(j) > max)
					max = this.row(i).get(j);
			}
		}
		return max;
	}

	/// Returns the


	/// Returns the maximum element in the specifed column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double columnMax(int col)
	{
		double max = -Double.MAX_VALUE;
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
				max = Math.max(max, val);
		}

		return max;
	}


	/// Returns the most common value in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	public double mostCommonValue(int col)
	{
		HashMap<Double, Integer> counts = new HashMap<Double, Integer>();
		for (double[] list : m_data)
		{
			double val = list[col];
			if (val != UNKNOWN_VALUE)
			{
				Integer result = counts.get(val);
				if (result == null) result = 0;

				counts.put(val, result + 1);
			}
		}

		int valueCount = 0;
		double value   = 0;
		for (Map.Entry<Double, Integer> entry : counts.entrySet())
		{
			if (entry.getValue() > valueCount)
			{
				value      = entry.getKey();
				valueCount = entry.getValue();
			}
		}

		return value;
	}


	/// Copies the specified rectangular portion of that matrix, and puts it in the specified location in this matrix.
	public void copyBlock(int destRow, int destCol, Matrix that, int rowBegin, int colBegin, int rowCount, int colCount)
	{
		if (destRow + rowCount > this.rows())
			throw new IllegalArgumentException("01. Out of range for destination matrix. " + '\n'
				+ "destRow + rowCount: " + (destRow+rowCount) + " > this.rows(): " + this.rows());
		if( destCol + colCount > this.cols())
			throw new IllegalArgumentException("01. Out of range for destination matrix. " + '\n'
				+ "destCol + colCount: " + (destCol+colCount) + " > this.cols(): " + this.cols());
		if (rowBegin + rowCount > that.rows())
			throw new IllegalArgumentException("02. Out of range for source matrix." + '\n'
			 + "rowBegin + rowCount: " + (rowBegin+rowCount) + " > that.rows(): " + that.rows());
		if(colBegin + colCount > that.cols())
			throw new IllegalArgumentException("02. Out of range for source matrix." + '\n'
				+ "colBegin + colCount: " + (colBegin+colCount) + " > that.cols(): " + that.cols());

		// Copy the specified region of meta-data
		for (int i = 0; i < colCount; i++)
		{
			m_attr_name.set(destCol + i, that.m_attr_name.get(colBegin + i));
			m_str_to_enum.set(destCol + i, new HashMap<String, Integer>(that.m_str_to_enum.get(colBegin + i)));
			m_enum_to_str.set(destCol + i, new HashMap<Integer, String>(that.m_enum_to_str.get(colBegin + i)));
		}

		// Copy the specified region of data
		for (int i = 0; i < rowCount; i++)
		{
			double[] source = that.m_data.get(rowBegin + i);
			double[] dest = this.m_data.get(destRow + i);
			for(int j = 0; j < colCount; j++)
				dest[destCol + j] = source[colBegin + j];
		}
	}


	/// Sets every element in the matrix to the specified value.
	public void fill(double val)
	{
		for (double[] vec : m_data)
		{
			for(int i = 0; i < vec.length; i++)
				vec[i] = val;
		}
	}


	/// Sets every element in the matrix to the specified value.
	public void scale(double scalar)
	{
		for (double[] vec : m_data)
		{
			for(int i = 0; i < vec.length; i++)
				vec[i] *= scalar;
		}
	}


	/// Adds every element in that matrix to this one
	public void addScaled(Matrix that, double scalar)
	{
		if(that.rows() != this.rows() || that.cols() != this.cols())
			throw new IllegalArgumentException("Mismatching size");
		for (int i = 0; i < rows(); i++)
		{
			Vec dest = this.row(i);
			Vec src = that.row(i);
			dest.addScaled(scalar, src);
		}
	}


	/// Sets this to the identity matrix.
	public void setToIdentity()
	{
		fill(0.0);
		int m = Math.min(cols(), rows());
		for(int i = 0; i < m; i++)
			m_data.get(i)[i] = 1.0;
	}


	/// Throws an exception if that has a different number of columns than
	/// this, or if one of its columns has a different number of values.
	public void checkCompatibility(Matrix that)
	{
		int c = cols();
		if (that.cols() != c)
			throw new IllegalArgumentException("Matrices have different number of columns.");

		for (int i = 0; i < c; i++)
		{
			if (valueCount(i) != that.valueCount(i))
				throw new IllegalArgumentException("Column " + i + " has mis-matching number of values.");
		}
	}

	private static class SortComparator implements Comparator<double[]>
	{
		int column;
		boolean ascending;

		SortComparator(int col, boolean ascend)
		{
			column = col;
			ascending = ascend;
		}

		public int compare(double[] a, double[] b)
		{
			if(ascending)
			{
				if(a[column] < b[column])
					return -1;
				else if(a[column] > b[column])
					return 1;
				else
					return 0;
			}
			else
			{
				if(a[column] < b[column])
					return 1;
				else if(a[column] > b[column])
					return -1;
				else
					return 0;
			}
		}
	}

	public void sort(int column, boolean ascending)
	{
		m_data.sort(new SortComparator(column, ascending));
	}



	double Matrix_pythag(double a, double b)
	{
		double at = Math.abs(a);
		double bt = Math.abs(b);
		if(at > bt)
		{
			double ct = bt / at;
			return at * Math.sqrt(1.0 + ct * ct);
		}
		else if(bt > 0.0)
		{
			double ct = at / bt;
			return bt * Math.sqrt(1.0 + ct * ct);
		}
		else
			return 0.0;
	}

	double Matrix_safeDivide(double n, double d)
	{
		if(d == 0.0 && n == 0.0)
			return 0.0;
		else
		{
			double t = n / d;
			//GAssert(t > -1e200, "prob");
			return t;
		}
	}

 	double Matrix_takeSign(double a, double b)
	{
		return (b >= 0.0 ? Math.abs(a) : -Math.abs(a));
	}

	void fixNans()
	{
		int colCount = cols();
		for(int i = 0; i < rows(); i++)
		{
			double[] pRow = m_data.get(i);
			for(int j = 0; j < colCount; j++)
			{
				if(Double.isNaN(pRow[j]))
					pRow[j] = (i == j ? 1.0 : 0.0);
			}
		}
	}

	Matrix transpose()
	{
		Matrix res = new Matrix(cols(), rows());
		for(int i = 0; i < rows(); i++)
		{
			for(int j = 0; j < cols(); j++)
				res.m_data.get(j)[i] = m_data.get(i)[j];
		}
		return res;
	}


	/// Swaps the the two specified columns */
	public void swapColumns(int a, int b)
	{
		for(int i = 0; i < rows(); i++)
		{
			double[] r = m_data.get(i);
			double t = r[a];
			r[a] = r[b];
			r[b] = t;
		}
		String t = m_attr_name.get(a);
		m_attr_name.set(a, m_attr_name.get(b));
		m_attr_name.set(b, t);
		HashMap<String, Integer> t2 = m_str_to_enum.get(a);
		m_str_to_enum.set(a, m_str_to_enum.get(b));
		m_str_to_enum.set(b, t2);
		HashMap<Integer, String> t3 = m_enum_to_str.get(a);
		m_enum_to_str.set(a, m_enum_to_str.get(b));
		m_enum_to_str.set(b, t3);
	}

/*
	/// Multiplies this matrix by v. If transpose is true, transposes this matrix first.
	double[] multiply(double[] v, boolean transpose)
	{
		int r = rows();
		if(transpose)
		{
			Vec y = new Vec(cols());
			for(int i = 0; i < r; i++)
				y.addScaled(v[i], row(i));
			return y;
		}
		else
		{
			double[] y = new double[r];
			for(int i = 0; i < r; i++)
				y[i] = v.dotProduct(row(i));
			return y;
		}
	}
*/

	static Matrix outer_product(Vec v, Vec w) {
		if(v.size() != w.size())
			throw new IllegalArgumentException("mismatching sizes");

		Matrix res = new Matrix(v.size(), w.size());
		for(int i = 0; i < v.size(); ++i) {
			double[] newRow = new double[w.size()];
			for(int j = 0; j < w.size(); ++j) {
				newRow[i] = v.get(j) * w.get(j);
			}
			res.takeRow(newRow);
		}
		return res;
	}

	/// Multiplies two  matrices together
	static Matrix multiply(Matrix a, Matrix b, boolean transposeA, boolean transposeB)
	{
		Matrix res = new Matrix(transposeA ? a.cols() : a.rows(), transposeB ? b.rows() : b.cols());
		if(transposeA)
		{
			if(transposeB)
			{
				if(a.rows() != b.cols())
					throw new IllegalArgumentException("No can do");
				for(int i = 0; i < res.rows(); i++)
				{
					for(int j = 0; j < res.cols(); j++)
					{
						double d = 0.0;
						for(int k = 0; k < a.cols(); k++)
							d += a.m_data.get(k)[i] * b.m_data.get(j)[k];
						res.m_data.get(i)[j] = d;
					}
				}
			}
			else
			{
				if(a.rows() != b.rows())
					throw new IllegalArgumentException("No can do");
				for(int i = 0; i < res.rows(); i++)
				{
					for(int j = 0; j < res.cols(); j++)
					{
						double d = 0.0;
						for(int k = 0; k < a.cols(); k++)
							d += a.m_data.get(k)[i] * b.m_data.get(k)[j];
						res.m_data.get(i)[j] = d;
					}
				}
			}
		}
		else
		{
			if(transposeB)
			{
				if(a.cols() != b.cols())
					throw new IllegalArgumentException("No can do");
				for(int i = 0; i < res.rows(); i++)
				{
					for(int j = 0; j < res.cols(); j++)
					{
						double d = 0.0;
						for(int k = 0; k < a.cols(); k++)
							d += a.m_data.get(i)[k] * b.m_data.get(j)[k];
						res.m_data.get(i)[j] = d;
					}
				}
			}
			else
			{
				if(a.cols() != b.rows())
					throw new IllegalArgumentException("A cols != B rows");
				for(int i = 0; i < res.rows(); i++)
				{
					for(int j = 0; j < res.cols(); j++)
					{
						double d = 0.0;
						for(int k = 0; k < a.cols(); k++)
							d += a.m_data.get(i)[k] * b.m_data.get(k)[j];
						res.m_data.get(i)[j] = d;
					}
				}
			}
		}
		return res;
	}




	class SVDResult
	{
		Matrix u;
		Matrix v;
		double[] diag;
	}


	/// Performs singular value decomposition of this matrix
	SVDResult singularValueDecompositionHelper(boolean throwIfNoConverge, int maxIters)
	{
		int m = rows();
		int n = cols();
		if(m < n)
			throw new IllegalArgumentException("Expected at least as many rows as columns");
		int j, k;
		int l = 0;
		int p, q;
		double c, f, h, s, x, y, z;
		double norm = 0.0;
		double g = 0.0;
		double scale = 0.0;
		SVDResult res = new SVDResult();
		Matrix pU = new Matrix(m, m);
		res.u = pU;
		pU.fill(0.0);
		for(int i = 0; i < m; i++)
		{
			double[] rOut = pU.m_data.get(i);
			double[] rIn = m_data.get(i);
			for(j = 0; j < n; j++)
				rOut[j] = rIn[j];
		}
		double[] pSigma = new double[n];
		res.diag = pSigma;
		Matrix pV = new Matrix(n, n);
		res.v = pV;
		pV.fill(0.0);
		double[] temp = new double[n];

		// Householder reduction to bidiagonal form
		for(int i = 0; i < n; i++)
		{
			// Left-hand reduction
			temp[i] = scale * g;
			l = i + 1;
			g = 0.0;
			s = 0.0;
			scale = 0.0;
			if(i < m)
			{
				for(k = i; k < m; k++)
					scale += Math.abs(pU.m_data.get(k)[i]);
				if(scale != 0.0)
				{
					for(k = i; k < m; k++)
					{
						pU.m_data.get(k)[i] = Matrix_safeDivide(pU.m_data.get(k)[i], scale);
						double t = pU.m_data.get(k)[i];
						s += t * t;
					}
					f = pU.m_data.get(i)[i];
					g = -Matrix_takeSign(Math.sqrt(s), f);
					h = f * g - s;
					pU.m_data.get(i)[i] = f - g;
					if(i != n - 1)
					{
						for(j = l; j < n; j++)
						{
							s = 0.0;
							for(k = i; k < m; k++)
								s += pU.m_data.get(k)[i] * pU.m_data.get(k)[j];
							f = Matrix_safeDivide(s, h);
							for(k = i; k < m; k++)
								pU.m_data.get(k)[j] += f * pU.m_data.get(k)[i];
						}
					}
					for(k = i; k < m; k++)
						pU.m_data.get(k)[i] *= scale;
				}
			}
			pSigma[i] = scale * g;

			// Right-hand reduction
			g = 0.0;
			s = 0.0;
			scale = 0.0;
			if(i < m && i != n - 1)
			{
				for(k = l; k < n; k++)
					scale += Math.abs(pU.m_data.get(i)[k]);
				if(scale != 0.0)
				{
					for(k = l; k < n; k++)
					{
						pU.m_data.get(i)[k] = Matrix_safeDivide(pU.m_data.get(i)[k], scale);
						double t = pU.m_data.get(i)[k];
						s += t * t;
					}
					f = pU.m_data.get(i)[l];
					g = -Matrix_takeSign(Math.sqrt(s), f);
					h = f * g - s;
					pU.m_data.get(i)[l] = f - g;
					for(k = l; k < n; k++)
						temp[k] = Matrix_safeDivide(pU.m_data.get(i)[k], h);
					if(i != m - 1)
					{
						for(j = l; j < m; j++)
						{
							s = 0.0;
							for(k = l; k < n; k++)
								s += pU.m_data.get(j)[k] * pU.m_data.get(i)[k];
							for(k = l; k < n; k++)
								pU.m_data.get(j)[k] += s * temp[k];
						}
					}
					for(k = l; k < n; k++)
						pU.m_data.get(i)[k] *= scale;
				}
			}
			norm = Math.max(norm, Math.abs(pSigma[i]) + Math.abs(temp[i]));
		}

		// Accumulate right-hand transform
		for(int i = n - 1; i >= 0; i--)
		{
			if(i < n - 1)
			{
				if(g != 0.0)
				{
					for(j = l; j < n; j++)
						pV.m_data.get(i)[j] = Matrix_safeDivide(Matrix_safeDivide(pU.m_data.get(i)[j], pU.m_data.get(i)[l]), g); // (double-division to avoid underflow)
					for(j = l; j < n; j++)
					{
						s = 0.0;
						for(k = l; k < n; k++)
							s += pU.m_data.get(i)[k] * pV.m_data.get(j)[k];
						for(k = l; k < n; k++)
							pV.m_data.get(j)[k] += s * pV.m_data.get(i)[k];
					}
				}
				for(j = l; j < n; j++)
				{
					pV.m_data.get(i)[j] = 0.0;
					pV.m_data.get(j)[i] = 0.0;
				}
			}
			pV.m_data.get(i)[i] = 1.0;
			g = temp[i];
			l = i;
		}

		// Accumulate left-hand transform
		for(int i = n - 1; i >= 0; i--)
		{
			l = i + 1;
			g = pSigma[i];
			if(i < n - 1)
			{
				for(j = l; j < n; j++)
					pU.m_data.get(i)[j] = 0.0;
			}
			if(g != 0.0)
			{
				g = Matrix_safeDivide(1.0, g);
				if(i != n - 1)
				{
					for(j = l; j < n; j++)
					{
						s = 0.0;
						for(k = l; k < m; k++)
							s += pU.m_data.get(k)[i] * pU.m_data.get(k)[j];
						f = Matrix_safeDivide(s, pU.m_data.get(i)[i]) * g;
						for(k = i; k < m; k++)
							pU.m_data.get(k)[j] += f * pU.m_data.get(k)[i];
					}
				}
				for(j = i; j < m; j++)
					pU.m_data.get(j)[i] *= g;
			}
			else
			{
				for(j = i; j < m; j++)
					pU.m_data.get(j)[i] = 0.0;
			}
			pU.m_data.get(i)[i] += 1.0;
		}

		// Diagonalize the bidiagonal matrix
		for(k = n - 1; k >= 0; k--) // For each singular value
		{
			for(int iter = 1; iter <= maxIters; iter++)
			{
				// Test for splitting
				boolean flag = true;
				q = 0;
				for(l = k; l >= 0; l--)
				{
					q = l - 1;
					if(Math.abs(temp[l]) + norm == norm)
					{
						flag = false;
						break;
					}
					if(Math.abs(pSigma[q]) + norm == norm)
						break;
				}

				if(flag)
				{
					c = 0.0;
					s = 1.0;
					for(int i = l; i <= k; i++)
					{
						f = s * temp[i];
						temp[i] *= c;
						if(Math.abs(f) + norm == norm)
							break;
						g = pSigma[i];
						h = Matrix_pythag(f, g);
						pSigma[i] = h;
						h = Matrix_safeDivide(1.0, h);
						c = g * h;
						s = -f * h;
						for(j = 0; j < m; j++)
						{
							y = pU.m_data.get(j)[q];
							z = pU.m_data.get(j)[i];
							pU.m_data.get(j)[q] = y * c + z * s;
							pU.m_data.get(j)[i] = z * c - y * s;
						}
					}
				}

				z = pSigma[k];
				if(l == k)
				{
					// Detect convergence
					if(z < 0.0)
					{
						// Singular value should be positive
						pSigma[k] = -z;
						for(j = 0; j < n; j++)
							pV.m_data.get(k)[j] *= -1.0;
					}
					break;
				}
				if(throwIfNoConverge && iter >= maxIters)
					throw new IllegalArgumentException("failed to converge");

				// Shift from bottom 2x2 minor
				x = pSigma[l];
				q = k - 1;
				y = pSigma[q];
				g = temp[q];
				h = temp[k];
				f = Matrix_safeDivide(((y - z) * (y + z) + (g - h) * (g + h)), (2.0 * h * y));
				g = Matrix_pythag(f, 1.0);
				f = Matrix_safeDivide(((x - z) * (x + z) + h * (Matrix_safeDivide(y, (f + Matrix_takeSign(g, f))) - h)), x);

				// QR transform
				c = 1.0;
				s = 1.0;
				for(j = l; j <= q; j++)
				{
					int i = j + 1;
					g = temp[i];
					y = pSigma[i];
					h = s * g;
					g = c * g;
					z = Matrix_pythag(f, h);
					temp[j] = z;
					c = Matrix_safeDivide(f, z);
					s = Matrix_safeDivide(h, z);
					f = x * c + g * s;
					g = g * c - x * s;
					h = y * s;
					y = y * c;
					for(p = 0; p < n; p++)
					{
						x = pV.m_data.get(j)[p];
						z = pV.m_data.get(i)[p];
						pV.m_data.get(j)[p] = x * c + z * s;
						pV.m_data.get(i)[p] = z * c - x * s;
					}
					z = Matrix_pythag(f, h);
					pSigma[j] = z;
					if(z != 0.0)
					{
						z = Matrix_safeDivide(1.0, z);
						c = f * z;
						s = h * z;
					}
					f = c * g + s * y;
					x = c * y - s * g;
					for(p = 0; p < m; p++)
					{
						y = pU.m_data.get(p)[j];
						z = pU.m_data.get(p)[i];
						pU.m_data.get(p)[j] = y * c + z * s;
						pU.m_data.get(p)[i] = z * c - y * s;
					}
				}
				temp[l] = 0.0;
				temp[k] = f;
				pSigma[k] = x;
			}
		}

		// Sort the singular values from largest to smallest
		for(int i = 1; i < n; i++)
		{
			for(j = i; j > 0; j--)
			{
				if(pSigma[j - 1] >= pSigma[j])
					break;
				pU.swapColumns(j - 1, j);
				pV.swapRows(j - 1, j);
				double tmp = pSigma[j];
				pSigma[j] = pSigma[j - 1];
				pSigma[j - 1] = tmp;
			}
		}

		// Return results
		pU.fixNans();
		pV.fixNans();
		return res;
	}


	/// Returns the Moore-Penrose pseudoinverse of this matrix
	Matrix pseudoInverse()
	{
		SVDResult res;
		int colCount = cols();
		int rowCount = rows();
		if(rowCount < colCount)
		{
			Matrix pTranspose = transpose();
			res = pTranspose.singularValueDecompositionHelper(false, 80);
		}
		else
			res = singularValueDecompositionHelper(false, 80);
		Matrix sigma = new Matrix(rowCount < colCount ? colCount : rowCount, rowCount < colCount ? rowCount : colCount);
		sigma.fill(0.0);
		int m = Math.min(rowCount, colCount);
		for(int i = 0; i < m; i++)
		{
			if(Math.abs(res.diag[i]) > 1e-9)
				sigma.m_data.get(i)[i] = Matrix_safeDivide(1.0, res.diag[i]);
			else
				sigma.m_data.get(i)[i] = 0.0;
		}
		Matrix pT = Matrix.multiply(res.u, sigma, false, false);
		if(rowCount < colCount)
			return Matrix.multiply(pT, res.v, false, false);
		else
			return Matrix.multiply(res.v, pT, true, true);
	}
/*
	/// Computes the first principal component of this matrix
	double[] firstPrincipalComponent(Random rand)
	{
		int c = cols();
		int r = rows();
		double[] p = new double[c];
		double[] t = new double[c];
		for(int i = 0; i < c; i++)
			p[i] = rand.nextGaussian();
		Vec.normalize(p);
		double m = 0;
		for(int i = 0; i < 200; i++)
		{
			t.fill(0.0);
			for(int j = 0; j < r; j++)
				Vec.addScaled(t, row(j), Vec.dotProduct(row(j), p));
			double d = Math.sqrt(Vec.squaredMagnitude(t));
			Vec.normalize(t);
			double[] tt = p;
			p = t;
			t = tt;
			if(i < 6 || d - m > 0.0001)
				m = d;
			else
				break;
		}
		return p;
	}


	/// Uses the Gram Schmidt process to remove a component from a matrix
	void gramSchmidt(double[] v)
	{
		int b = 0;
		int r = rows();
		for(int i = 0; i < r; i++)
		{
			double d = Vec.dotProduct(row(i), v);
			Vec.addScaled(row(i), v, -d);
		}
	}


	/// Returns a matrix containing the first k principal components of this matrix. (Destroys the contents of this matrix.)
	Matrix pca(int k, Random rand, double[] centroid)
	{
		int c = cols();
		int r = rows();
		for(int i = 0; i < c; i++)
		{
			for(int j = 0; j < r; j++)
				row(j)[i] -= centroid[i];
		}
		Matrix m = new Matrix(k, c);
		for(int i = 0; i < k; i++)
		{
			double[] p = firstPrincipalComponent(rand);
			gramSchmidt(p);
			Vec.copy(m.row(i), p);
		}
		return m;
	}*/
}
