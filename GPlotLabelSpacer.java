


/// If you need to place grid lines or labels at regular intervals
/// (like 1000, 2000, 3000, 4000... or 20, 25, 30, 35... or 0, 2, 4, 6, 8, 10...)
/// this class will help you pick where to place the labels so that
/// there are a reasonable number of them, and they all land on nice label
/// values.
class GPlotLabelSpacer {
  // Equivalent to Log base 10 of e
  public static final double M_LOG10E = 0.434294481903251827651;

  protected double m_spacing;
  protected int m_start;
  protected int m_count;

  /// Returns the number of labels that have been picked. It will be a value
  /// smaller than maxLabels.
  int count() { return m_count; }

  /// Returns the location of the n'th label (where 0 <= n < count())
  double label(int index) { return (m_start + index) * m_spacing; }

  /// maxLabels specifies the maximum number of labels that it can ever
  /// decide to use. (It should be just smaller than the number of labels
  /// that would make the graph look too crowded.)
  GPlotLabelSpacer(double min, double max, int maxLabels) {
    if(max <= min) throw new RuntimeException("max: " + max + " <= min: " + min + "; Invalid Range!");

    if(maxLabels == 0) {
      m_spacing = 0.0;
      m_start = 0;
      m_count = 0;
      return;
    }

    int p = (int)Math.ceil(Math.log((max-min) / maxLabels) * M_LOG10E);
    // Every 10
    m_spacing = Math.pow(10.0, p);
    m_start = (int)Math.ceil(min / m_spacing);
    m_count = (int)Math.floor(max / m_spacing) - m_start + 1;

    if(m_count * 5 + 4 < maxLabels) {
      // Every 2
      m_spacing *= 0.2;
      m_start = (int)Math.ceil(min / m_spacing);
      m_count = (int)Math.floor(max / m_spacing) - m_start + 1;
    } else if(m_count * 2 + 1 < maxLabels) {
      // Every 5
      m_spacing *= 0.5;
      m_start = (int)Math.ceil(min / m_spacing);
      m_count = (int)Math.floor(max / m_spacing) - m_start + 1;
    }
  }

}
