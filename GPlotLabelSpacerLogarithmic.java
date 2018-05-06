


/// Similar to GPlotLabelSpacer, except for logarithmic grids. To plot in
/// logarithmic space, set your plot window to have a range from log_e(min)
/// to log_e(max). When you actually plot things, plot them at log_e(x), where
/// x is the position of the thing you want to plot.
public class GPlotLabelSpacerLogarithmic {
  // Equivalent to Log base 10 of e
  public static final double M_LOG10E = 0.434294481903251827651;

  double m_max;
  int m_n, m_i;

  /// Pass in the log (base e) of your min and max values. (We make you
	/// pass them in logarithmic form, so you can't use a negative min value.)
  GPlotLabelSpacerLogarithmic(double log_e_min, double log_e_max) {

    double min = Math.exp(log_e_min);
    m_max = Math.exp(Math.min(500.0, log_e_max));
    m_n = (int)Math.floor(log_e_min * M_LOG10E);
    m_i = 1;

    while(true) {
      double p = Math.pow((double)10, m_n);
      if((m_i * p) >= min) break;

      ++m_i;
      if(m_i >= 10) {
        m_i = 0;
        ++m_n;
      }
    }
  }

  /// Returns true and sets *pos to the position of the next label.
  /// (You should actually plot it at log_e(*pos) in your plot window.)
  /// Returns false if there are no more (and doesn't set *pos).
  /// primary is set to true if the label is the primary
  /// label for the new scale.
  boolean next(Double pos, Boolean primary) {
    // We need to pass these parameters by reference
    // In C++ the prototype is (double* pos, bool* primary)
    // So we use Integer and Double class

    double p = Math.pow((double)10, m_n);
    pos = p * m_i;

    if(pos > m_max) return false;

    if(m_i == 1)
      primary = true;
    else
      primary = false;

    ++m_i;
    if(m_i >= 10) {
      m_i = 0;
      ++m_n;
    }

    return true;
  }
}
