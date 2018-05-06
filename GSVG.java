import java.util.ArrayList;
import java.io.PrintWriter;



enum Anchor { START, MIDDLE, END };

/// This class simplifies plotting data to an SVG file
public class GSVG {
  public static final String g_hexChars = "0123456789abcdef";
  public static final double BOGUS_XMIN = -1e308;

  protected StringBuilder m_ss;
  protected int m_width;
  protected int m_height;
  protected int m_hPos;
  protected int m_vPos;
  protected double m_hunit;
  protected double m_vunit;
  protected double m_margin;
  protected double m_xmin;
  protected double m_ymin;
  protected double m_xmax;
  protected double m_ymax;
  protected boolean m_clipping;

  double hunit() { return m_hunit; } // (xmax - xmin) / width

  double vunit() { return m_vunit; } // (ymax - ymin) / height

  /// Returns a good y position for the horizontal axis label
  double horizLabelPos() {
    return m_ymin - m_vunit * ((m_margin / 2));
  }

  /// Returns a good x position for the vertical axis label
  double vertLabelPos() {
    return m_xmin - m_hunit * ((m_margin / 2));
  }

  /// This object represents a hWindows-by-vWindows grid of charts.
	/// width and height specify the width and height of the entire grid of charts.
	/// xmin, ymin, xmax, and ymax specify the coordinates in the chart to begin drawing.
	/// margin specifies the size of the margin for axis labels.
	GSVG(int width, int height, double xmin, double ymin, double xmax, double ymax, double margin) {

    width = 1024;
    height = 768;
    xmin = 0;
    ymin = 0;
    xmax = 80;
    ymax = 50;
    margin = 50;

    m_ss.append("<?xml version=\"1.0\"?><svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"");
    m_ss.append(width + "\" height=\"" + height + "\">\n");
    closeTags();
    double chartWidth = (double)m_width;
    double chartHeight = (double)m_height;
    margin = Math.min(margin, 0.75 * chartWidth);
    margin = Math.min(margin, 0.75 * chartHeight);
    m_hunit = ((xmax - xmin)) / (chartWidth - margin);
    m_vunit = ((ymax - ymin)) / (chartHeight - margin);
    m_margin = margin;
    m_xmin = xmin;
    m_ymin = ymin;
    m_xmax = xmax;
    m_ymax = ymax;
    m_ss.append("<defs><clipPath id=\"chart0-0\"><rect x=\"" + xmin + "\" y=\""
      + ymin + "\" width=\"" + (xmax - xmin)+ "\" height=\"" + (ymax - ymin)
      + "\" /></clipPath></defs>\n");

    m_ss.append("<g transform=\"translate(" + margin + " "+ (chartHeight - margin)
      + ") scale(" + ((chartWidth - margin) / (xmax - xmin)) + " "
      + (-(chartHeight - margin) / (ymax - ymin)) + ") translate(" + (-xmin)
      + " " + (-ymin) + ")\"" + ">\n");
  }

  void dot(double x, double y, double r, int col) { // Draw a dot
    // These values are from translating C++
    r = 1.0;
    col = 0x000080;

    m_ss.append("<ellipse cx=\"" + (x) + "\" cy=\"" + (y) + "\" rx=\""
      + (r * 4 * m_hunit) + "\" ry=\"" + (r * 4 * m_vunit) + "\" fill=\"");
    color(col);
    m_ss.append("\" />\n");
  }

  /// Draw a line
  void line(double x1, double y1, double x2, double y2, double thickness, int col) {
    thickness = 1.0;
    col = 0x008000;

    m_ss.append("<line x1=\"" + (x1) + "\" y1=\"" + (y1) + "\" x2=\"" + (x2)
      + "\" y2=\"" + (y2) + "\" style=\"stroke:");
    color(col);
    double l = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    double w = thickness * (Math.abs(x2 - x1) * m_vunit + Math.abs(y2 - y1) * m_hunit) / 1.0;
    m_ss.append(";stroke-width:" + (w) + "\"/>\n");
  }

  /// Draw a rectangle
  void rect(double x, double y, double w, double h, int col) {
    col = 0x008080;

    m_ss.append("<rect x=\"" + (x) + "\" y=\"" + (y) + "\" width=\"" + (w)
      + "\" height=\"" + (h) + "\" style=\"fill:");
    color(col);
    m_ss.append("\"/>\n");
  }

  /// Draw text
  void text(double x, double y, String szText, double size, Anchor eAnchor,
    int col, double angle) { // Removed boolean parameter serifs

    size = 1.0;
    eAnchor = Anchor.START;
    col = 0x000000;
    angle = 0.0;
    boolean serifs = true; // just hardcoded because the code had a bug

    double xx = x / (m_hunit * size);
    double yy = -y / (m_vunit * size);
    m_ss.append("<text x=\"" + (xx) + "\" y=\"" + (yy) + "\" style=\"fill:");
    color(col);
    if(!serifs)
      m_ss.append(";font-family:Sans");
    m_ss.append("\" transform=\"");
    m_ss.append("scale(" + (size * m_hunit) + " " + (-size * m_vunit) + ")");
    if(angle != 0.0)
      m_ss.append(" rotate(" + (-angle) + " " + (xx) + " " + (yy) + ")");
    m_ss.append("\"");
    if(eAnchor == Anchor.MIDDLE)
      m_ss.append(" text-anchor=\"middle\"");
    else if(eAnchor == Anchor.END)
      m_ss.append(" text-anchor=\"end\"");
    m_ss.append(">" + szText + "</text>\n");
  }

  /// Generate an SVG file with all of the components that have been added so far.
  void print(PrintWriter ps) {
    // Maybe StringBuilder?

    closeTags();

    // Close the whole SVG File
    m_ss.append("</svg>\n");

    // Print it
    /// stream << m_ss.str();
    ps.write(m_ss.toString());
  }

  /// Label the horizontal axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
  /// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
  /// the grid-lines instead of the continuous values.
  void horizMarks(int maxLabels, ArrayList<String> pLabels) {
    pLabels = null;

    m_ss.append("\n<!-- Horiz labels -->\n");
    if(maxLabels >=0) {
      GPlotLabelSpacer spacer = new GPlotLabelSpacer(m_xmin, m_xmax, maxLabels);
      int count = spacer.count();
      for(int i = 0; i < count; ++i) {
        double x = spacer.label(i);
        line(x, m_ymin, x, m_ymax, 0.2, 0xa0a0a0);
        if(pLabels.isEmpty()) {
          if(pLabels.size() > i)
            text(x+3*m_hunit, m_ymin-m_vunit, pLabels.get(i).toString(), 1, Anchor.END, 0x000000, 90);
        } else {
          text(x+3*m_hunit, m_ymin-m_vunit, Double.toString(x), 1, Anchor.END, 0x000000, 90);
        }
      }
    } else {
      GPlotLabelSpacerLogarithmic spacer = new GPlotLabelSpacerLogarithmic(m_xmin, m_xmax);
      Double x = new Double(0.0);
      Boolean primary = new Boolean(false);
      while(true) {
        if(!spacer.next(x, primary)) break;

        line(Math.log(x), m_ymin, Math.log(x), m_ymax, 0.2, 0xa0a0a0); // log(x) may need Math.log(x)
        if(primary)
          text(Math.log(x)+3*m_hunit, m_ymin-m_vunit, Double.toString(x), 1, Anchor.END, 0x000000, 90); // verify log(x)
      }
    }
  }

  /// Label the vertical axis. If maxLabels is 0, then no grid-lines will be drawn. If maxLabels is -1, then
  /// Logarithmic grid-lines will be drawn. If pLabels is non-NULL, then its values will be used to label
  /// the grid-lines instead of the continuous values.
  void vertMarks(int maxLabels, ArrayList<String> pLabels) {
    pLabels = null;

    m_ss.append("\n<!-- Vert labels -->\n");
    if(maxLabels >= 0) {
      GPlotLabelSpacer spacer = new GPlotLabelSpacer(m_ymin, m_ymax, maxLabels);
      int count = spacer.count();
      for(int i = 0; i < count; ++i) {
        double y = spacer.label(i);
        line(m_xmin, y, m_xmax, y, 0.2, 0xa0a0a0);

        if(pLabels.isEmpty()) {
          if(pLabels.size() > i)
            text(m_xmin-m_hunit, y-3*m_vunit, pLabels.get(i).toString(), 1, Anchor.END, 0x000000, 0); // added 0
        } else {
          text(m_xmin-m_hunit, y-3*m_vunit, Double.toString(y), 1, Anchor.END, 0x000000, 0); // added 0
        }
      }
    } else {
      GPlotLabelSpacerLogarithmic spacer = new GPlotLabelSpacerLogarithmic(m_xmin, m_xmax);
      Double y = new Double(0.0);
      Boolean primary = new Boolean(false);
      while(true) {
        if(!spacer.next(y, primary)) break;

        line(m_xmin, Math.log(y), m_xmax, Math.log(y), 0.2, 0xa0a0a0);
        if(primary)
          text(m_xmin-m_hunit, Math.log(y)-3*m_vunit, Double.toString(y), 1, Anchor.END, 0x000000, 0); // added 0
      }
    }
    m_ss.append("\n");
  }

  /// After calling this method, all draw operations will be clipped to fall within (xmin, ymin)-(xmax, ymax),
  /// until a new chart is started.
    void clip() {
      m_ss.append("\n<!-- Clipped region -->\n");
      m_ss.append("<g clip-path=\"url(#chart0-0)\">\n");
      m_clipping = true;
    }

    void color(int c) {
      m_ss.append(String.format("%06x", c));
    }

    void closeTags() {
      // Close the current clipping group
      if(m_clipping) {
        m_ss.append("</g>");
        m_clipping = false;
      }

      // Close the current chart
      if(m_xmin != BOGUS_XMIN)
        m_ss.append("</g>");
      m_ss.append("\n\n");
    }

}
