package utils;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RectangleInsets;
import org.jfree.util.ShapeUtilities;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.text.FieldPosition;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CostPlot {
    private static List<List<double[]>> buildTestData(int seriesCount, int n) {
        Random rnd = new Random(0);
        List<List<double[]>> lines = new ArrayList<>();

        double[] scales = new double[]{1.0, 1.3, 1.7, 2.2, 2.8, 3.6};

        for (int s = 0; s < seriesCount; s++) {
            List<double[]> series = new ArrayList<>(n);

            for (int i = 1; i <= n; i++) {
                double x = i;

                double t = (i - 1) / (double) (n - 1);       // 0..1
                double base = 1e-2 * Math.pow(10, 4 * t);     // 1e-2 -> 1e2

                double noise = 1.0 + 0.12 * (rnd.nextDouble() - 0.5);

                double y = base * scales[s % scales.length] * noise;

                if (y <= 0) y = 1e-9;

                series.add(new double[]{x, y});
            }

            lines.add(series);
        }

        return lines;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                int seriesCount = 6;
                int n = 300;

                List<List<double[]>> count_costs = buildTestData(seriesCount, n);

                plot(count_costs, "Demo: 6 lines, 300 points (log-y)");

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }


    public static void plot(List<List<double[]>> count_costs, String title) throws Exception {
        if (count_costs == null || count_costs.isEmpty())
            throw new IllegalArgumentException("count_costs must contain >=1 series, each is List<double[]{x,y}>.");

        XYSeriesCollection dataset = new XYSeriesCollection();

        double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;

        for (int s = 0; s < count_costs.size(); s++) {
            List<double[]> pts = count_costs.get(s);
            if (pts == null || pts.isEmpty()) continue;

            XYSeries series = new XYSeries("cost-" + (s + 1));
            int limit = Math.min(pts.size(), 100);

            for (int i = 0; i < limit; i++) {
                double x = pts.get(i)[0];
                double y = pts.get(i)[1];

                if (!Double.isFinite(x) || !Double.isFinite(y)) continue;
                if (y <= 0) y = 1e-10;

                series.add(x, y);

                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }
            dataset.addSeries(series);
        }

        if (dataset.getSeriesCount() == 0)
            throw new IllegalArgumentException("All series are empty.");

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                "Dimension",
                "Query time (ms)",
                dataset,
                PlotOrientation.VERTICAL,
                true,   // legend
                true,
                false
        );
        chart.setAntiAlias(true);

        XYPlot plot = chart.getXYPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setOutlinePaint(Color.BLACK);
        plot.setOutlineStroke(new BasicStroke(1.2f));
        plot.setAxisOffset(new RectangleInsets(6, 6, 6, 6));

        Color grid = new Color(220, 220, 220);
        plot.setDomainGridlinePaint(grid);
        plot.setRangeGridlinePaint(grid);
        plot.setDomainGridlinesVisible(true);
        plot.setRangeGridlinesVisible(true);


        Font titleFont = new Font("SansSerif", Font.BOLD, 18);
        Font labelFont = new Font("SansSerif", Font.BOLD, 18);
        Font tickFont  = new Font("SansSerif", Font.PLAIN, 14);

        chart.getTitle().setFont(titleFont);
        if (chart.getLegend() != null) {
            chart.getLegend().setItemFont(new Font("SansSerif", Font.PLAIN, 12));
        }


        NumberAxis xAxis = new NumberAxis("Dimension");
        xAxis.setLabelFont(labelFont);
        xAxis.setTickLabelFont(tickFont);
        xAxis.setAxisLineStroke(new BasicStroke(1.2f));
        xAxis.setTickMarkStroke(new BasicStroke(1.2f));

        xAxis.setLowerMargin(0.06);
        xAxis.setUpperMargin(0.03);

        if (Double.isFinite(minX) && Double.isFinite(maxX) && minX < maxX) {

            double span = maxX - minX;
            double pad = span * 0.03;
            double minX2 = minX - pad;
            double maxX2 = maxX + pad;
            xAxis.setRange(minX2, maxX2);


            double raw = (maxX2 - minX2) / 6.0;
            double tickUnit = niceTickUnit(raw);

            xAxis.setAutoTickUnitSelection(false);
            xAxis.setTickUnit(new NumberTickUnit(tickUnit));
        }

        plot.setDomainAxis(xAxis);

        LogAxis yAxis = new LogAxis("Query time (ms)");
        yAxis.setLabelFont(labelFont);
        yAxis.setTickLabelFont(tickFont);
        yAxis.setAxisLineStroke(new BasicStroke(1.2f));
        yAxis.setTickMarkStroke(new BasicStroke(1.2f));
        yAxis.setSmallestValue(1e-12);


        yAxis.setNumberFormatOverride(new PowerOfTenUnicodeFormat());

        if (Double.isFinite(minY) && Double.isFinite(maxY) && minY > 0 && maxY > 0) {
            if (minY == maxY) {
                yAxis.setRange(minY / 10.0, maxY * 10.0);
            } else {
                yAxis.setRange(minY / 1.5, maxY * 1.5);
            }
        } else {

            yAxis.setRange(1e-3, 1e3);
        }
        plot.setRangeAxis(yAxis);


        XYLineAndShapeRenderer r = new XYLineAndShapeRenderer(true, true);
//        r.setDefaultShapesVisible(true);
//        r.setDefaultShapesFilled(false);
        r.setUseOutlinePaint(true);

        Color[] colors = new Color[] {
                new Color(31, 119, 180),
                new Color(255, 127, 14),
                new Color(44, 160, 44),
                new Color(148, 103, 189),
                new Color(23, 190, 207),
                new Color(0, 0, 0)
        };

        Shape[] shapes = new Shape[] {
                new Ellipse2D.Double(-4, -4, 8, 8),                 // o
                ShapeUtilities.createUpTriangle(5f),                // ^
                ShapeUtilities.createDiagonalCross(4f, 1.6f),       // x
                new Rectangle2D.Double(-4, -4, 8, 8),               // s
                ShapeUtilities.createDiamond(5f),                   // D
                ShapeUtilities.createRegularCross(4f, 1.6f)         // +
        };

        for (int i = 0; i < dataset.getSeriesCount(); i++) {
            Color c = colors[i % colors.length];
            Shape sh = shapes[i % shapes.length];

            r.setSeriesPaint(i, c);
            r.setSeriesStroke(i, new BasicStroke(5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            r.setSeriesShape(i, sh);

            r.setSeriesOutlinePaint(i, c);
            r.setSeriesOutlineStroke(i, new BasicStroke(5f));
        }

        plot.setRenderer(r);

        ChartFrame frame = new ChartFrame(title, chart);
        frame.setPreferredSize(new Dimension(820, 520));
        frame.pack();
        frame.setVisible(true);
    }

    private static double niceTickUnit(double step) {
        if (!(step > 0) || Double.isInfinite(step) || Double.isNaN(step)) return 1.0;
        double exp = Math.floor(Math.log10(step));
        double base = step / Math.pow(10, exp);

        double nice;
        if (base <= 1.0) nice = 1.0;
        else if (base <= 2.0) nice = 2.0;
        else if (base <= 5.0) nice = 5.0;
        else nice = 10.0;

        return nice * Math.pow(10, exp);
    }

    static class PowerOfTenUnicodeFormat extends NumberFormat {
        @Override
        public StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition pos) {
            if (number <= 0 || !Double.isFinite(number)) return toAppendTo.append("0");
            double e = Math.log10(number);
            int ei = (int) Math.rint(e);
            double pow = Math.pow(10, ei);


            if (pow > 0 && Math.abs(number - pow) / pow < 1e-8) {
                return toAppendTo.append("10").append(toSuperscript(ei));
            }

            return toAppendTo.append(String.format("%.0e", number));
        }

        @Override
        public StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition pos) {
            return format((double) number, toAppendTo, pos);
        }

        @Override
        public Number parse(String source, ParsePosition parsePosition) {
            return null;
        }

        private static String toSuperscript(int exp) {
            String s = Integer.toString(exp);
            StringBuilder sb = new StringBuilder();
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '-': sb.append('\u207B'); break; // ⁻
                    case '0': sb.append('\u2070'); break; // ⁰
                    case '1': sb.append('\u00B9'); break; // ¹
                    case '2': sb.append('\u00B2'); break; // ²
                    case '3': sb.append('\u00B3'); break; // ³
                    case '4': sb.append('\u2074'); break; // ⁴
                    case '5': sb.append('\u2075'); break; // ⁵
                    case '6': sb.append('\u2076'); break; // ⁶
                    case '7': sb.append('\u2077'); break; // ⁷
                    case '8': sb.append('\u2078'); break; // ⁸
                    case '9': sb.append('\u2079'); break; // ⁹
                    default: sb.append(c);
                }
            }
            return sb.toString();
        }
    }
}
