package data_structure;

import java.util.Arrays;

import static data_structure.Parameters.D;

public class Point {
    public double[] data;
    public Point(double[] data) {
        this.data = data;
    }

    public Point() {
        data = new double[D];
    }

    public Point(boolean isMinPoint) {
        data = new double[D];
        if (isMinPoint) {
            Arrays.fill(data, -Double.MAX_VALUE);
        } else {
            Arrays.fill(data, Double.MAX_VALUE);
        }
    }

    public boolean valueEqual(Point p) {
        for (int i = 0; i < D; i++) {
            if (p.data[i] != data[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String toString() {
        return "Point{" +
                "data=" + Arrays.toString(data) +
                '}';
    }
}
