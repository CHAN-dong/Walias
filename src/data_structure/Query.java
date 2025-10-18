package data_structure;

import static data_structure.Parameters.D;
import static data_structure.Parameters.metaLength;

public class Query {
    public Point pointMin;
    public Point pointMax;

    @Override
    public String toString() {
        return "Query{" +
                "pointMin=" + pointMin +
                ", pointMax=" + pointMax +
                '}';
    }

    public boolean valueEqual(Query query) {
        return pointMin.valueEqual(query.pointMin) && pointMax.valueEqual(query.pointMax);
    }

    public Query() {
        pointMin = new Point(false);
        pointMax = new Point(true);
    }

    public Query(Point pointMin, Point pointMax) {
        this.pointMin = pointMin;
        this.pointMax = pointMax;
    }

    public void addPoint(double[] point) {

        // update the rectangle range
        for (int i = 0; i < D; i++) {
            pointMin.data[i] = Math.min(point[i], this.pointMin.data[i]);
            pointMax.data[i] = Math.max(point[i], this.pointMax.data[i]);
        }
    }

    public Query merge(Query other) {
        Point pointMin = new Point();
        Point pointMax = new Point();
        for (int i = 0; i < D; i++) {
            pointMin.data[i] = Math.min(other.pointMin.data[i], this.pointMin.data[i]);
            pointMax.data[i] = Math.max(other.pointMax.data[i], this.pointMax.data[i]);
        }
        return new Query(pointMin, pointMax);
    }


    public boolean contains(Query other) {
        for (int i = 0; i < D; i++) {
            if (other.pointMin.data[i] < pointMin.data[i] || other.pointMax.data[i] > pointMax.data[i]) {
                return false;
            }
        }
        return true;
    }

    public boolean hasIntersection(Query other) {
        for (int i = 0; i < D; i++) {
            if (pointMax.data[i] < other.pointMin.data[i] || pointMin.data[i] > other.pointMax.data[i]) {
                return false;
            }
        }
        return true;
    }

    public boolean hasIntersection(Point point) {
        for (int i = 0; i < D; i++) {
            if (pointMax.data[i] < point.data[i] || pointMin.data[i] > point.data[i]) {
                return false;
            }
        }
        return true;
    }

    public boolean checkRelation(Query other) {

        // Check if the current rectangle contains another rectangle
        if (contains(other)) {
            return false; // embody
        }

        // check if there is an intersection
        if (hasIntersection(other)) {
            return true; // intersect
        }

        // Otherwise, they will not intersect
        return false; // they do not intersect
    }

    public Query union(Query q2) {
        double[] newMin = new double[D];
        double[] newMax = new double[D];

        for (int d = 0; d < D; d++) {
            double qMinD = this.pointMin.data[d];
            double qMaxD = this.pointMax.data[d];
            double dsMinD = q2.pointMin.data[d];
            double dsMaxD = q2.pointMax.data[d];

            // computing union
            newMin[d] = Math.min(qMinD, dsMinD);
            newMax[d] = Math.max(qMaxD, dsMaxD);
        }

        return new Query(new Point(newMin), new Point(newMax));
    }


    public Query intersect(Query q2) {
        double[] newMin = new double[D];
        double[] newMax = new double[D];

        for (int d = 0; d < D; d++) {
            double qMinD = this.pointMin.data[d];
            double qMaxD = this.pointMax.data[d];
            double dsMinD = q2.pointMin.data[d];
            double dsMaxD = q2.pointMax.data[d];

            // there is no intersection
            if (qMaxD < dsMinD || qMinD > dsMaxD) {
                return null;
            }

            // Take the intersection range
            newMin[d] = Math.max(qMinD, dsMinD);
            newMax[d] = Math.min(qMaxD, dsMaxD);
        }

        return new Query(new Point(newMin), new Point(newMax));
    }



    public double computePerimeter() {
        double[] min = this.pointMin.data;
        double[] max = this.pointMax.data;
        int dim = min.length;

        double[] lengths = new double[dim];
        for (int i = 0; i < dim; i++) {
            lengths[i] = max[i] - min[i];
            if (lengths[i] < 0) return 0; // illegalMBR
        }

        // n-dimensional surface area: 2 * Σ (∏ (length except one-dimensional))
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double product = 1.0;
            for (int j = 0; j < dim; j++) {
                if (j != i) product *= lengths[j];
            }
            sum += product;
        }

        return 2 * sum;
    }


    // calculate the volume of mbr
    public double computeVolume() {
        double[] min = this.pointMin.data;
        double[] max = this.pointMax.data;
        double volume = 1.0;

        for (int i = 0; i < D; i++) {
            double length = (max[i] - min[i] + 1) / metaLength[i];
            if (length < 0) return 0; // illegal mbr
            volume *= length;
        }

        return volume;
    }

}
