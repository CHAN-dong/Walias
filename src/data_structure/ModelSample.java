package data_structure;

import static data_structure.Parameters.*;

public class ModelSample {
//    public int OiQ = 0;//objectSizeInQueries
//    public int querySize = 0;
//    public int objectSize = 0;
//    public double perimeter = 0;
//    public double volume = 0;
    public double[] features = new double[featureSize];

    public double weight;

    public void setWeight(double height, double t) {
        this.weight = (1 - (height - 1) / H) * (1 - (t - 1) / T);
    }

    public void setCost(double cost) {
        features[0] = cost;
    }
    public double getCost() {
        return features[0];
    }

    public void addOiQFeature(int oiq) {
        features[1] += oiq;
    }

    public void setOiQFeature(int oiq) {
        features[1] = oiq;
    }

    public void addQuerySizeToFeature(int querySize) {
        features[2] += querySize;
    }

    public void setQuerySizeToFeature(int querySize) {
        features[2] = querySize;
    }

    public void setObjectSizeToFeature(int objectSize) {
        features[3] = objectSize;
    }

    public void setVolumeToFeature(Query query) {
        features[4] = query.computeVolume();
    }

    public void setPerimeterToFeature(Query query) {
        features[5] = query.computePerimeter();
    }

}
