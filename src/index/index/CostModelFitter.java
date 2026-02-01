package index;

import java.util.ArrayList;
import java.util.List;

public class CostModelFitter {

    public static class FitResult {
        public final double beta;
        public final double gamma;

        public final long n;
        public final double sse;  // sum squared error
        public final double rmse; // sqrt(sse/n)
        public final double mae;  // mean absolute error
        public final double r2;   // coefficient of determination

        public FitResult(double beta, double gamma, long n, double sse, double rmse, double mae, double r2) {
            this.beta = beta;
            this.gamma = gamma;
            this.n = n;
            this.sse = sse;
            this.rmse = rmse;
            this.mae = mae;
            this.r2 = r2;
        }

        @Override
        public String toString() {
            return String.format(
                    "FitResult{beta=%.6f, gamma=%.6f, n=%d, rmse=%.4f, mae=%.4f, r2=%.6f, sse=%.4f}",
                    beta, gamma, n, rmse, mae, r2, sse
            );
        }
    }

    /**
     * 拟合：t ≈ beta*x + gamma*z
     * info[0]=t(queryTime), info[1]=x(nodeSize), info[2]=z(resSize)
     *
     * [Σx^2  Σxz] [beta ] = [Σxt]
     * [Σxz  Σz^2] [gamma]   [Σzt]
     */
    public static FitResult fitBetaGamma(List<Long[]> infos) {
        if (infos == null || infos.isEmpty()) {
            throw new IllegalArgumentException("infos is null or empty");
        }

        long n = 0;
        double sxx = 0.0; // Σ x^2
        double szz = 0.0; // Σ z^2
        double sxz = 0.0; // Σ xz
        double sxt = 0.0; // Σ xt
        double szt = 0.0; // Σ zt

        // 为了算 R^2 需要 mean(t)
        double sumT = 0.0;

        for (Long[] info : infos) {
            if (info == null || info.length < 3 || info[0] == null || info[1] == null || info[2] == null) {
                continue; // 跳过脏数据
            }
            double t = info[0];
            double x = info[1];
            double z = info[2];

            sxx += x * x;
            szz += z * z;
            sxz += x * z;
            sxt += x * t;
            szt += z * t;

            sumT += t;
            n++;
        }

        if (n < 2) {
            throw new IllegalArgumentException("valid sample size < 2");
        }

        // 解 2x2 线性方程
        double det = sxx * szz - sxz * sxz;
        if (Math.abs(det) < 1e-12) {
            // x 和 z 近似共线（或全为0）无法稳定拟合
            throw new IllegalStateException("Singular/ill-conditioned system: det ~ 0. Check nodeSize/resSize collinearity.");
        }

        double beta = (sxt * szz - szt * sxz) / det;
        double gamma = (szt * sxx - sxt * sxz) / det;


        double meanT = sumT / n;
        double sse = 0.0;
        double mae = 0.0;
        double sst = 0.0;

        for (Long[] info : infos) {
            if (info == null || info.length < 3 || info[0] == null || info[1] == null || info[2] == null) {
                continue;
            }
            double t = info[0];
            double x = info[1];
            double z = info[2];

            double pred = beta * x + gamma * z;
            double err = t - pred;

            sse += err * err;
            mae += Math.abs(err);
            sst += (t - meanT) * (t - meanT);
        }

        mae /= n;
        double rmse = Math.sqrt(sse / n);
        double r2 = (sst <= 1e-12) ? 1.0 : (1.0 - sse / sst);

        return new FitResult(beta, gamma, n, sse, rmse, mae, r2);
    }

    public static FitResult fitCombined(List<Long[]> leafInfo, List<Long[]> nonLeafInfo) {
        List<Long[]> all = new ArrayList<>();
        if (leafInfo != null) all.addAll(leafInfo);
        if (nonLeafInfo != null) all.addAll(nonLeafInfo);
        return fitBetaGamma(all);
    }
}
