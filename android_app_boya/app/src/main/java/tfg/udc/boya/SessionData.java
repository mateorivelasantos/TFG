package tfg.udc.boya;

public class SessionData {
    public final double[] tSec;
    public final double[] az;

    public SessionData(double[] tSec, double[] az) {
        this.tSec = tSec;
        this.az = az;
    }

    public int sampleCount() {
        return tSec.length;
    }

    public double durationSec() {
        if (tSec.length < 2) {
            return 0.0;
        }
        return tSec[tSec.length - 1] - tSec[0];
    }
}
