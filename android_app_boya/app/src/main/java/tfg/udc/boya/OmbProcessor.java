package tfg.udc.boya;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import java.util.Arrays;

public class OmbProcessor {

    public enum Mode {
        OMB_10HZ,
        NATIVE_FS
    }

    private static final double OMB_FS_TARGET_HZ = 10.0;
    private static final double OMB_GRAVITY = 9.81;
    private static final int OMB_FFT_LENGTH = 2048;
    private static final int OMB_FFT_OVERLAP = 512;
    private static final int OMB_START_MARGIN = 50;
    private static final int OMB_SEGMENTS_TARGET = 21;
    private static final int OMB_BIN_MIN = 9;
    private static final int OMB_BIN_MAX = 64; // exclusive
    private static final double OMB_HANNING_ENERGY_SCALING = 1.63;
    private static final double OMB_BAND_MIN_HZ = (OMB_BIN_MIN * OMB_FS_TARGET_HZ) / OMB_FFT_LENGTH;
    private static final double OMB_BAND_MAX_HZ = ((OMB_BIN_MAX - 1) * OMB_FS_TARGET_HZ) / OMB_FFT_LENGTH;

    private static final FastFourierTransformer FFT = new FastFourierTransformer(DftNormalization.STANDARD);

    public OmbResult process(SessionData data, Mode mode) {
        boolean useNativeFs = mode == Mode.NATIVE_FS;

        double fsIn = estimateFs(data.tSec);
        if (!(fsIn > 0.0)) {
            throw new IllegalArgumentException("No se pudo estimar fs de entrada");
        }

        double fsProc = useNativeFs ? fsIn : OMB_FS_TARGET_HZ;
        double[] tUniform = buildUniformTime(data.durationSec(), fsProc);
        double[] azUniform = interpolateLinear(data.tSec, data.az, tUniform);

        int n = azUniform.length;
        int maxSegments = (n - OMB_START_MARGIN - OMB_FFT_LENGTH) / OMB_FFT_OVERLAP + 1;
        if (maxSegments < 1) {
            throw new IllegalArgumentException("Captura demasiado corta para OMB");
        }

        int segments = useNativeFs ? maxSegments : Math.min(OMB_SEGMENTS_TARGET, maxSegments);
        int usedLen = OMB_START_MARGIN + (segments - 1) * OMB_FFT_OVERLAP + OMB_FFT_LENGTH;
        int base = n - usedLen;
        if (base < 0) {
            throw new IllegalArgumentException("Ventana de analisis invalida");
        }

        double[] signal = Arrays.copyOfRange(azUniform, base, base + usedLen);

        double df = fsProc / OMB_FFT_LENGTH;
        int binMin;
        int binMaxInclusive;
        if (useNativeFs) {
            binMin = Math.max(1, (int) Math.ceil(OMB_BAND_MIN_HZ / df));
            binMaxInclusive = Math.min((OMB_FFT_LENGTH / 2) - 1, (int) Math.floor(OMB_BAND_MAX_HZ / df));
            if (binMaxInclusive < binMin) {
                throw new IllegalArgumentException("Banda OMB invalida para fs nativa");
            }
        } else {
            binMin = OMB_BIN_MIN;
            binMaxInclusive = OMB_BIN_MAX - 1;
        }

        int binsCount = binMaxInclusive - binMin + 1;
        double[] welch = new double[binsCount];
        double[] freqHz = new double[binsCount];
        double[] setaSpectrum = new double[binsCount];

        double[] window = new double[OMB_FFT_LENGTH];
        for (int i = 0; i < OMB_FFT_LENGTH; i++) {
            double s = Math.sin(Math.PI * i / (double) OMB_FFT_LENGTH);
            window[i] = OMB_HANNING_ENERGY_SCALING * s * s;
        }

        for (int seg = 0; seg < segments; seg++) {
            int start = OMB_START_MARGIN + seg * OMB_FFT_OVERLAP;
            double[] x = new double[OMB_FFT_LENGTH];
            for (int i = 0; i < OMB_FFT_LENGTH; i++) {
                x[i] = (signal[start + i] - OMB_GRAVITY) * window[i];
            }

            Complex[] out = FFT.transform(x, TransformType.FORWARD);

            for (int b = 0; b < binsCount; b++) {
                int k = binMin + b;
                double real = out[k].getReal();
                double imag = out[k].getImaginary();
                double energy = real * real + imag * imag;
                energy /= segments;
                welch[b] += 2.0 * energy / OMB_FFT_LENGTH / OMB_FFT_LENGTH / df;
            }
        }

        double m0 = 0.0;
        double m2 = 0.0;
        double m4 = 0.0;

        double fp = Double.NaN;
        double maxSeta = Double.NEGATIVE_INFINITY;

        for (int b = 0; b < binsCount; b++) {
            double f = (binMin + b) * df;
            freqHz[b] = f;
            double omega = 2.0 * Math.PI * f;
            double omega4 = omega * omega * omega * omega;
            double seta = welch[b] / omega4;
            setaSpectrum[b] = seta;

            m0 += seta * df;
            m2 += f * f * seta * df;
            m4 += f * f * f * f * seta * df;

            if (seta > maxSeta) {
                maxSeta = seta;
                fp = f;
            }
        }

        double sqrtM0 = Math.sqrt(Math.max(m0, 0.0));
        double sqrtM2 = Math.sqrt(Math.max(m2, 0.0));
        double sqrtM4 = Math.sqrt(Math.max(m4, 0.0));

        double hs = 4.0 * sqrtM0;
        double tzRawHz = sqrtM0 > 0.0 ? (sqrtM2 / sqrtM0) : Double.NaN;
        double tcRawHz = sqrtM2 > 0.0 ? (sqrtM4 / sqrtM2) : Double.NaN;

        double tzSec = (Double.isFinite(tzRawHz) && tzRawHz > 0.0) ? 1.0 / tzRawHz : Double.NaN;
        double tcSec = (Double.isFinite(tcRawHz) && tcRawHz > 0.0) ? 1.0 / tcRawHz : Double.NaN;
        double tpSec = (Double.isFinite(fp) && fp > 0.0) ? 1.0 / fp : Double.NaN;

        int previewSamples = Math.min(azUniform.length, (int) Math.round(120.0 * fsProc));
        int previewStart = azUniform.length - previewSamples;
        double t0Preview = tUniform[previewStart];
        double[] previewTime = new double[previewSamples];
        double[] previewAzDyn = new double[previewSamples];
        for (int i = 0; i < previewSamples; i++) {
            previewTime[i] = tUniform[previewStart + i] - t0Preview;
            previewAzDyn[i] = azUniform[previewStart + i] - OMB_GRAVITY;
        }

        String modeLabel = useNativeFs ? "Nativo (fs real)" : "OMB 10Hz";

        return new OmbResult(
                modeLabel,
                fsProc,
                data.sampleCount(),
                data.durationSec(),
                azUniform.length,
                segments,
                hs,
                tzSec,
                tcSec,
                tpSec,
                freqHz,
                setaSpectrum,
                previewTime,
                previewAzDyn
        );
    }

    private double estimateFs(double[] tSec) {
        if (tSec.length < 3) {
            return Double.NaN;
        }
        double[] dt = new double[tSec.length - 1];
        int n = 0;
        for (int i = 1; i < tSec.length; i++) {
            double d = tSec[i] - tSec[i - 1];
            if (d > 0.0 && Double.isFinite(d)) {
                dt[n++] = d;
            }
        }
        if (n < 3) {
            return Double.NaN;
        }
        Arrays.sort(dt, 0, n);
        double median = (n % 2 == 0) ? (dt[n / 2 - 1] + dt[n / 2]) * 0.5 : dt[n / 2];
        return (median > 0.0) ? (1.0 / median) : Double.NaN;
    }

    private double[] buildUniformTime(double durationSec, double fsTarget) {
        if (durationSec <= 0.0) {
            throw new IllegalArgumentException("Duracion invalida");
        }
        if (fsTarget <= 0.0) {
            throw new IllegalArgumentException("fs invalida");
        }

        double dt = 1.0 / fsTarget;
        int n = (int) Math.floor(durationSec / dt) + 1;
        if (n < 100) {
            throw new IllegalArgumentException("Muy pocas muestras tras remuestreo");
        }

        double[] t = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = i * dt;
        }
        return t;
    }

    private double[] interpolateLinear(double[] t, double[] y, double[] tQuery) {
        double[] out = new double[tQuery.length];

        int j = 0;
        int n = t.length;

        for (int i = 0; i < tQuery.length; i++) {
            double tq = tQuery[i];

            while (j + 1 < n && t[j + 1] < tq) {
                j++;
            }

            if (j + 1 >= n) {
                out[i] = y[n - 1];
                continue;
            }

            double t0 = t[j];
            double t1 = t[j + 1];
            double y0 = y[j];
            double y1 = y[j + 1];

            double alpha = (t1 > t0) ? (tq - t0) / (t1 - t0) : 0.0;
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 1.0) alpha = 1.0;

            out[i] = y0 + alpha * (y1 - y0);
        }

        return out;
    }
}
