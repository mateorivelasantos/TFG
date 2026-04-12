#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

typedef struct {
    double *t;
    double *ax;
    double *ay;
    double *az;
    double *gx;
    double *gy;
    double *gz;
    size_t n;
    size_t cap;
} Samples;

typedef struct {
    double hs;
    double tz;
    double tc;
    double tp;
    double fp;
    double fs_proc;
    int n_uniform;
    int segments;
    int ok;
    const char *error;
} Metrics;

static int reserve_samples(Samples *s, size_t want) {
    if (want <= s->cap) return 1;
    size_t new_cap = s->cap ? s->cap * 2 : 4096;
    if (new_cap < want) new_cap = want;

    double *t = (double *)realloc(s->t, new_cap * sizeof(double));
    double *ax = (double *)realloc(s->ax, new_cap * sizeof(double));
    double *ay = (double *)realloc(s->ay, new_cap * sizeof(double));
    double *az = (double *)realloc(s->az, new_cap * sizeof(double));
    double *gx = (double *)realloc(s->gx, new_cap * sizeof(double));
    double *gy = (double *)realloc(s->gy, new_cap * sizeof(double));
    double *gz = (double *)realloc(s->gz, new_cap * sizeof(double));
    if (!t || !ax || !ay || !az || !gx || !gy || !gz) return 0;

    s->t = t; s->ax = ax; s->ay = ay; s->az = az;
    s->gx = gx; s->gy = gy; s->gz = gz;
    s->cap = new_cap;
    return 1;
}

static void free_samples(Samples *s) {
    free(s->t); free(s->ax); free(s->ay); free(s->az);
    free(s->gx); free(s->gy); free(s->gz);
    memset(s, 0, sizeof(*s));
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double median_positive_dt(const double *t, size_t n) {
    if (n < 3) return NAN;
    double *dt = (double *)malloc((n - 1) * sizeof(double));
    if (!dt) return NAN;
    size_t m = 0;
    for (size_t i = 1; i < n; ++i) {
        double d = t[i] - t[i - 1];
        if (d > 0.0 && isfinite(d)) dt[m++] = d;
    }
    if (m < 3) {
        free(dt);
        return NAN;
    }
    qsort(dt, m, sizeof(double), cmp_double);
    double med = (m % 2 == 0) ? 0.5 * (dt[m / 2 - 1] + dt[m / 2]) : dt[m / 2];
    free(dt);
    return med;
}

static int load_csv(const char *path, Samples *s) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[512];
    int first = 1;
    while (fgets(line, sizeof(line), f)) {
        if (first) {
            first = 0;
            if (strncmp(line, "t_ms", 4) == 0) continue;
        }

        double t_ms, ax, ay, az, gx, gy, gz;
        if (sscanf(line, " %lf , %lf , %lf , %lf , %lf , %lf , %lf", &t_ms, &ax, &ay, &az, &gx, &gy, &gz) != 7) {
            continue;
        }

        double t_s = t_ms / 1000.0;
        if (s->n > 0 && !(t_s > s->t[s->n - 1])) {
            continue;
        }

        if (!reserve_samples(s, s->n + 1)) {
            fclose(f);
            return 0;
        }

        size_t i = s->n++;
        s->t[i] = t_s;
        s->ax[i] = ax;
        s->ay[i] = ay;
        s->az[i] = az;
        s->gx[i] = gx;
        s->gy[i] = gy;
        s->gz[i] = gz;
    }

    fclose(f);
    return s->n >= 100;
}

static void quat_normalize(double *q0, double *q1, double *q2, double *q3) {
    double n = sqrt((*q0) * (*q0) + (*q1) * (*q1) + (*q2) * (*q2) + (*q3) * (*q3));
    if (n <= 0.0 || !isfinite(n)) {
        *q0 = 1.0; *q1 = 0.0; *q2 = 0.0; *q3 = 0.0;
        return;
    }
    *q0 /= n; *q1 /= n; *q2 /= n; *q3 /= n;
}

static void madgwick_update_imu(double *q0, double *q1, double *q2, double *q3,
                                double gx, double gy, double gz,
                                double ax, double ay, double az,
                                double beta, double dt) {
    double recipNorm;
    double s0, s1, s2, s3;
    double qDot1, qDot2, qDot3, qDot4;
    double _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2;
    double q0q0, q1q1, q2q2, q3q3;

    qDot1 = 0.5 * (-(*q1) * gx - (*q2) * gy - (*q3) * gz);
    qDot2 = 0.5 * ((*q0) * gx + (*q2) * gz - (*q3) * gy);
    qDot3 = 0.5 * ((*q0) * gy - (*q1) * gz + (*q3) * gx);
    qDot4 = 0.5 * ((*q0) * gz + (*q1) * gy - (*q2) * gx);

    if (!(ax == 0.0 && ay == 0.0 && az == 0.0)) {
        recipNorm = 1.0 / sqrt(ax * ax + ay * ay + az * az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        _2q0 = 2.0 * (*q0);
        _2q1 = 2.0 * (*q1);
        _2q2 = 2.0 * (*q2);
        _2q3 = 2.0 * (*q3);
        _4q0 = 4.0 * (*q0);
        _4q1 = 4.0 * (*q1);
        _4q2 = 4.0 * (*q2);
        _8q1 = 8.0 * (*q1);
        _8q2 = 8.0 * (*q2);
        q0q0 = (*q0) * (*q0);
        q1q1 = (*q1) * (*q1);
        q2q2 = (*q2) * (*q2);
        q3q3 = (*q3) * (*q3);

        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * (*q1) - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
        s2 = 4.0 * q0q0 * (*q2) + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
        s3 = 4.0 * q1q1 * (*q3) - _2q1 * ax + 4.0 * q2q2 * (*q3) - _2q2 * ay;

        recipNorm = 1.0 / sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
        s0 *= recipNorm;
        s1 *= recipNorm;
        s2 *= recipNorm;
        s3 *= recipNorm;

        qDot1 -= beta * s0;
        qDot2 -= beta * s1;
        qDot3 -= beta * s2;
        qDot4 -= beta * s3;
    }

    *q0 += qDot1 * dt;
    *q1 += qDot2 * dt;
    *q2 += qDot3 * dt;
    *q3 += qDot4 * dt;

    quat_normalize(q0, q1, q2, q3);
}

static void euler_to_quat(double roll, double pitch, double yaw,
                          double *q0, double *q1, double *q2, double *q3) {
    double cr = cos(roll * 0.5), sr = sin(roll * 0.5);
    double cp = cos(pitch * 0.5), sp = sin(pitch * 0.5);
    double cy = cos(yaw * 0.5), sy = sin(yaw * 0.5);

    *q0 = cr * cp * cy + sr * sp * sy;
    *q1 = sr * cp * cy - cr * sp * sy;
    *q2 = cr * sp * cy + sr * cp * sy;
    *q3 = cr * cp * sy - sr * sp * cy;
    quat_normalize(q0, q1, q2, q3);
}

static int compute_adown_madgwick(const Samples *s, double beta, double *adown) {
    if (!s || s->n < 3 || !adown) return 0;

    double dt_med = median_positive_dt(s->t, s->n);
    if (!(dt_med > 0.0)) return 0;

    double ax0 = s->ax[0], ay0 = s->ay[0], az0 = s->az[0];
    double roll0 = atan2(ay0, az0);
    double pitch0 = atan2(-ax0, sqrt(ay0 * ay0 + az0 * az0));

    double q0, q1, q2, q3;
    euler_to_quat(roll0, pitch0, 0.0, &q0, &q1, &q2, &q3);

    const double d2r = PI / 180.0;
    double mean = 0.0;

    for (size_t i = 0; i < s->n; ++i) {
        double dt = (i == 0) ? dt_med : (s->t[i] - s->t[i - 1]);
        if (!(dt > 0.0) || !isfinite(dt)) dt = dt_med;

        double gx = s->gx[i] * d2r;
        double gy = s->gy[i] * d2r;
        double gz = s->gz[i] * d2r;

        madgwick_update_imu(&q0, &q1, &q2, &q3, gx, gy, gz, s->ax[i], s->ay[i], s->az[i], beta, dt);

        double gbx = 2.0 * (q1 * q3 - q0 * q2);
        double gby = 2.0 * (q0 * q1 + q2 * q3);
        double gbz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

        adown[i] = s->ax[i] * gbx + s->ay[i] * gby + s->az[i] * gbz;
        mean += adown[i];
    }

    mean /= (double)s->n;
    if (mean < 0.0) {
        for (size_t i = 0; i < s->n; ++i) adown[i] = -adown[i];
    }

    return 1;
}

static int compute_adown_accel_projection_instant(const Samples *s, double *adown) {
    if (!s || s->n < 3 || !adown) return 0;

    double mean = 0.0;
    for (size_t i = 0; i < s->n; ++i) {
        double ax = s->ax[i], ay = s->ay[i], az = s->az[i];
        double an = sqrt(ax * ax + ay * ay + az * az);
        if (!(an > 1e-12) || !isfinite(an)) return 0;
        double ux = ax / an, uy = ay / an, uz = az / an;
        adown[i] = ax * ux + ay * uy + az * uz;
        mean += adown[i];
    }

    mean /= (double)s->n;
    if (mean < 0.0) {
        for (size_t i = 0; i < s->n; ++i) adown[i] = -adown[i];
    }

    return 1;
}

static int compute_adown_accel_projection_lpf(const Samples *s, double alpha, double *adown) {
    if (!s || s->n < 3 || !adown) return 0;
    if (!(alpha >= 0.0 && alpha < 1.0)) return 0;

    double gx = s->ax[0], gy = s->ay[0], gz = s->az[0];
    double mean = 0.0;

    for (size_t i = 0; i < s->n; ++i) {
        double ax = s->ax[i], ay = s->ay[i], az = s->az[i];
        if (i > 0) {
            gx = alpha * gx + (1.0 - alpha) * ax;
            gy = alpha * gy + (1.0 - alpha) * ay;
            gz = alpha * gz + (1.0 - alpha) * az;
        }

        double gn = sqrt(gx * gx + gy * gy + gz * gz);
        if (!(gn > 1e-12) || !isfinite(gn)) return 0;
        double ux = gx / gn, uy = gy / gn, uz = gz / gn;
        adown[i] = ax * ux + ay * uy + az * uz;
        mean += adown[i];
    }

    mean /= (double)s->n;
    if (mean < 0.0) {
        for (size_t i = 0; i < s->n; ++i) adown[i] = -adown[i];
    }

    return 1;
}

static int resample_linear(const double *t, const double *x, size_t n, double fs,
                           double **t_u, double **x_u, int *n_u) {
    if (!t || !x || n < 3 || fs <= 0.0) return 0;
    double duration = t[n - 1] - t[0];
    if (!(duration > 0.0)) return 0;

    double dt = 1.0 / fs;
    int m = (int)floor(duration / dt) + 1;
    if (m < 100) return 0;

    double *tu = (double *)malloc((size_t)m * sizeof(double));
    double *xu = (double *)malloc((size_t)m * sizeof(double));
    if (!tu || !xu) {
        free(tu); free(xu);
        return 0;
    }

    size_t j = 0;
    for (int i = 0; i < m; ++i) {
        double tq = (double)i * dt;
        tu[i] = tq;
        while (j + 1 < n && (t[j + 1] - t[0]) < tq) ++j;
        if (j + 1 >= n) {
            xu[i] = x[n - 1];
            continue;
        }
        double t0 = t[j] - t[0];
        double t1 = t[j + 1] - t[0];
        double y0 = x[j];
        double y1 = x[j + 1];
        double a = (t1 > t0) ? (tq - t0) / (t1 - t0) : 0.0;
        if (a < 0.0) a = 0.0;
        if (a > 1.0) a = 1.0;
        xu[i] = y0 + a * (y1 - y0);
    }

    *t_u = tu;
    *x_u = xu;
    *n_u = m;
    return 1;
}

static void accumulate_bin_energy(const double *x, int k, int nfft, double *energy) {
    double angle_inc = 2.0 * PI * (double)k / (double)nfft;
    double c_inc = cos(angle_inc);
    double s_inc = sin(angle_inc);
    double c = 1.0, s = 0.0;
    double re = 0.0, im = 0.0;

    for (int i = 0; i < nfft; ++i) {
        re += x[i] * c;
        im -= x[i] * s;
        double nc = c * c_inc - s * s_inc;
        double ns = s * c_inc + c * s_inc;
        c = nc;
        s = ns;
    }

    *energy = re * re + im * im;
}

static Metrics analyze_openmet_strict(const double *t, const double *adown, size_t n) {
    Metrics m = {0};

    const int NFFT = 2048;
    const int OVERLAP = 512;
    const int START_MARGIN = 50;
    const int SEGMENTS = 21;
    const int BIN_MIN = 9;
    const int BIN_MAX_EX = 64;
    const int TOTAL = NFFT * 6;
    const int MIN_REQ = TOTAL + 75;
    const int TARGET = TOTAL + 99;
    const double WIN_SCALE = 1.63;
    const double GRAV = 9.81;

    if (n < (size_t)MIN_REQ) {
        m.ok = 0;
        m.error = "serie demasiado corta para OpenMetBuoy estricto";
        return m;
    }

    double *tu = NULL, *xu = NULL;
    int nu = 0;
    if (!resample_linear(t, adown, n, 10.0, &tu, &xu, &nu)) {
        m.ok = 0;
        m.error = "fallo remuestreo a 10 Hz";
        return m;
    }

    if (nu < MIN_REQ) {
        free(tu); free(xu);
        m.ok = 0;
        m.error = "serie remuestreada demasiado corta";
        return m;
    }

    int offset = (nu > TARGET) ? (nu - TARGET) : 0;
    int required_used = START_MARGIN + (SEGMENTS - 1) * OVERLAP + NFFT;
    if ((nu - offset) < required_used) {
        free(tu); free(xu);
        m.ok = 0;
        m.error = "no hay muestras suficientes para 21 segmentos";
        return m;
    }

    int bins = BIN_MAX_EX - BIN_MIN;
    double *welch = (double *)calloc((size_t)bins, sizeof(double));
    double *window = (double *)malloc((size_t)NFFT * sizeof(double));
    double *xseg = (double *)malloc((size_t)NFFT * sizeof(double));
    if (!welch || !window || !xseg) {
        free(tu); free(xu); free(welch); free(window); free(xseg);
        m.ok = 0;
        m.error = "fallo reserva memoria";
        return m;
    }

    for (int i = 0; i < NFFT; ++i) {
        double s1 = sin(PI * (double)i / (double)NFFT);
        window[i] = WIN_SCALE * s1 * s1;
    }

    double df = 10.0 / (double)NFFT;

    for (int seg = 0; seg < SEGMENTS; ++seg) {
        int start = offset + START_MARGIN + seg * OVERLAP;
        for (int i = 0; i < NFFT; ++i) {
            xseg[i] = (xu[start + i] - GRAV) * window[i];
        }

        for (int b = 0; b < bins; ++b) {
            int k = BIN_MIN + b;
            double e = 0.0;
            accumulate_bin_energy(xseg, k, NFFT, &e);
            e /= (double)SEGMENTS;
            welch[b] += 2.0 * e / (double)NFFT / (double)NFFT / df;
        }
    }

    double m0 = 0.0, m2 = 0.0, m4 = 0.0;
    double max_seta = -1.0, fp = NAN;

    for (int b = 0; b < bins; ++b) {
        double f = (double)(BIN_MIN + b) * df;
        double w = 2.0 * PI * f;
        double w4 = w * w * w * w;
        double seta = welch[b] / w4;

        m0 += seta * df;
        m2 += f * f * seta * df;
        m4 += f * f * f * f * seta * df;

        if (seta > max_seta) {
            max_seta = seta;
            fp = f;
        }
    }

    double sm0 = sqrt(fmax(m0, 0.0));
    double sm2 = sqrt(fmax(m2, 0.0));
    double sm4 = sqrt(fmax(m4, 0.0));

    m.hs = 4.0 * sm0;
    m.tz = (sm0 > 0.0 && sm2 > 0.0) ? 1.0 / (sm2 / sm0) : NAN;
    m.tc = (sm2 > 0.0 && sm4 > 0.0) ? 1.0 / (sm4 / sm2) : NAN;
    m.tp = (isfinite(fp) && fp > 0.0) ? 1.0 / fp : NAN;
    m.fp = fp;
    m.fs_proc = 10.0;
    m.n_uniform = nu;
    m.segments = SEGMENTS;
    m.ok = 1;

    free(tu); free(xu); free(welch); free(window); free(xseg);
    return m;
}

static void print_metrics(const char *label, Metrics m) {
    if (!m.ok) {
        printf("%s -> ERROR: %s\n", label, m.error ? m.error : "desconocido");
        return;
    }
    printf("%s -> Hs=%.5f m | Tz=%.5f s | Tc=%.5f s | Tp=%.5f s | fp=%.5f Hz | fs=%.2f | N=%d | seg=%d\n",
           label, m.hs, m.tz, m.tc, m.tp, m.fp, m.fs_proc, m.n_uniform, m.segments);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: %s <csv_android> [beta] [alpha_lpf]\n", argv[0]);
        return 2;
    }

    const char *csv_path = argv[1];
    double beta = (argc >= 3) ? atof(argv[2]) : 0.08;
    if (!(beta > 0.0)) beta = 0.08;
    double alpha_lpf = (argc >= 4) ? atof(argv[3]) : 0.95;
    if (!(alpha_lpf >= 0.0 && alpha_lpf < 1.0)) alpha_lpf = 0.95;

    Samples s = {0};
    if (!load_csv(csv_path, &s)) {
        fprintf(stderr, "no se pudo leer CSV o hay pocas muestras: %s\n", csv_path);
        free_samples(&s);
        return 1;
    }

    double dt_med = median_positive_dt(s.t, s.n);
    printf("CSV=%s\n", csv_path);
    printf("muestras=%zu | dur=%.2f min | fs~%.4f Hz | beta=%.4f | alpha_lpf=%.3f\n",
           s.n,
           (s.t[s.n - 1] - s.t[0]) / 60.0,
           (dt_med > 0.0 ? 1.0 / dt_med : NAN),
           beta,
           alpha_lpf);

    double *adown_raw = (double *)malloc(s.n * sizeof(double));
    double *adown_imu = (double *)malloc(s.n * sizeof(double));
    double *adown_inst = (double *)malloc(s.n * sizeof(double));
    double *adown_lpf = (double *)malloc(s.n * sizeof(double));
    if (!adown_raw || !adown_imu || !adown_inst || !adown_lpf) {
        fprintf(stderr, "fallo memoria\n");
        free(adown_raw); free(adown_imu); free(adown_inst); free(adown_lpf); free_samples(&s);
        return 1;
    }

    for (size_t i = 0; i < s.n; ++i) adown_raw[i] = s.az[i];

    if (!compute_adown_madgwick(&s, beta, adown_imu)) {
        fprintf(stderr, "fallo en orientacion Madgwick\n");
        free(adown_raw); free(adown_imu); free(adown_inst); free(adown_lpf); free_samples(&s);
        return 1;
    }

    if (!compute_adown_accel_projection_instant(&s, adown_inst)) {
        fprintf(stderr, "fallo en proyeccion instantanea por acelerometro\n");
        free(adown_raw); free(adown_imu); free(adown_inst); free(adown_lpf); free_samples(&s);
        return 1;
    }

    if (!compute_adown_accel_projection_lpf(&s, alpha_lpf, adown_lpf)) {
        fprintf(stderr, "fallo en proyeccion LPF por acelerometro\n");
        free(adown_raw); free(adown_imu); free(adown_inst); free(adown_lpf); free_samples(&s);
        return 1;
    }

    Metrics m_raw = analyze_openmet_strict(s.t, adown_raw, s.n);
    Metrics m_imu = analyze_openmet_strict(s.t, adown_imu, s.n);
    Metrics m_inst = analyze_openmet_strict(s.t, adown_inst, s.n);
    Metrics m_lpf = analyze_openmet_strict(s.t, adown_lpf, s.n);

    print_metrics("OpenMet (az cruda)", m_raw);
    print_metrics("OpenMet (IMU orientada)", m_imu);
    print_metrics("OpenMet (proj accel instant)", m_inst);
    print_metrics("OpenMet (proj accel LPF)", m_lpf);

    if (m_raw.ok && m_imu.ok) {
        printf("delta | dHs=%.5f m | dTz=%.5f s | dTc=%.5f s | dTp=%.5f s\n",
               m_imu.hs - m_raw.hs,
               m_imu.tz - m_raw.tz,
               m_imu.tc - m_raw.tc,
               m_imu.tp - m_raw.tp);
    }
    if (m_raw.ok && m_inst.ok) {
        printf("delta_inst_vs_raw | dHs=%.5f m | dTz=%.5f s | dTc=%.5f s | dTp=%.5f s\n",
               m_inst.hs - m_raw.hs,
               m_inst.tz - m_raw.tz,
               m_inst.tc - m_raw.tc,
               m_inst.tp - m_raw.tp);
    }
    if (m_raw.ok && m_lpf.ok) {
        printf("delta_lpf_vs_raw | dHs=%.5f m | dTz=%.5f s | dTc=%.5f s | dTp=%.5f s\n",
               m_lpf.hs - m_raw.hs,
               m_lpf.tz - m_raw.tz,
               m_lpf.tc - m_raw.tc,
               m_lpf.tp - m_raw.tp);
    }

    free(adown_raw);
    free(adown_imu);
    free(adown_inst);
    free(adown_lpf);
    free_samples(&s);
    return 0;
}
