package es.udc.boya;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final int REQ_WRITE = 1001;

    private static final String PREFS = "boya_prefs";
    private static final String KEY_IP = "server_ip";
    private static final String KEY_PORT = "server_port";

    // Ajustado a tu red compartida actual.
    private static final String DEFAULT_IP = "192.168.137.1";
    private static final String DEFAULT_PORT = "8001";

    private static final long SEND_MIN_INTERVAL_MS = 20;

    private SensorManager sensorManager;
    private Sensor accel, gyro;

    private TextView tvStatus;
    private Button btnStart, btnStop;

    private EditText etServerIp, etServerPort;

    private boolean recording = false;

    private BufferedWriter writer;
    private long t0ElapsedNs = 0L;

    private float ax, ay, az;
    private float gx, gy, gz;

    private long lastSendMs = 0L;

    private volatile String serverUrl = null;

    // Evita crear un thread por muestra.
    private final ExecutorService httpExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        btnStart = findViewById(R.id.btnStart);
        btnStop = findViewById(R.id.btnStop);

        etServerIp = findViewById(R.id.etServerIp);
        etServerPort = findViewById(R.id.etServerPort);

        btnStart.setEnabled(true);
        btnStop.setEnabled(false);
        tvStatus.setText("Listo. Pulsa Start.");

        SharedPreferences sp = getSharedPreferences(PREFS, MODE_PRIVATE);
        etServerIp.setText(sp.getString(KEY_IP, DEFAULT_IP));
        etServerPort.setText(sp.getString(KEY_PORT, DEFAULT_PORT));

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        if (accel == null || gyro == null) {
            tvStatus.setText("ERROR: el movil no tiene acelerometro o giroscopio.");
            btnStart.setEnabled(false);
            return;
        }

        btnStart.setOnClickListener(v -> {
            tvStatus.setText("Start pulsado...");

            if (needsLegacyWritePermission() && !hasWritePermission()) {
                tvStatus.setText("Pidiendo permiso de escritura...");
                requestWritePermission();
                return;
            }

            String ip = etServerIp.getText().toString().trim();
            String port = etServerPort.getText().toString().trim();

            if (ip.isEmpty()) ip = DEFAULT_IP;
            if (port.isEmpty()) port = DEFAULT_PORT;

            serverUrl = "http://" + ip + ":" + port + "/data";

            sp.edit().putString(KEY_IP, ip).putString(KEY_PORT, port).apply();

            Toast.makeText(this, "Grabando y enviando a " + serverUrl, Toast.LENGTH_SHORT).show();
            Log.d("BOYA", "Start -> serverUrl=" + serverUrl);

            startRecording();
        });

        btnStop.setOnClickListener(v -> stopRecording());
    }

    private boolean needsLegacyWritePermission() {
        return Build.VERSION.SDK_INT <= Build.VERSION_CODES.P;
    }

    private boolean hasWritePermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestWritePermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                REQ_WRITE);
    }

    private void startRecording() {
        if (recording) {
            tvStatus.setText("Ya estaba grabando.");
            return;
        }

        if (serverUrl == null || serverUrl.isEmpty()) {
            tvStatus.setText("ERROR: serverUrl vacia.");
            Toast.makeText(this, "URL del servidor no valida", Toast.LENGTH_LONG).show();
            return;
        }

        File outFile = createOutputFile();
        if (outFile == null) {
            tvStatus.setText("ERROR: No pude crear el archivo CSV.");
            Toast.makeText(this, "No pude crear el archivo CSV", Toast.LENGTH_LONG).show();
            return;
        }

        try {
            writer = new BufferedWriter(new FileWriter(outFile));
            writer.write("t_ms,ax,ay,az,gx,gy,gz\n");

            recording = true;
            btnStart.setEnabled(false);
            btnStop.setEnabled(true);

            etServerIp.setEnabled(false);
            etServerPort.setEnabled(false);

            t0ElapsedNs = SystemClock.elapsedRealtimeNanos();
            lastSendMs = 0L;

            tvStatus.setText("GRABANDO\nCSV:\n" + outFile.getAbsolutePath() + "\n\nHTTP:\n" + serverUrl);

            sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME);
            sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME);

        } catch (IOException e) {
            recording = false;
            tvStatus.setText("ERROR abriendo CSV: " + e.getMessage());
            Log.e("BOYA", "Error abriendo CSV", e);
        }
    }

    private void stopRecording() {
        if (!recording) {
            tvStatus.setText("No estaba grabando.");
            return;
        }

        sensorManager.unregisterListener(this);
        recording = false;

        btnStart.setEnabled(true);
        btnStop.setEnabled(false);

        etServerIp.setEnabled(true);
        etServerPort.setEnabled(true);

        try {
            if (writer != null) {
                writer.flush();
                writer.close();
            }
            tvStatus.setText("Parado. CSV guardado.");
        } catch (IOException e) {
            tvStatus.setText("Error cerrando CSV: " + e.getMessage());
        } finally {
            writer = null;
        }
    }

    private File createOutputFile() {
        File base = getExternalFilesDir(null);
        if (base == null) return null;

        File dir = new File(base, "boya");
        if (!dir.exists() && !dir.mkdirs()) return null;

        String name = String.format(Locale.US, "imu_%d.csv", System.currentTimeMillis());
        return new File(dir, name);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!recording || writer == null) return;

        int type = event.sensor.getType();

        if (type == Sensor.TYPE_ACCELEROMETER) {
            ax = event.values[0];
            ay = event.values[1];
            az = event.values[2];
        } else if (type == Sensor.TYPE_GYROSCOPE) {
            gx = event.values[0];
            gy = event.values[1];
            gz = event.values[2];
        } else {
            return;
        }

        long tMs = (SystemClock.elapsedRealtimeNanos() - t0ElapsedNs) / 1_000_000L;

        try {
            writer.write(String.format(Locale.US,
                    "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    tMs, ax, ay, az, gx, gy, gz));
        } catch (IOException e) {
            Log.e("BOYA", "Error escribiendo CSV", e);
            stopRecording();
            return;
        }

        if (type != Sensor.TYPE_ACCELEROMETER) return;

        long nowMs = SystemClock.elapsedRealtime();
        if (nowMs - lastSendMs < SEND_MIN_INTERVAL_MS) return;
        lastSendMs = nowMs;

        final long tMsFinal = tMs;
        final float axFinal = ax, ayFinal = ay, azFinal = az;
        final float gxFinal = gx, gyFinal = gy, gzFinal = gz;
        final String urlFinal = serverUrl;

        httpExecutor.execute(() -> sendHttpSample(urlFinal, tMsFinal, axFinal, ayFinal, azFinal, gxFinal, gyFinal, gzFinal));
    }

    private void sendHttpSample(String serverUrl, long tMs, float ax, float ay, float az,
                                float gx, float gy, float gz) {
        HttpURLConnection conn = null;
        try {
            URL url = new URL(serverUrl);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(2000);
            conn.setReadTimeout(2000);
            conn.setDoOutput(true);

            JSONObject json = new JSONObject();
            json.put("t", tMs);
            json.put("ax", ax);
            json.put("ay", ay);
            json.put("az", az);
            json.put("gx", gx);
            json.put("gy", gy);
            json.put("gz", gz);

            byte[] payload = json.toString().getBytes();
            try (OutputStream os = conn.getOutputStream()) {
                os.write(payload);
                os.flush();
            }

            int code = conn.getResponseCode();
            if (code != 200) {
                Log.w("HTTP", "Respuesta servidor: " + code);
            }
        } catch (Exception e) {
            Log.e("HTTP", "ERROR enviando HTTP a " + serverUrl, e);
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    @Override
    protected void onPause() {
        super.onPause();
        if (recording) stopRecording();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        httpExecutor.shutdownNow();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_WRITE &&
                grantResults.length > 0 &&
                grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            btnStart.performClick();
        } else if (requestCode == REQ_WRITE) {
            tvStatus.setText("Permiso de escritura DENEGADO.");
        }
    }
}
