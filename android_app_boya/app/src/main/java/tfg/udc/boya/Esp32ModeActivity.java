package tfg.udc.boya;

import android.graphics.Color;
import android.os.Bundle;
import android.text.InputType;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ArrayAdapter;
import android.widget.PopupMenu;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;
import android.app.AlertDialog;

import androidx.appcompat.app.AppCompatActivity;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class Esp32ModeActivity extends AppCompatActivity {
    private static final int ACTION_PROCESS = 1;
    private static final int ACTION_INFO = 2;
    private static final int ACTION_RENAME = 3;
    private static final int ACTION_DELETE = 4;

    private TextView tvEsp32StateScreen;
    private TextView tvEsp32Log;
    private TextView tvEsp32SelectedSession;
    private EditText etEsp32Host;
    private EditText etEsp32DurationSec;
    private EditText etEsp32SessionName;
    private RadioButton rbEsp32ModeOmb10;
    private RadioButton rbEsp32ModeNative;
    private LineChart chartEsp32Spectrum;
    private LineChart chartEsp32Series;
    private ListView lvEsp32Sessions;
    private ArrayAdapter<String> sessionsAdapter;
    private final List<File> listedSessionFiles = new ArrayList<>();
    private final List<String> logLines = new ArrayList<>();
    private static final int MAX_LOG_LINES = 120;
    private boolean isLogVisible = true;
    private final SessionRepository sessionRepository = new SessionRepository();
    private final OmbProcessor ombProcessor = new OmbProcessor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_esp32_mode);

        tvEsp32StateScreen = findViewById(R.id.tvEsp32StateScreen);
        tvEsp32Log = findViewById(R.id.tvEsp32Log);
        tvEsp32SelectedSession = findViewById(R.id.tvEsp32SelectedSession);
        etEsp32Host = findViewById(R.id.etEsp32Host);
        etEsp32DurationSec = findViewById(R.id.etEsp32DurationSec);
        etEsp32SessionName = findViewById(R.id.etEsp32SessionName);
        rbEsp32ModeOmb10 = findViewById(R.id.rbEsp32ModeOmb10);
        rbEsp32ModeNative = findViewById(R.id.rbEsp32ModeNative);
        chartEsp32Spectrum = findViewById(R.id.chartEsp32Spectrum);
        chartEsp32Series = findViewById(R.id.chartEsp32Series);
        lvEsp32Sessions = findViewById(R.id.lvEsp32Sessions);

        Button btnConnectEsp32 = findViewById(R.id.btnConnectEsp32);
        Button btnDisconnectEsp32 = findViewById(R.id.btnDisconnectEsp32);
        Button btnCheckEsp32 = findViewById(R.id.btnCheckEsp32);
        Button btnEsp32StartCapture = findViewById(R.id.btnEsp32StartCapture);
        Button btnEsp32StopCapture = findViewById(R.id.btnEsp32StopCapture);
        Button btnToggleEsp32Log = findViewById(R.id.btnToggleEsp32Log);
        Button btnRefreshEsp32Sessions = findViewById(R.id.btnRefreshEsp32Sessions);
        Button btnBackToMenu = findViewById(R.id.btnBackToMenu);

        sessionsAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, new ArrayList<>());
        lvEsp32Sessions.setAdapter(sessionsAdapter);
        setupCharts();

        btnConnectEsp32.setOnClickListener(v -> connectEsp32());
        btnDisconnectEsp32.setOnClickListener(v -> disconnectEsp32());
        btnCheckEsp32.setOnClickListener(v -> verifyConnection());
        btnEsp32StartCapture.setOnClickListener(v -> startCapture());
        btnEsp32StopCapture.setOnClickListener(v -> stopCapture());
        btnToggleEsp32Log.setOnClickListener(v -> {
            isLogVisible = !isLogVisible;
            tvEsp32Log.setVisibility(isLogVisible ? View.VISIBLE : View.GONE);
            btnToggleEsp32Log.setText(isLogVisible ? "Ocultar log" : "Mostrar log");
        });
        btnRefreshEsp32Sessions.setOnClickListener(v -> refreshSessionList());
        btnBackToMenu.setOnClickListener(v -> finish());

        lvEsp32Sessions.setOnItemClickListener((parent, view, position, id) -> {
            if (position >= 0 && position < listedSessionFiles.size()) {
                processAndShowSession(listedSessionFiles.get(position));
            }
        });

        lvEsp32Sessions.setOnItemLongClickListener((parent, view, position, id) -> {
            if (position >= 0 && position < listedSessionFiles.size()) {
                showSessionOptions(view, listedSessionFiles.get(position));
                return true;
            }
            return false;
        });

        lvEsp32Sessions.setOnTouchListener((v, event) -> {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                case MotionEvent.ACTION_MOVE:
                    v.getParent().requestDisallowInterceptTouchEvent(true);
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    v.getParent().requestDisallowInterceptTouchEvent(false);
                    break;
                default:
                    break;
            }
            return false;
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        updateConnectionState();
        refreshSessionList();
    }

    private void connectEsp32() {
        addLog("Intentando conectar a SSID " + Esp32ConnectionManager.SSID + "...");
        Esp32ConnectionManager.connect(this, (ok, message) -> runOnUiThread(() -> {
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
            addLog(message);
            updateConnectionState();
        }));
    }

    private void disconnectEsp32() {
        Esp32ConnectionManager.disconnect(this);
        addLog("Conexion ESP32 liberada en la app.");
        updateConnectionState();
    }

    private void verifyConnection() {
        if (!Esp32ConnectionManager.isConnected(this)) {
            addLog("No conectado al ESP32. Pulsa primero en Conectar.");
            Toast.makeText(this, "Conecta primero al ESP32", Toast.LENGTH_SHORT).show();
            updateConnectionState();
            return;
        }

        String host = readHost();
        addLog("Verificando ESP32 en " + host + "...");

        new Thread(() -> {
            try {
                String pingResp = doHttpGet(host, "/ping");
                String statusResp = doHttpGet(host, "/status");

                runOnUiThread(() -> {
                    addLog("GET /ping ok: " + compact(pingResp));
                    addLog("GET /status ok: " + compact(statusResp));
                    updateConnectionState();
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    addLog("Fallo verificando ESP32: " + String.valueOf(e.getMessage()));
                    Toast.makeText(this, "No se pudo verificar el ESP32", Toast.LENGTH_SHORT).show();
                    updateConnectionState();
                });
            }
        }).start();
    }

    private void startCapture() {
        if (!Esp32ConnectionManager.isConnected(this)) {
            addLog("No conectado al ESP32. Pulsa primero en Conectar.");
            Toast.makeText(this, "Conecta primero al ESP32", Toast.LENGTH_SHORT).show();
            return;
        }

        String host = readHost();
        int durationSec = readDuration();
        String sessionName = etEsp32SessionName.getText().toString().trim();
        if (sessionName.isEmpty()) {
            sessionName = "esp32_prueba";
        }
        final String finalSessionName = sessionName;
        String path = String.format(
                Locale.US,
                "/capture/start?duration=%d&name=%s",
                durationSec,
                urlEncodeSafe(finalSessionName));

        addLog("Solicitando inicio de captura ESP32...");
        new Thread(() -> {
            try {
                String resp = doHttpGetWithRetry(host, path, 3, 700);
                if ("prueba".equalsIgnoreCase(finalSessionName)) {
                    runOnUiThread(() -> addLog("Modo prueba: generando y descargando traza inmediatamente..."));
                    downloadAndProcessCapture(host, durationSec, finalSessionName, false);
                } else {
                    runOnUiThread(() -> {
                        addLog("Captura iniciada: " + compact(resp));
                        addLog("Cuando quieras descargar/procesar, pulsa Detener.");
                    });
                }
            } catch (Exception e) {
                runOnUiThread(() -> addLog("No se pudo iniciar captura: " + e.getMessage()));
            }
        }).start();
    }

    private void stopCapture() {
        if (!Esp32ConnectionManager.isConnected(this)) {
            addLog("No conectado al ESP32. Pulsa primero en Conectar.");
            Toast.makeText(this, "Conecta primero al ESP32", Toast.LENGTH_SHORT).show();
            return;
        }

        String host = readHost();
        int durationSec = readDuration();
        String sessionName = etEsp32SessionName.getText().toString().trim();
        if (sessionName.isEmpty()) {
            sessionName = "esp32_prueba";
        }
        final String finalSessionName = sessionName;

        addLog("Solicitando parada de captura ESP32...");
        downloadAndProcessCapture(host, durationSec, finalSessionName, true);
    }

    private void updateConnectionState() {
        tvEsp32StateScreen.setText(Esp32ConnectionManager.getStateText(this));
    }

    private String readHost() {
        String raw = etEsp32Host.getText().toString().trim();
        if (raw.isEmpty()) {
            return "192.168.4.1";
        }
        return raw;
    }

    private int readDuration() {
        String raw = etEsp32DurationSec.getText().toString().trim();
        try {
            int value = Integer.parseInt(raw);
            return Math.max(1, value);
        } catch (NumberFormatException ignored) {
            return 30;
        }
    }

    private void addLog(String line) {
        if (line == null) {
            return;
        }
        logLines.add("- " + line);
        int overflow = logLines.size() - MAX_LOG_LINES;
        if (overflow > 0) {
            logLines.subList(0, overflow).clear();
        }
        StringBuilder sb = new StringBuilder();
        for (String l : logLines) {
            sb.append(l).append('\n');
        }
        tvEsp32Log.setText(sb.toString().trim());
    }

    private String doHttpGet(String host, String path) throws IOException {
        String normalizedPath = path.startsWith("/") ? path : "/" + path;
        URL url = new URL("http://" + host + normalizedPath);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(4000);
        conn.setReadTimeout(4000);
        conn.connect();

        int code = conn.getResponseCode();
        InputStream stream = (code >= 200 && code < 300) ? conn.getInputStream() : conn.getErrorStream();
        String body = (stream == null) ? "" : readAll(stream);
        conn.disconnect();

        if (code < 200 || code >= 300) {
            throw new IOException("HTTP " + code + " " + body);
        }
        return body;
    }

    private String doHttpGetWithRetry(String host, String path, int attempts, long delayMs) throws IOException {
        IOException last = null;
        for (int i = 1; i <= attempts; i++) {
            try {
                return doHttpGet(host, path);
            } catch (IOException e) {
                last = e;
                if (i < attempts) {
                    final int retryNo = i + 1;
                    runOnUiThread(() -> addLog("Reintento HTTP " + retryNo + "/" + attempts + " para " + path));
                    try {
                        Thread.sleep(delayMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new IOException("Interrumpido durante reintento");
                    }
                }
            }
        }
        throw (last != null) ? last : new IOException("Error HTTP desconocido");
    }

    private String readAll(InputStream in) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }

    private String compact(String s) {
        if (s == null) {
            return "";
        }
        String c = s.replace('\n', ' ').trim();
        if (c.length() > 180) {
            return c.substring(0, 180) + "...";
        }
        return c;
    }

    private String urlEncodeSafe(String raw) {
        if (raw == null || raw.trim().isEmpty()) {
            return "";
        }
        return raw.trim().replace(" ", "_");
    }

    private void downloadAndProcessCapture(String host, int durationSec, String sessionName, boolean tryStopFirst) {
        new Thread(() -> {
            String stopResp = "";
            if (tryStopFirst) {
                try {
                    stopResp = doHttpGet(host, "/capture/stop");
                } catch (Exception e) {
                    final String msg = e.getMessage();
                    runOnUiThread(() -> addLog("Aviso: /capture/stop no disponible, se intenta descargar igual. " + msg));
                }
            }

            try {
                String csv = doHttpGet(host, "/capture/download");
                List<ImuSample> samples = parseCsvToSamples(csv);
                if (samples.size() < 100) {
                    throw new IOException(
                            "CSV recibido con muy pocas muestras: " + samples.size() +
                                    " | respuesta=" + compact(csv));
                }

                File saved = sessionRepository.saveSession(
                        this,
                        samples,
                        durationSec * 1000L,
                        sessionName + "_esp32"
                );
                if (saved == null) {
                    throw new IOException("No se pudo guardar la sesion ESP32");
                }

                File finalSaved = saved;
                String finalStopResp = stopResp;
                runOnUiThread(() -> {
                    if (tryStopFirst && !finalStopResp.isEmpty()) {
                        addLog("Captura detenida: " + compact(finalStopResp));
                    }
                    addLog("CSV descargado y guardado: " + finalSaved.getName() + " (" + samples.size() + " muestras)");
                    refreshSessionList();
                    processAndShowSession(finalSaved);
                });
            } catch (Exception e) {
                runOnUiThread(() -> addLog("No se pudo descargar/procesar captura ESP32: " + e.getMessage()));
            }
        }).start();
    }

    private List<ImuSample> parseCsvToSamples(String csv) {
        List<ImuSample> out = new ArrayList<>();
        if (csv == null || csv.trim().isEmpty()) {
            return out;
        }

        String[] lines = csv.split("\\r?\\n");
        for (String line : lines) {
            String row = line.trim();
            if (row.isEmpty() || row.startsWith("t_ms")) {
                continue;
            }

            String[] p = row.split(",");
            if (p.length < 7) {
                continue;
            }
            try {
                long tMs = Long.parseLong(p[0].trim());
                float ax = Float.parseFloat(p[1].trim());
                float ay = Float.parseFloat(p[2].trim());
                float az = Float.parseFloat(p[3].trim());
                float gx = Float.parseFloat(p[4].trim());
                float gy = Float.parseFloat(p[5].trim());
                float gz = Float.parseFloat(p[6].trim());
                out.add(new ImuSample(tMs, ax, ay, az, gx, gy, gz));
            } catch (NumberFormatException ignored) {
            }
        }
        return out;
    }

    private OmbProcessor.Mode getSelectedMode() {
        return rbEsp32ModeNative.isChecked() && !rbEsp32ModeOmb10.isChecked()
                ? OmbProcessor.Mode.NATIVE_FS
                : OmbProcessor.Mode.OMB_10HZ;
    }

    private void processAndShowSession(File file) {
        tvEsp32SelectedSession.setText("Procesando sesion...\n" + file.getName());

        final OmbProcessor.Mode mode = getSelectedMode();
        new Thread(() -> {
            try {
                SessionData data = sessionRepository.readSession(file);
                OmbResult result = ombProcessor.process(data, mode);

                String text = String.format(
                        Locale.US,
                        "Sesion: %s\n" +
                                "Muestras: %d | Duracion: %.2f s\n" +
                                "Modo: %s\n" +
                                "Hs: %.5f m\n" +
                                "Tz: %.5f s\n" +
                                "Tc: %.5f s\n" +
                                "Tp: %.5f s\n" +
                                "fs proc: %.3f Hz | N uniforme: %d | segmentos: %d",
                        file.getName(),
                        result.inputN,
                        result.inputDurationSec,
                        result.modeLabel,
                        result.hs,
                        result.tzSec,
                        result.tcSec,
                        result.tpSec,
                        result.processingFsHz,
                        result.uniformN,
                        result.segments
                );

                runOnUiThread(() -> {
                    tvEsp32SelectedSession.setText(text);
                    renderCharts(result);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    tvEsp32SelectedSession.setText("Error procesando sesion: " + file.getName() + "\n" + e.getMessage());
                    chartEsp32Spectrum.clear();
                    chartEsp32Series.clear();
                });
            }
        }).start();
    }

    private void refreshSessionList() {
        listedSessionFiles.clear();
        sessionsAdapter.clear();

        List<File> files = sessionRepository.listSessions(this);
        listedSessionFiles.addAll(files);

        if (files.isEmpty()) {
            tvEsp32SelectedSession.setText("No hay capturas guardadas.");
            sessionsAdapter.notifyDataSetChanged();
            return;
        }

        for (File file : files) {
            sessionsAdapter.add(formatSessionRow(file));
        }
        sessionsAdapter.notifyDataSetChanged();
    }

    private String formatSessionRow(File file) {
        return file.getName() + " (" + formatFileSize(file.length()) + ")";
    }

    private String formatFileSize(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        double kb = bytes / 1024.0;
        if (kb < 1024) {
            return String.format(Locale.US, "%.1f KB", kb);
        }
        double mb = kb / 1024.0;
        return String.format(Locale.US, "%.2f MB", mb);
    }

    private void showSessionOptions(View anchor, File file) {
        PopupMenu popup = new PopupMenu(this, anchor);
        Menu menu = popup.getMenu();
        menu.add(0, ACTION_PROCESS, 0, "Procesar");
        menu.add(0, ACTION_INFO, 1, "Ver info");
        menu.add(0, ACTION_RENAME, 2, "Renombrar");
        menu.add(0, ACTION_DELETE, 3, "Eliminar");

        popup.setOnMenuItemClickListener(item -> {
            int id = item.getItemId();
            if (id == ACTION_PROCESS) {
                processAndShowSession(file);
                return true;
            }
            if (id == ACTION_INFO) {
                showSessionInfo(file);
                return true;
            }
            if (id == ACTION_RENAME) {
                showRenameDialog(file);
                return true;
            }
            if (id == ACTION_DELETE) {
                confirmDelete(file);
                return true;
            }
            return false;
        });

        popup.show();
    }

    private void setupCharts() {
        Description d1 = new Description();
        d1.setText("S_eta(f)");
        chartEsp32Spectrum.setDescription(d1);
        chartEsp32Spectrum.setNoDataText("Sin espectro");
        chartEsp32Spectrum.getAxisRight().setEnabled(false);
        chartEsp32Spectrum.getXAxis().setTextColor(Color.DKGRAY);
        chartEsp32Spectrum.getAxisLeft().setTextColor(Color.DKGRAY);
        chartEsp32Spectrum.getLegend().setTextColor(Color.DKGRAY);

        Description d2 = new Description();
        d2.setText("az dinamica (preview)");
        chartEsp32Series.setDescription(d2);
        chartEsp32Series.setNoDataText("Sin serie temporal");
        chartEsp32Series.getAxisRight().setEnabled(false);
        chartEsp32Series.getXAxis().setTextColor(Color.DKGRAY);
        chartEsp32Series.getAxisLeft().setTextColor(Color.DKGRAY);
        chartEsp32Series.getLegend().setTextColor(Color.DKGRAY);
    }

    private void renderCharts(OmbResult result) {
        List<Entry> spectrumEntries = new ArrayList<>();
        for (int i = 0; i < result.spectrumFreqHz.length; i++) {
            spectrumEntries.add(new Entry((float) result.spectrumFreqHz[i], (float) result.spectrumElevation[i]));
        }

        LineDataSet spectrumSet = new LineDataSet(spectrumEntries, "S_eta(f) [m^2/Hz]");
        spectrumSet.setColor(Color.parseColor("#1565C0"));
        spectrumSet.setLineWidth(2f);
        spectrumSet.setDrawCircles(false);
        spectrumSet.setDrawValues(false);
        chartEsp32Spectrum.setData(new LineData(spectrumSet));
        chartEsp32Spectrum.invalidate();

        List<Entry> seriesEntries = new ArrayList<>();
        for (int i = 0; i < result.previewTimeSec.length; i++) {
            seriesEntries.add(new Entry((float) result.previewTimeSec[i], (float) result.previewAzDyn[i]));
        }

        LineDataSet seriesSet = new LineDataSet(seriesEntries, "az_dyn [m/s^2]");
        seriesSet.setColor(Color.parseColor("#2E7D32"));
        seriesSet.setLineWidth(1.8f);
        seriesSet.setDrawCircles(false);
        seriesSet.setDrawValues(false);
        chartEsp32Series.setData(new LineData(seriesSet));
        chartEsp32Series.invalidate();
    }

    private void showSessionInfo(File file) {
        String info = String.format(
                Locale.US,
                "Nombre: %s\nTamano: %s\nUltima modificacion: %s\nRuta: %s",
                file.getName(),
                formatFileSize(file.length()),
                new Date(file.lastModified()).toString(),
                file.getAbsolutePath()
        );

        new AlertDialog.Builder(this)
                .setTitle("Info de sesion")
                .setMessage(info)
                .setPositiveButton("OK", null)
                .show();
    }

    private void showRenameDialog(File file) {
        EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        String current = file.getName().toLowerCase(Locale.US).endsWith(".csv")
                ? file.getName().substring(0, file.getName().length() - 4)
                : file.getName();
        input.setText(current);
        input.setSelection(input.getText().length());

        new AlertDialog.Builder(this)
                .setTitle("Renombrar sesion")
                .setView(input)
                .setPositiveButton("Guardar", (dialog, which) -> {
                    try {
                        sessionRepository.renameSession(file, input.getText().toString());
                        refreshSessionList();
                    } catch (IOException e) {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }

    private void confirmDelete(File file) {
        new AlertDialog.Builder(this)
                .setTitle("Eliminar sesion")
                .setMessage("Seguro que quieres eliminar:\n" + file.getName() + "?")
                .setPositiveButton("Eliminar", (dialog, which) -> {
                    try {
                        sessionRepository.deleteSession(file);
                        refreshSessionList();
                        tvEsp32SelectedSession.setText("Sesion eliminada: " + file.getName());
                    } catch (IOException e) {
                        Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                    }
                })
                .setNegativeButton("Cancelar", null)
                .show();
    }
}
