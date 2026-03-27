package tfg.udc.boya;

import android.content.Context;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class SessionRepository {

    public File getSessionsDir(Context context) {
        File dir = new File(context.getExternalFilesDir(null), "sessions");
        if (!dir.exists() && !dir.mkdirs()) {
            return null;
        }
        return dir;
    }

    public File saveSession(Context context, List<ImuSample> samples, long captureDurationMs, String customName) throws IOException {
        if (samples == null || samples.isEmpty()) {
            return null;
        }

        File dir = getSessionsDir(context);
        if (dir == null) {
            throw new IOException("No se pudo crear la carpeta de sesiones");
        }

        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String safeName = sanitizeBaseName(customName);
        String fileName;
        if (safeName.isEmpty()) {
            fileName = String.format(Locale.US, "session_%s_%ds.csv", timestamp, captureDurationMs / 1000L);
        } else {
            fileName = String.format(Locale.US, "%s_%s_%ds.csv", safeName, timestamp, captureDurationMs / 1000L);
        }
        File outFile = new File(dir, fileName);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outFile))) {
            writer.write("t_ms,ax,ay,az,gx,gy,gz\n");
            for (ImuSample s : samples) {
                writer.write(String.format(
                        Locale.US,
                        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        s.tMs, s.ax, s.ay, s.az, s.gx, s.gy, s.gz
                ));
            }
            writer.flush();
        }

        return outFile;
    }

    public File renameSession(File file, String newBaseName) throws IOException {
        if (file == null || !file.exists()) {
            throw new IOException("El archivo no existe");
        }

        String safeName = sanitizeBaseName(newBaseName);
        if (safeName.isEmpty()) {
            throw new IOException("Nombre invalido");
        }

        File parent = file.getParentFile();
        if (parent == null) {
            throw new IOException("Ruta invalida");
        }

        String targetName = safeName + ".csv";
        File target = new File(parent, targetName);
        if (target.exists()) {
            throw new IOException("Ya existe una sesion con ese nombre");
        }

        boolean ok = file.renameTo(target);
        if (!ok) {
            throw new IOException("No se pudo renombrar el archivo");
        }

        return target;
    }

    public void deleteSession(File file) throws IOException {
        if (file == null || !file.exists()) {
            throw new IOException("El archivo no existe");
        }
        if (!file.delete()) {
            throw new IOException("No se pudo eliminar el archivo");
        }
    }

    public List<File> listSessions(Context context) {
        List<File> out = new ArrayList<>();
        File dir = getSessionsDir(context);
        if (dir == null) {
            return out;
        }

        File[] files = dir.listFiles((d, name) -> name.toLowerCase(Locale.US).endsWith(".csv"));
        if (files == null || files.length == 0) {
            return out;
        }

        Arrays.sort(files, Comparator.comparingLong(File::lastModified).reversed());
        out.addAll(Arrays.asList(files));
        return out;
    }

    public SessionData readSession(File file) throws IOException {
        List<Double> tList = new ArrayList<>();
        List<Double> azList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            if (line == null) {
                throw new IOException("CSV vacio");
            }

            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                if (parts.length < 4) {
                    continue;
                }

                try {
                    double tMs = Double.parseDouble(parts[0].trim());
                    double azValue = Double.parseDouble(parts[3].trim());
                    tList.add(tMs / 1000.0);
                    azList.add(azValue);
                } catch (NumberFormatException ignored) {
                }
            }
        }

        if (tList.size() < 100) {
            throw new IOException("No hay suficientes muestras validas");
        }

        double[] t = new double[tList.size()];
        double[] az = new double[azList.size()];

        int valid = 0;
        double prevT = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < tList.size(); i++) {
            double ti = tList.get(i);
            double azi = azList.get(i);
            if (!Double.isFinite(ti) || !Double.isFinite(azi)) {
                continue;
            }
            if (ti <= prevT) {
                continue;
            }
            t[valid] = ti;
            az[valid] = azi;
            prevT = ti;
            valid++;
        }

        if (valid < 100) {
            throw new IOException("No hay suficientes muestras con timestamp creciente");
        }

        double t0 = t[0];
        double[] tOut = new double[valid];
        double[] azOut = new double[valid];
        for (int i = 0; i < valid; i++) {
            tOut[i] = t[i] - t0;
            azOut[i] = az[i];
        }

        return new SessionData(tOut, azOut);
    }

    private String sanitizeBaseName(String raw) {
        if (raw == null) {
            return "";
        }
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) {
            return "";
        }

        // Solo caracteres seguros para nombre de archivo.
        String safe = trimmed
                .replaceAll("[^a-zA-Z0-9_\\- ]", "")
                .replace(' ', '_')
                .replaceAll("_+", "_");

        if (safe.startsWith("_")) {
            safe = safe.substring(1);
        }
        if (safe.endsWith("_")) {
            safe = safe.substring(0, safe.length() - 1);
        }

        // Evita nombres excesivos.
        if (safe.length() > 64) {
            safe = safe.substring(0, 64);
        }

        return safe;
    }
}
