package tfg.udc.boya;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.net.wifi.WifiNetworkSpecifier;
import android.os.Build;

import java.util.Locale;

public final class Esp32ConnectionManager {

    public interface ConnectCallback {
        void onResult(boolean ok, String message);
    }

    public static final String SSID = "BOYA_ESP32";
    public static final String PASSWORD = "boya1234";

    private static Network esp32Network;
    private static ConnectivityManager.NetworkCallback networkCallback;

    private Esp32ConnectionManager() {
    }

    public static boolean isConnected(Context context) {
        if (esp32Network != null) {
            return true;
        }

        try {
            WifiManager wifiManager = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wifiManager == null) {
                return false;
            }
            WifiInfo info = wifiManager.getConnectionInfo();
            if (info == null) {
                return false;
            }
            String ssid = normalizeSsid(info.getSSID());
            return SSID.equals(ssid);
        } catch (Exception ignored) {
            return false;
        }
    }

    public static String getStateText(Context context) {
        return isConnected(context) ? "Conectado al ESP32" : "No conectado al ESP32";
    }

    public static void connect(Context context, ConnectCallback callback) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            callback.onResult(false, "Android < 10: requiere conexion manual en ajustes Wi-Fi");
            return;
        }

        disconnect(context);

        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm == null) {
            callback.onResult(false, "No se pudo obtener ConnectivityManager");
            return;
        }

        WifiNetworkSpecifier specifier = new WifiNetworkSpecifier.Builder()
                .setSsid(SSID)
                .setWpa2Passphrase(PASSWORD)
                .build();

        NetworkRequest request = new NetworkRequest.Builder()
                .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
                .setNetworkSpecifier(specifier)
                .build();

        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                esp32Network = network;
                cm.bindProcessToNetwork(network);
                callback.onResult(true, "Conectado a " + SSID);
            }

            @Override
            public void onUnavailable() {
                callback.onResult(false, "No se pudo conectar al ESP32");
            }

            @Override
            public void onLost(Network network) {
                if (esp32Network != null && esp32Network.equals(network)) {
                    esp32Network = null;
                    cm.bindProcessToNetwork(null);
                }
            }
        };

        cm.requestNetwork(request, networkCallback, 20000);
    }

    public static void disconnect(Context context) {
        ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm == null) {
            return;
        }

        if (networkCallback != null) {
            try {
                cm.unregisterNetworkCallback(networkCallback);
            } catch (Exception ignored) {
            }
            networkCallback = null;
        }

        cm.bindProcessToNetwork(null);
        esp32Network = null;
    }

    private static String normalizeSsid(String raw) {
        if (raw == null) {
            return "";
        }
        String ssid = raw.trim();
        if (ssid.startsWith("\"") && ssid.endsWith("\"") && ssid.length() >= 2) {
            ssid = ssid.substring(1, ssid.length() - 1);
        }
        return ssid.toUpperCase(Locale.US).equals("<UNKNOWN SSID>") ? "" : ssid;
    }
}
