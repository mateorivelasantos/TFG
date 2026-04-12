set(MAIN_SRC "main.cpp")
if(DEFINED ENV{BOYA_DIAG_IMU} AND "$ENV{BOYA_DIAG_IMU}" STREQUAL "1")
    set(MAIN_SRC "main_diag_imu.cpp")
endif()

set(BOYA_REQUIRES
    esp_wifi
    esp_netif
    esp_event
    esp_http_server
    nvs_flash
    driver
    esp_timer
)

# `esp_driver_gpio` existe en versiones recientes de ESP-IDF.
if(EXISTS "${IDF_PATH}/components/esp_driver_gpio")
    list(APPEND BOYA_REQUIRES esp_driver_gpio)
endif()

idf_component_register(
    SRCS "${MAIN_SRC}"
    INCLUDE_DIRS "."
    REQUIRES ${BOYA_REQUIRES}
)
