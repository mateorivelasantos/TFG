set(MAIN_SRC "main.cpp")
if(DEFINED ENV{BOYA_DIAG_IMU} AND "$ENV{BOYA_DIAG_IMU}" STREQUAL "1")
    set(MAIN_SRC "main_diag_imu.cpp")
endif()

idf_component_register(
    SRCS "${MAIN_SRC}"
    INCLUDE_DIRS "."
    REQUIRES esp_wifi esp_netif esp_event esp_http_server nvs_flash driver esp_driver_gpio esp_timer
)
