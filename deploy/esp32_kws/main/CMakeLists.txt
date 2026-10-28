idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Milestone 2 (docs/ONCHIP.md): -DKWS_APP=eval builds the UART evaluation server
# (app_eval.cpp, this board's version of deploy/onchip/eval_firmware/app_eval.cpp) in
# place of app_main.cpp. Give it its own build directory:
#   idf.py -B build_eval -DKWS_APP=eval build
if(KWS_APP STREQUAL "eval")
    set(app_src app_eval.cpp)
else()
    set(app_src app_main.cpp)
endif()

idf_component_register(SRCS ${app_src}
                       INCLUDE_DIRS . ../../onchip/bcresnet-2
                       REQUIRES esp-dl esp_timer esp_driver_uart)

# Model stays in flash .rodata (symbol _binary_model_espdl_start).
target_add_aligned_binary_data(${COMPONENT_LIB} ${CMAKE_CURRENT_SOURCE_DIR}/models/model.espdl BINARY)
